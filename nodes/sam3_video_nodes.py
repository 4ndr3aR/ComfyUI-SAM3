"""
SAM3 Video Tracking Nodes for ComfyUI - Stateless Architecture

These nodes provide video object tracking and segmentation using SAM3.
All state is encoded in immutable outputs - no global mutable state.

Key design principles:
1. All nodes are stateless - state flows through outputs
2. SAM3VideoState is immutable - adding prompts returns NEW state
3. Inference state is reconstructed on-demand
4. Temp directories are automatically cleaned up at process exit
5. No manual SAM3CloseVideoSession needed
"""
import gc
import os
import time
import threading
import datetime
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, wait as futures_wait, ALL_COMPLETED


# =============================================================================
# Logging helpers — timestamped + ANSI color
# =============================================================================
class _C:
    """ANSI color codes."""
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    BLUE   = "\033[94m"
    CYAN   = "\033[96m"
    WHITE  = "\033[97m"
    GREY   = "\033[90m"
    BG_RED = "\033[41m"


def _ts() -> str:
    """Return a fixed-width timestamp: '2026-03-13 16.27.34'"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H.%M.%S")


def _log(msg: str, color: str = "") -> None:
    """Print with timestamp prefix and optional ANSI color."""
    reset = _C.RESET if color else ""
    print(f"{_C.GREY}{_ts()}{_C.RESET} {color}{msg}{reset}")


def _hi(value, color: str = _C.CYAN) -> str:
    """Wrap a value in bold color for inline highlighting."""
    return f"{_C.BOLD}{color}{value}{_C.RESET}"

import folder_paths
import comfy.model_management

from .video_state import (
    SAM3VideoState,
    VideoPrompt,
    VideoConfig,
    create_video_state,
    cleanup_temp_dir,
)
from .inference_reconstructor import (
    get_inference_state,
    invalidate_session,
    clear_inference_cache,
)
from .sam3_model_patcher import SAM3ModelWrapper, SAM3ModelPatcher


# =============================================================================
# Autocast dtype detection - handles GPUs without bf16 support
# =============================================================================
def _get_autocast_dtype():
    """
    Get appropriate autocast dtype based on GPU capability.
    Returns None if autocast should not be used.
    """
    if not torch.cuda.is_available():
        return None
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:  # Ampere+ supports bf16
        return torch.bfloat16
    elif major >= 7:  # Volta/Turing use fp16
        return torch.float16
    else:
        return None  # Older GPUs - no autocast


def _get_autocast_context():
    """Get autocast context manager based on GPU capability."""
    dtype = _get_autocast_dtype()
    if dtype is not None:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return torch.no_grad()


# =============================================================================
# VRAM Debug Utility
# =============================================================================

def print_vram(label: str, detailed: bool = False):
    """Print current VRAM usage for debugging memory leaks."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        _log(f"[VRAM] {label}: {_hi(f'{alloc:.2f}GB', _C.YELLOW)} allocated, "
             f"{_hi(f'{reserved:.2f}GB', _C.YELLOW)} reserved")
        if detailed:
            # Print memory stats breakdown
            stats = torch.cuda.memory_stats()
            _log(f"[VRAM]   Active:            {stats.get('active_bytes.all.current', 0) / 1024**3:.2f}GB")
            _log(f"[VRAM]   Inactive:          {stats.get('inactive_split_bytes.all.current', 0) / 1024**3:.2f}GB")
            _log(f"[VRAM]   Allocated retries: {stats.get('num_alloc_retries', 0)}")


# =============================================================================
# Video Segmentation Nodes
# =============================================================================
# NOTE: SAM3VideoModelLoader has been removed.
# Use LoadSAM3Model instead - it returns a unified model that works for both
# image segmentation and video tracking.


# =============================================================================
# Video Segmentation (Unified Node)
# =============================================================================

class SAM3VideoSegmentation:
    """
    Initialize video tracking and add prompts.

    Select prompt_mode to choose between:
    - text: Track objects by text description (comma-separated for multiple)
    - point: Track objects by clicking points (positive/negative)
    - box: Track objects by drawing boxes (positive/negative)

    Note: SAM3 video does NOT support combining different prompt types.
    Each mode is mutually exclusive.
    """
    # Class-level cache for video state results
    _cache = {}

    PROMPT_MODES = ["text", "point", "box"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_frames": ("IMAGE", {
                    "tooltip": "Video frames as batch of images [N, H, W, C]"
                }),
                "prompt_mode": (cls.PROMPT_MODES, {
                    "default": "text",
                    "tooltip": "Prompt type: text (describe objects), point (click on objects), or box (draw rectangles)"
                }),
            },
            "optional": {
                # Text mode inputs
                "text_prompt": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "[text mode] Text description(s) to track. Comma-separated for multiple objects (e.g., 'person, dog, car')"
                }),
                # Point mode inputs
                "positive_points": ("SAM3_POINTS_PROMPT", {
                    "tooltip": "[point mode] Positive points - click on objects to track"
                }),
                "negative_points": ("SAM3_POINTS_PROMPT", {
                    "tooltip": "[point mode] Negative points - click on areas to exclude"
                }),
                # Box mode inputs
                "positive_boxes": ("SAM3_BOXES_PROMPT", {
                    "tooltip": "[box mode] Positive boxes - draw around objects to track"
                }),
                "negative_boxes": ("SAM3_BOXES_PROMPT", {
                    "tooltip": "[box mode] Negative boxes - draw around areas to exclude"
                }),
                # Common inputs
                "frame_idx": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Frame index to apply prompts (usually 0 for first frame)"
                }),
                "score_threshold": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Detection confidence threshold"
                }),
            }
        }

    @classmethod
    def IS_CHANGED(cls, video_frames, prompt_mode="text", text_prompt="",
                   positive_points=None, negative_points=None,
                   positive_boxes=None, negative_boxes=None,
                   frame_idx=0, score_threshold=0.3):
        # Use a stable hash based on video content
        # Don't use float(mean()) - it has floating point precision issues on GPU
        import hashlib

        # Create a stable hash from video frame content
        # Use shape + corner pixels from first and last frame (deterministic bytes, no float issues)
        h = hashlib.md5()
        h.update(str(video_frames.shape).encode())

        # Sample corner pixels from first and last frame
        first_frame = video_frames[0].cpu().numpy()
        last_frame = video_frames[-1].cpu().numpy()
        h.update(first_frame[0, 0, :].tobytes())      # top-left
        h.update(first_frame[-1, -1, :].tobytes())    # bottom-right
        h.update(last_frame[0, 0, :].tobytes())
        h.update(last_frame[-1, -1, :].tobytes())

        video_hash = h.hexdigest()

        result = hash((
            video_hash,
            prompt_mode,
            text_prompt,
            str(positive_points),
            str(negative_points),
            str(positive_boxes),
            str(negative_boxes),
            frame_idx,
            score_threshold,
        ))
        print(f"[IS_CHANGED DEBUG] SAM3VideoSegmentation: video_hash={video_hash}, prompt_mode={prompt_mode}")
        print(f"[IS_CHANGED DEBUG] SAM3VideoSegmentation: positive_points={positive_points}")
        print(f"[IS_CHANGED DEBUG] SAM3VideoSegmentation: negative_points={negative_points}")
        print(f"[IS_CHANGED DEBUG] SAM3VideoSegmentation: returning hash={result}")
        return result

    RETURN_TYPES = ("SAM3_VIDEO_STATE",)
    RETURN_NAMES = ("video_state",)
    FUNCTION = "segment"
    CATEGORY = "SAM3/video"

    def segment(self, video_frames, prompt_mode="text", text_prompt="",
                positive_points=None, negative_points=None,
                positive_boxes=None, negative_boxes=None,
                frame_idx=0, score_threshold=0.3):
        """Initialize video state and add prompts based on selected mode."""
        # Create cache key from inputs
        import hashlib
        h = hashlib.md5()
        h.update(str(video_frames.shape).encode())
        # Sample corner pixels for video identity
        first_frame = video_frames[0].cpu().numpy()
        last_frame = video_frames[-1].cpu().numpy()
        h.update(first_frame[0, 0, :].tobytes())
        h.update(first_frame[-1, -1, :].tobytes())
        h.update(last_frame[0, 0, :].tobytes())
        h.update(last_frame[-1, -1, :].tobytes())
        h.update(prompt_mode.encode())
        h.update(text_prompt.encode())
        h.update(str(id(positive_points)).encode() if positive_points else b"none")
        h.update(str(id(negative_points)).encode() if negative_points else b"none")
        h.update(str(id(positive_boxes)).encode() if positive_boxes else b"none")
        h.update(str(id(negative_boxes)).encode() if negative_boxes else b"none")
        h.update(str(frame_idx).encode())
        h.update(str(score_threshold).encode())
        cache_key = h.hexdigest()

        # Check if we have cached result
        if cache_key in SAM3VideoSegmentation._cache:
            cached = SAM3VideoSegmentation._cache[cache_key]
            _log(f"[SAM3 Video] CACHE HIT — returning cached video_state for key={cache_key[:8]}, session={cached.session_uuid[:8]}", _C.GREEN)
            return (cached,)

        _log(f"[SAM3 Video] CACHE MISS — computing new video_state for key={cache_key[:8]}")
        print_vram("Before video segmentation")

        # 1. Initialize video state
        config = VideoConfig(
            score_threshold_detection=score_threshold,
        )
        video_state = create_video_state(
            video_frames=video_frames,
            config=config,
        )

        _log(f"[SAM3 Video] Initialized session {_hi(video_state.session_uuid[:8])}")
        _log(f"[SAM3 Video] Frames: {_hi(video_state.num_frames)}, Size: {_hi(f'{video_state.width}x{video_state.height}')}")
        _log(f"[SAM3 Video] Prompt mode: {_hi(prompt_mode, _C.YELLOW)}")

        # 2. Add prompts based on mode (mutually exclusive)
        obj_id = 1

        if prompt_mode == "text":
            # Text mode: parse comma-separated text prompts
            if text_prompt and text_prompt.strip():
                for text in text_prompt.split(","):
                    text = text.strip()
                    if text:
                        prompt = VideoPrompt.create_text(frame_idx, obj_id, text)
                        video_state = video_state.with_prompt(prompt)
                        _log(f"[SAM3 Video] Added text prompt: obj={_hi(obj_id)}, text='{_hi(text, _C.YELLOW)}'")
                        obj_id += 1
            else:
                _log("[SAM3 Video] Warning: text mode selected but no text_prompt provided", _C.YELLOW)

        elif prompt_mode == "point":
            # Point mode: combine positive and negative points
            all_points = []
            all_labels = []

            if positive_points and positive_points.get("points"):
                for pt in positive_points["points"]:
                    all_points.append([float(pt[0]), float(pt[1])])
                    all_labels.append(1)  # Positive

            if negative_points and negative_points.get("points"):
                for pt in negative_points["points"]:
                    all_points.append([float(pt[0]), float(pt[1])])
                    all_labels.append(0)  # Negative

            if all_points:
                prompt = VideoPrompt.create_point(frame_idx, obj_id, all_points, all_labels)
                video_state = video_state.with_prompt(prompt)
                pos_count = len(positive_points.get("points", [])) if positive_points else 0
                neg_count = len(negative_points.get("points", [])) if negative_points else 0
                _log(f"[SAM3 Video] Added point prompt: obj={_hi(obj_id)}, "
                      f"positive={_hi(pos_count, _C.GREEN)}, negative={_hi(neg_count, _C.RED)}")
            else:
                _log("[SAM3 Video] Warning: point mode selected but no points provided", _C.YELLOW)

        elif prompt_mode == "box":
            # Box mode: add positive and/or negative boxes
            has_boxes = False

            if positive_boxes and positive_boxes.get("boxes"):
                box_data = positive_boxes["boxes"][0]  # First box
                cx, cy, w, h = box_data
                x1 = cx - w/2
                y1 = cy - h/2
                x2 = cx + w/2
                y2 = cy + h/2
                prompt = VideoPrompt.create_box(frame_idx, obj_id, [x1, y1, x2, y2], is_positive=True)
                video_state = video_state.with_prompt(prompt)
                _log(f"[SAM3 Video] Added positive box: obj={_hi(obj_id)}, "
                      f"box=[{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}]")
                has_boxes = True

            if negative_boxes and negative_boxes.get("boxes"):
                box_data = negative_boxes["boxes"][0]  # First box
                cx, cy, w, h = box_data
                x1 = cx - w/2
                y1 = cy - h/2
                x2 = cx + w/2
                y2 = cy + h/2
                prompt = VideoPrompt.create_box(frame_idx, obj_id, [x1, y1, x2, y2], is_positive=False)
                video_state = video_state.with_prompt(prompt)
                _log(f"[SAM3 Video] Added negative box: obj={_hi(obj_id)}, "
                      f"box=[{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}]")
                has_boxes = True

            if not has_boxes:
                _log("[SAM3 Video] Warning: box mode selected but no boxes provided", _C.YELLOW)

        # Validate at least one prompt was added
        if len(video_state.prompts) == 0:
            _log(f"[SAM3 Video] Warning: No prompts added for mode '{prompt_mode}'", _C.YELLOW)

        _log(f"[SAM3 Video] Total prompts: {_hi(len(video_state.prompts))}")
        print_vram("After video segmentation")

        # Cache the result
        SAM3VideoSegmentation._cache[cache_key] = video_state

        return (video_state,)


# =============================================================================
# Propagation
# =============================================================================

class SAM3Propagate:
    """
    Run video propagation to track objects across frames.

    Reconstructs inference state on-demand from immutable video state.
    """
    # Class-level cache for propagation results
    _cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam3_model": ("SAM3_MODEL", {
                    "tooltip": "SAM3 model (from LoadSAM3Model)"
                }),
                "video_state": ("SAM3_VIDEO_STATE", {
                    "tooltip": "Video state with prompts"
                }),
            },
            "optional": {
                "start_frame": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Start frame for propagation"
                }),
                "end_frame": ("INT", {
                    "default": -1,
                    "min": -1,
                    "tooltip": "End frame (-1 for all)"
                }),
                "direction": (["forward", "backward", "both"], {
                    "default": "forward",
                    "tooltip": "Propagation direction: forward (future frames), backward (past frames), or both directions"
                }),
                "offload_model": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Move model to CPU after propagation to free VRAM (slower next run)"
                }),
            }
        }

    RETURN_TYPES = ("SAM3_VIDEO_MASKS", "SAM3_VIDEO_SCORES", "SAM3_VIDEO_STATE")
    RETURN_NAMES = ("masks", "scores", "video_state")
    FUNCTION = "propagate"
    CATEGORY = "SAM3/video"

    @classmethod
    def IS_CHANGED(cls, sam3_model, video_state, start_frame=0, end_frame=-1, direction="forward", offload_model=False):
        # Use object identity for caching - if upstream node is cached,
        # it returns the same object, so id() will match
        # This is more reliable than hashing content since video_state is immutable
        result = (id(video_state), start_frame, end_frame, direction)
        print(f"[IS_CHANGED DEBUG] SAM3Propagate: video_state id={id(video_state)}, session={video_state.session_uuid if video_state else None}")
        print(f"[IS_CHANGED DEBUG] SAM3Propagate: returning {result}")
        return result

    def propagate(self, sam3_model, video_state, start_frame=0, end_frame=-1, direction="forward", offload_model=False):
        """Run propagation using reconstructed inference state."""
        # Create cache key using video_state object id (since it's immutable and cached upstream)
        cache_key = (id(video_state), start_frame, end_frame, direction)

        # Check if we have cached result
        if cache_key in SAM3Propagate._cache:
            cached = SAM3Propagate._cache[cache_key]
            _log(f"[SAM3 Propagate] CACHE HIT — returning cached result for session={_hi(video_state.session_uuid[:8])}", _C.GREEN)
            # Still need to handle offload if requested
            if offload_model:
                _log("[SAM3 Video] Offloading model to CPU to free VRAM...", _C.YELLOW)
                if hasattr(sam3_model, 'model'):
                    sam3_model.model.cpu()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print_vram("After model offload")
            return cached

        _log(f"[SAM3 Propagate] CACHE MISS — running propagation for session={_hi(video_state.session_uuid[:8])}")

        if len(video_state.prompts) == 0:
            raise ValueError("[SAM3 Video] No prompts added. Add point, box, or text prompts before propagating.")

        # Ensure model is on GPU before inference (may have been offloaded)
        if hasattr(sam3_model, 'model') and hasattr(sam3_model.model, 'to'):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            sam3_model.model.to(device)

        end_label = end_frame if end_frame >= 0 else 'end'
        _log(f"[SAM3 Video] Starting propagation: frames {_hi(start_frame)} → {_hi(end_label)}, "
             f"direction={_hi(direction, _C.YELLOW)}")
        _log(f"[SAM3 Video] Prompts: {_hi(len(video_state.prompts))}")
        print_vram("Before propagation start")

        # Determine frame range
        if end_frame < 0:
            end_frame = video_state.num_frames - 1

        # Build propagation request - uses predictor's handle_stream_request API
        # direction is already "forward", "backward", or "both"
        request = {
            "type": "propagate_in_video",
            "session_id": video_state.session_uuid,
            "propagation_direction": direction,
            "start_frame_index": start_frame,
            "max_frame_num_to_track": end_frame - start_frame + 1,
        }

        # Run ALL inference inside autocast context for dtype consistency
        # SAM3 requires bf16/fp16 - wrap reconstruction AND propagation
        masks_dict = {}
        scores_dict = {}
        # Use autocast with dtype based on GPU capability (bf16 for Ampere+, fp16 for Volta/Turing)
        autocast_context = _get_autocast_context()
        with autocast_context:
            print_vram("Before reconstruction (in autocast)")
            # Reconstruct inference state from immutable state
            inference_state = get_inference_state(sam3_model, video_state)
            print_vram("After reconstruction")

            # Run propagation
            try:
                for response in sam3_model.handle_stream_request(request):
                    frame_idx = response.get("frame_index", response.get("frame_idx"))
                    if frame_idx is None:
                        continue

                    outputs = response.get("outputs", response)
                    if outputs is None:
                        continue

                    # Try different possible mask keys
                    mask_key = None
                    for key in ["out_binary_masks", "video_res_masks", "masks"]:
                        if key in outputs and outputs[key] is not None:
                            mask_key = key
                            break

                    if mask_key:
                        # Move masks to CPU immediately to free GPU memory
                        mask = outputs[mask_key]
                        if hasattr(mask, 'cpu'):
                            mask = mask.cpu()
                        masks_dict[frame_idx] = mask

                    # Capture confidence scores
                    for score_key in ["out_probs", "scores", "confidences", "obj_scores"]:
                        if score_key in outputs and outputs[score_key] is not None:
                            probs = outputs[score_key]
                            if hasattr(probs, 'cpu'):
                                probs = probs.cpu()
                            elif isinstance(probs, np.ndarray):
                                probs = torch.from_numpy(probs)
                            scores_dict[frame_idx] = probs
                            break

                    # Periodic cleanup and VRAM monitoring
                    if frame_idx % 10 == 0:
                        print_vram(f"Frame {frame_idx}")
                        gc.collect()

            except Exception as e:
                _log(f"[SAM3 Video] Propagation error: {e}", _C.RED)
                import traceback
                traceback.print_exc()
                raise

        print_vram("After propagation loop")
        _log(f"[SAM3 Video] Propagation complete: {_hi(len(masks_dict))} frames processed")
        _log(f"[SAM3 Video] Frames with scores: {_hi(len(scores_dict))}")

        # Clean up
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Offload model to CPU if requested (Issue #28)
        if offload_model:
            _log("[SAM3 Video] Offloading model to CPU to free VRAM...", _C.YELLOW)
            if hasattr(sam3_model, 'model'):
                sam3_model.model.cpu()
            # Clear inference state cache to free GPU memory
            from .sam3_lib.sam3_video_predictor import Sam3VideoPredictor
            Sam3VideoPredictor._ALL_INFERENCE_STATES.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print_vram("After model offload")

        # Cache the result
        result = (masks_dict, scores_dict, video_state)
        SAM3Propagate._cache[cache_key] = result

        return result


# =============================================================================
# Output Extraction
# =============================================================================

class SAM3VideoOutput:
    """
    Extract masks from propagation results.

    Converts SAM3_VIDEO_MASKS to ComfyUI-compatible mask tensors.
    Returns all frames as a batch.

    Changing obj_id does NOT re-run propagation - only this node re-executes.
    """
    # Class-level cache for extraction results
    _cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("SAM3_VIDEO_MASKS", {
                    "tooltip": "Masks from SAM3Propagate"
                }),
                "video_state": ("SAM3_VIDEO_STATE", {
                    "tooltip": "Video state for dimensions"
                }),
            },
            "optional": {
                "scores": ("SAM3_VIDEO_SCORES", {
                    "tooltip": "Confidence scores from SAM3Propagate"
                }),
                "obj_id": ("INT", {
                    "default": -1,
                    "min": -1,
                    "tooltip": "Specific object ID for mask output (-1 for all combined). Changing this is fast - no re-inference needed."
                }),
                "plot_all_masks": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show all object masks in visualization (True) or only selected obj_id (False)"
                }),
            }
        }

    @classmethod
    def IS_CHANGED(cls, masks, video_state, scores=None, obj_id=-1, plot_all_masks=True):
        # Always re-run this node when params change, but this is cheap
        # The key is that changing these here does NOT invalidate upstream cache
        # ComfyUI caches based on input values - masks/video_state don't change
        return (id(masks), video_state.session_uuid, id(scores), obj_id, plot_all_masks)

    RETURN_TYPES = ("MASK", "IMAGE", "IMAGE")
    RETURN_NAMES = ("masks", "frames", "visualization")
    FUNCTION = "extract"
    CATEGORY = "SAM3/video"

    # Pre-built 3x5 font bitmaps as numpy arrays (shape [5, 3]), computed once at class level.
    # Each array is True where the pixel should be white.
    _FONT = {c: np.array(pat, dtype=bool) for c, pat in {
        '0': [[1,1,1],[1,0,1],[1,0,1],[1,0,1],[1,1,1]],
        '1': [[0,1,0],[1,1,0],[0,1,0],[0,1,0],[1,1,1]],
        '2': [[1,1,1],[0,0,1],[1,1,1],[1,0,0],[1,1,1]],
        '3': [[1,1,1],[0,0,1],[1,1,1],[0,0,1],[1,1,1]],
        '4': [[1,0,1],[1,0,1],[1,1,1],[0,0,1],[0,0,1]],
        '5': [[1,1,1],[1,0,0],[1,1,1],[0,0,1],[1,1,1]],
        '6': [[1,1,1],[1,0,0],[1,1,1],[1,0,1],[1,1,1]],
        '7': [[1,1,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]],
        '8': [[1,1,1],[1,0,1],[1,1,1],[1,0,1],[1,1,1]],
        '9': [[1,1,1],[1,0,1],[1,1,1],[0,0,1],[1,1,1]],
        ':': [[0,0,0],[0,1,0],[0,0,0],[0,1,0],[0,0,0]],
        '.': [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,1,0]],
    }.items()}

    def _draw_legend(self, vis_frame, num_objects, colors, obj_id=-1, frame_scores=None):
        """Draw a legend — fully vectorized with numpy slices, no Python pixel loops."""
        # Work on a numpy array to avoid GIL-holding torch scalar assignments
        frame_np = vis_frame.numpy()   # zero-copy view of the underlying storage
        h, w = frame_np.shape[:2]

        # Legend parameters
        box_size = max(16, min(32, h // 20))
        padding = max(4, box_size // 4)
        text_width = box_size * 6
        legend_item_height = box_size + padding

        # Build list of (obj_id, score) pairs
        if obj_id >= 0:
            items = [(obj_id, frame_scores[obj_id] if frame_scores is not None and obj_id < len(frame_scores) else None)]
        else:
            items = [(oid, frame_scores[oid] if frame_scores is not None and oid < len(frame_scores) else None)
                     for oid in range(num_objects)]
            items.sort(key=lambda x: (x[1] is None, -(x[1] if x[1] is not None else 0)))

        num_items = len(items)
        legend_height = num_items * legend_item_height + padding
        legend_width  = box_size + text_width + padding * 2

        start_x, start_y = padding, padding
        y1 = start_y
        y2 = min(start_y + legend_height, h)
        x1 = start_x
        x2 = min(start_x + legend_width, w)

        # Semi-transparent dark background — one vectorized multiply over the slice
        bg_color = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        frame_np[y1:y2, x1:x2] = (frame_np[y1:y2, x1:x2] * 0.3 + bg_color * 0.7)

        for idx, (oid, score) in enumerate(items):
            item_y = start_y + padding + idx * legend_item_height
            iy1 = item_y
            iy2 = min(item_y + box_size, h)
            cx1 = start_x + padding
            cx2 = min(cx1 + box_size, w)

            # Solid color swatch — single slice assignment
            color_np = np.array(colors[oid % len(colors)], dtype=np.float32)
            frame_np[iy1:iy2, cx1:cx2] = color_np

            # Text label
            score_str = f"{oid}:{score:.2f}" if score is not None else f"{oid}"
            text_x = start_x + padding + box_size + padding
            self._draw_text_np(frame_np, score_str, text_x, item_y, box_size)

        # Write result back into the torch tensor in one shot
        vis_frame.copy_(torch.from_numpy(frame_np))
        return vis_frame

    def _draw_text_np(self, img_np, text, x, y, size):
        """Render bitmap text onto a numpy HWC float32 array — no Python pixel loops."""
        h, w = img_np.shape[:2]
        scale = max(1, size // 6)
        char_width = 4 * scale
        white = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        curr_x = x
        for char in text:
            glyph = self._FONT.get(char)
            if glyph is not None:
                # glyph shape: [5, 3] bool → scale to [5*scale, 3*scale] via repeat
                scaled = np.repeat(np.repeat(glyph, scale, axis=0), scale, axis=1)  # [5s, 3s]
                rows, cols = np.where(scaled)   # pixel coordinates of set bits
                py = y + rows
                px = curr_x + cols
                # Clip to image bounds
                mask = (py >= 0) & (py < h) & (px >= 0) & (px < w)
                img_np[py[mask], px[mask]] = white
                curr_x += char_width
            elif char == ' ':
                curr_x += char_width

    def extract(self, masks, video_state, scores=None, obj_id=-1, plot_all_masks=True):
        """Extract all masks as a batch [N, H, W] using memory-mapped streaming.

        Uses numpy.memmap to write output directly to disk, avoiding OOM for large videos.
        Memory usage is ~100MB regardless of video size (vs 32GB+ for 551 frames at 1080p).
        """
        from PIL import Image

        # Create cache key
        cache_key = (id(masks), video_state.session_uuid, id(scores), obj_id, plot_all_masks)

        # Check if we have cached result
        if cache_key in SAM3VideoOutput._cache:
            _log(f"[SAM3 Video Output] CACHE HIT — returning cached result for session={_hi(video_state.session_uuid[:8])}", _C.GREEN)
            return SAM3VideoOutput._cache[cache_key]

        _log(f"[SAM3 Video Output] CACHE MISS — streaming extraction for session={_hi(video_state.session_uuid[:8])}")
        print_vram("Before extract")
        h, w = video_state.height, video_state.width
        num_frames = video_state.num_frames

        if not masks:
            _log("[SAM3 Video] No masks to extract", _C.YELLOW)
            empty_mask = torch.zeros(num_frames, h, w)
            empty_frames = torch.zeros(num_frames, h, w, 3)
            return (empty_mask, empty_frames, empty_frames)

        # ============================================================
        # STREAMING: Create memory-mapped files on disk
        # Data is written directly to disk, not accumulated in RAM
        # ============================================================
        mmap_dir = os.path.join(video_state.temp_dir, "mmap_output")
        os.makedirs(mmap_dir, exist_ok=True)

        mask_path  = os.path.join(mmap_dir, "masks.mmap")
        frame_path = os.path.join(mmap_dir, "frames.mmap")
        vis_path   = os.path.join(mmap_dir, "vis.mmap")

        # Create memory-mapped arrays (written to disk, not RAM)
        mask_mmap  = np.memmap(mask_path,  dtype='uint8', mode='w+', shape=(num_frames, h, w))
        frame_mmap = np.memmap(frame_path, dtype='uint8', mode='w+', shape=(num_frames, h, w, 3))
        vis_mmap   = np.memmap(vis_path,   dtype='uint8', mode='w+', shape=(num_frames, h, w, 3))

        _log(f"[SAM3 Video] Streaming {_hi(num_frames)} frames "
             f"({_hi(f'{w}x{h}')}) to disk: {_hi(mmap_dir, _C.BLUE)}")

        # Color palette for multiple objects (RGB, 0-1 range)
        colors = [
            [0.0, 0.5, 1.0],   # Blue
            [1.0, 0.3, 0.3],   # Red
            [0.3, 1.0, 0.3],   # Green
            [1.0, 1.0, 0.0],   # Yellow
            [1.0, 0.0, 1.0],   # Magenta
            [0.0, 1.0, 1.0],   # Cyan
            [1.0, 0.5, 0.0],   # Orange
            [0.5, 0.0, 1.0],   # Purple
        ]

        # Pre-scan masks to know num_objects before the parallel loop starts.
        # This avoids needing a lock around shared mutable state inside threads.
        num_objects = 0
        for fm in masks.values():
            if isinstance(fm, np.ndarray):
                fm = torch.from_numpy(fm)
            if fm.dim() == 4:
                fm = fm.squeeze(0)
            if fm.dim() == 3 and fm.shape[0] >= 1:
                num_objects = max(num_objects, fm.shape[0])
            elif fm.numel() > 0:
                num_objects = max(num_objects, 1)

        # ============================================================
        # Process frames in parallel across all available cores.
        # Each thread works on a private copy of its frame data and
        # writes to a unique mmap slice - no shared mutable state.
        # numpy/torch ops release the GIL so threads run truly in
        # parallel on separate cores.
        # ============================================================
        _progress = [0]
        _progress_lock = threading.Lock()
        num_workers = min(os.cpu_count() or 8, num_frames)

        def _process_frame(frame_idx):
            # Load original frame from disk (already stored as JPEG)
            frame_path_jpg = os.path.join(video_state.temp_dir, f"{frame_idx:05d}.jpg")
            if os.path.exists(frame_path_jpg):
                img = Image.open(frame_path_jpg).convert("RGB")
                img_np = np.array(img).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_np)  # [H, W, C]
            else:
                img_np = np.zeros((h, w, 3), dtype=np.float32)
                img_tensor = torch.from_numpy(img_np)

            # Write frame directly to mmap (scaled to uint8, 4x smaller than float32)
            frame_mmap[frame_idx] = (img_np * 255).clip(0, 255).astype('uint8')

            # Get mask for this frame
            if frame_idx in masks:
                frame_mask = masks[frame_idx]

                # Convert numpy to torch if needed
                if isinstance(frame_mask, np.ndarray):
                    frame_mask = torch.from_numpy(frame_mask)

                # Convert mask to ComfyUI format
                if frame_mask.dim() == 4:
                    frame_mask = frame_mask.squeeze(0)  # Remove batch dim

                # Create visualization with colored overlays
                vis_frame = img_tensor.clone()

                # Check for empty mask (no detections)
                if frame_mask.numel() == 0 or (frame_mask.dim() == 3 and frame_mask.shape[0] == 0):
                    # No detections - use empty mask
                    frame_mask = torch.zeros(h, w)
                    # vis_frame stays as original image
                elif frame_mask.dim() == 3 and frame_mask.shape[0] >= 1:
                    combined_mask = torch.zeros(h, w)

                    if plot_all_masks:
                        # Show ALL objects with different colors
                        for oid in range(frame_mask.shape[0]):
                            obj_mask = frame_mask[oid].float()
                            if obj_mask.numel() > 0 and obj_mask.max() > 1.0:
                                obj_mask = obj_mask / 255.0
                            color = torch.tensor(colors[oid % len(colors)])
                            mask_rgb = obj_mask.unsqueeze(-1) * color.view(1, 1, 3)
                            vis_frame = vis_frame * (1 - 0.5 * obj_mask.unsqueeze(-1)) + 0.5 * mask_rgb
                            combined_mask = torch.max(combined_mask, obj_mask)
                    else:
                        # Show only selected obj_id
                        vis_oid = obj_id if obj_id >= 0 and obj_id < frame_mask.shape[0] else 0
                        obj_mask = frame_mask[vis_oid].float()
                        if obj_mask.numel() > 0 and obj_mask.max() > 1.0:
                            obj_mask = obj_mask / 255.0
                        color = torch.tensor(colors[vis_oid % len(colors)])
                        mask_rgb = obj_mask.unsqueeze(-1) * color.view(1, 1, 3)
                        vis_frame = vis_frame * (1 - 0.5 * obj_mask.unsqueeze(-1)) + 0.5 * mask_rgb
                        # Still compute combined for mask output
                        for oid in range(frame_mask.shape[0]):
                            om = frame_mask[oid].float()
                            if om.numel() > 0 and om.max() > 1.0:
                                om = om / 255.0
                            combined_mask = torch.max(combined_mask, om)

                    # For mask output, select based on obj_id
                    if obj_id >= 0 and obj_id < frame_mask.shape[0]:
                        output_mask = frame_mask[obj_id].float()
                        if output_mask.numel() > 0 and output_mask.max() > 1.0:
                            output_mask = output_mask / 255.0
                    else:
                        output_mask = combined_mask
                    frame_mask = output_mask
                else:
                    # Single mask
                    if frame_mask.dim() == 3:
                        frame_mask = frame_mask.squeeze(0)
                    frame_mask = frame_mask.float()
                    if frame_mask.numel() > 0 and frame_mask.max() > 1.0:
                        frame_mask = frame_mask / 255.0
                    color = torch.tensor(colors[0])
                    mask_rgb = frame_mask.unsqueeze(-1) * color.view(1, 1, 3)
                    vis_frame = vis_frame * (1 - 0.5 * frame_mask.unsqueeze(-1)) + 0.5 * mask_rgb

                # Final check for empty masks
                if frame_mask.numel() == 0:
                    frame_mask = torch.zeros(h, w)

                # Draw legend on visualization
                if num_objects > 0:
                    legend_obj_id = -1 if plot_all_masks else obj_id
                    # Get scores for this frame
                    frame_scores = None
                    if scores is not None and frame_idx in scores:
                        frame_scores_tensor = scores[frame_idx]
                        if hasattr(frame_scores_tensor, 'tolist'):
                            frame_scores = frame_scores_tensor.tolist()
                            # Handle nested lists (e.g., [[0.95, 0.87]])
                            if frame_scores and isinstance(frame_scores[0], list):
                                frame_scores = frame_scores[0]
                        elif hasattr(frame_scores_tensor, '__iter__'):
                            frame_scores = list(frame_scores_tensor)
                    vis_frame = self._draw_legend(vis_frame, num_objects, colors, obj_id=legend_obj_id, frame_scores=frame_scores)

                # Write directly to mmap (scaled to uint8)
                vis_mmap[frame_idx]  = (np.clip(vis_frame.numpy(), 0, 1) * 255).astype('uint8')
                mask_mmap[frame_idx] = (frame_mask.cpu().numpy() * 255).clip(0, 255).astype('uint8')
            else:
                # No mask for this frame - use zeros
                mask_mmap[frame_idx] = np.zeros((h, w), dtype='uint8')
                vis_mmap[frame_idx]  = (img_np * 255).clip(0, 255).astype('uint8')

            # Progress reporting - the lock only guards the counter+print,
            # not any of the heavy per-frame work above.
            with _progress_lock:
                _progress[0] += 1
                done = _progress[0]
            if done % 50 == 0 or done == num_frames:
                _log(f"[SAM3 Video] Processed {_hi(f'{done}/{num_frames}')} frames")

        # ============================================================
        # Submit all frames and wait — with per-future diagnostics.
        #
        # WHY NOT `with ThreadPoolExecutor(...) as executor`?
        # The context manager calls shutdown(wait=True) on __exit__,
        # which can hang if any future raised an exception and the
        # executor is still draining its internal work queue.
        # Instead we submit everything, wait with a generous timeout,
        # then report exactly which frames are still pending.
        # ============================================================
        _log(f"[SAM3 Video] Parallel extraction using {_hi(num_workers)} workers")
        executor = ThreadPoolExecutor(max_workers=num_workers)
        futures = {executor.submit(_process_frame, i): i for i in range(num_frames)}

        _log("[SAM3 Video] Waiting for all frame workers to complete...")
        t_wait_start = time.time()
        done_fs, stuck_fs = futures_wait(futures, timeout=300, return_when=ALL_COMPLETED)

        if stuck_fs:
            stuck_ids = sorted(futures[f] for f in stuck_fs)
            _log(f"[SAM3 Video] {_C.BG_RED}WARNING{_C.RESET}{_C.RED} "
                 f"{len(stuck_fs)} frames timed out after 300s: {stuck_ids}{_C.RESET}", _C.RED)
            # Cancel what we can and move on — partial output is better than hanging forever
            for f in stuck_fs:
                f.cancel()
        else:
            elapsed = time.time() - t_wait_start
            _log(f"[SAM3 Video] All {_hi(len(done_fs))} workers finished in {_hi(f'{elapsed:.1f}s', _C.GREEN)}")

        # Re-raise the first exception from any worker (if any)
        exceptions = []
        for f in done_fs:
            exc = f.exception()
            if exc is not None:
                exceptions.append(exc)
        if exceptions:
            _log(f"[SAM3 Video] {len(exceptions)} worker(s) raised exceptions — re-raising first", _C.RED)
            raise exceptions[0]

        # Shut down without blocking (workers are already done)
        executor.shutdown(wait=False)

        # ============================================================
        # Flush — these are fast (just OS dirty-page writeback)
        # ============================================================
        _log("[SAM3 Video] Flushing mmaps to disk...")
        t0 = time.time()
        mask_mmap.flush()
        frame_mmap.flush()
        vis_mmap.flush()
        _log(f"[SAM3 Video] Flush complete in {_hi(f'{time.time()-t0:.1f}s', _C.GREEN)}")

        # ============================================================
        # Convert mmap to torch tensors (backed by disk, minimal RAM!)
        # Scale uint8 [0,255] back to float32 [0,1] as ComfyUI expects.
        # ============================================================
        _log("[SAM3 Video] Converting uint8 mmaps → float32 tensors (disk read + RAM alloc)...")
        t0 = time.time()
        all_masks  = torch.from_numpy(mask_mmap.astype('float32')  / 255.0)
        all_frames = torch.from_numpy(frame_mmap.astype('float32') / 255.0)
        all_vis    = torch.from_numpy(vis_mmap.astype('float32')   / 255.0)
        _log(f"[SAM3 Video] Tensor conversion complete in {_hi(f'{time.time()-t0:.1f}s', _C.GREEN)}")

        _log(f"[SAM3 Video] Output: {_hi(all_masks.shape[0])} masks, "
             f"shape {_hi(list(all_masks.shape), _C.CYAN)}")
        _log(f"[SAM3 Video] Objects tracked: {_hi(num_objects, _C.GREEN)}, "
             f"plot_all_masks: {_hi(plot_all_masks, _C.YELLOW)}")
        print_vram("After extract")

        # Cache the result (tensors backed by mmap files - minimal RAM)
        result = (all_masks, all_frames, all_vis)
        SAM3VideoOutput._cache[cache_key] = result

        return result


# =============================================================================
# Node Mappings
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "SAM3VideoSegmentation": SAM3VideoSegmentation,
    "SAM3Propagate": SAM3Propagate,
    "SAM3VideoOutput": SAM3VideoOutput,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3VideoSegmentation": "SAM3 Video Segmentation",
    "SAM3Propagate": "SAM3 Propagate",
    "SAM3VideoOutput": "SAM3 Video Output",
}
