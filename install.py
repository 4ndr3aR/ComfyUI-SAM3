#!/usr/bin/env python3
"""
Installation script for ComfyUI-SAM3.
"""

from comfy_env import install
install()

# SHOULD INSTALL:

# https://github.com/PozzettiAndrea/cuda-wheels/releases/download/cc_torch-latest/cc_torch-0.2+cu128torch2.9-cp313-cp313-linux_x86_64.whl

# https://github.com/PozzettiAndrea/cuda-wheels/releases/download/torch_generic_nms-latest/torch_generic_nms-0.1+cu128torch2.10-cp313-cp313-manylinux_2_34_x86_64.manylinux_2_35_x86_64.whl


#import sys
#from pathlib import Path
#
#
#def main():
#    print("\n" + "=" * 60)
#    print("ComfyUI-SAM3 Installation")
#    print("=" * 60)
#
#    from comfy_env import install, IsolatedEnvManager, discover_config
#
#    node_root = Path(__file__).parent.absolute()
#
#    # Run comfy-env install
#    try:
#        #install(config=node_root / "comfy-env.toml", mode="isolated", node_dir=node_root)
#        #install(config=node_root / "comfy-env.toml", node_dir=node_root)
#        install(config=node_root / "comfy-env-root.toml", node_dir=node_root)
#    except Exception as e:
#        print(f"\n[SAM3] Installation FAILED: {e}")
#        print("[SAM3] Report issues at: https://github.com/PozzettiAndrea/ComfyUI-SAM3/issues")
#        return 1
#
#    print("\n" + "=" * 60)
#    print("[SAM3] Installation completed!")
#    print("=" * 60)
#    return 0
#
#
#if __name__ == "__main__":
#    sys.exit(main())
