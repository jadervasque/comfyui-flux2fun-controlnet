"""
ComfyUI Flux2 Fun ControlNet

Implementation of FLUX.2-dev-Fun-Controlnet-Union from Alibaba's VideoX-Fun.
Supports pose, canny, depth, HED, MLSD, and tile control modes.

Model: https://huggingface.co/alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union
Based on: https://github.com/alibaba/VideoX-Fun
"""

from . import nodes as _nodes
from .controlbase_wrapper import build_controlbase_wrapper
from .dit_injection import build_controlnet_wrapper

# Replace only this extension's local wrapper class. The Apply node resolves the
# module global at execution time, so every newly created wrapper uses ComfyUI's
# official per-generation DIT hooks and the current ControlBase lifecycle without
# modifying Flux.forward_orig globally.
_nodes.ControlNetWrapper = build_controlbase_wrapper(
    build_controlnet_wrapper(_nodes.ControlNetWrapper)
)

NODE_CLASS_MAPPINGS = _nodes.NODE_CLASS_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = _nodes.NODE_DISPLAY_NAME_MAPPINGS

__version__ = "1.2.1"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print("[Flux2 Fun] Official composable DIT hook integration enabled")
