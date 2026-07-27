"""Modern ComfyUI ControlBase integration for Flux2 Fun wrappers.

ComfyUI's sampler treats every conditioning control object as a ``ControlBase``
instance.  In particular, current releases access ``multigpu_clones`` during
``pre_run_control`` even on single-GPU systems.  The original Flux2 Fun wrapper
predated that contract and implemented only a small duck-typed subset.

This adapter preserves the extension's DIT-hook behaviour while supplying the
complete lifecycle expected by current ComfyUI releases.
"""

from __future__ import annotations

import copy
from typing import Any

_CONTROLBASE_MARKER = "_flux2_fun_controlbase_wrapper"


def build_controlbase_wrapper(base_class):
    """Return ``base_class`` adapted to ComfyUI's current ``ControlBase`` API."""

    from comfy.controlnet import ControlBase

    if getattr(base_class, _CONTROLBASE_MARKER, False):
        return base_class
    if issubclass(base_class, ControlBase):
        setattr(base_class, _CONTROLBASE_MARKER, True)
        return base_class

    class ModernControlNetWrapper(base_class, ControlBase):
        def __init__(
            self,
            controlnet: Any,
            control_context: Any,
            strength: float,
            ctrl_h: int,
            ctrl_w: int,
            low_vram: bool = False,
        ):
            # Initialize the official contract first.  The legacy initializer then
            # fills the Flux2 Fun-specific fields used by the DIT integration.
            ControlBase.__init__(self)
            base_class.__init__(
                self,
                controlnet,
                control_context,
                strength,
                ctrl_h,
                ctrl_w,
                low_vram,
            )

            # The legacy wrapper installed a lightweight placeholder object here.
            # Current ControlBase expects either a real HookGroup or None.
            self.extra_hooks = None

        def pre_run(self, model, percent_to_timestep_function):
            ControlBase.pre_run(self, model, percent_to_timestep_function)

        def cleanup(self):
            ControlBase.cleanup(self)

        def get_models(self):
            # Flux2 Fun moves its plain nn.Module explicitly inside the DIT hook;
            # only chained controls and registered multi-GPU clones are additional
            # ComfyUI-managed models.
            return ControlBase.get_models(self)

        def get_extra_hooks(self):
            return ControlBase.get_extra_hooks(self)

        def copy(self):
            copied = type(self)(
                self.controlnet,
                self.control_context,
                self.strength,
                self.ctrl_h,
                self.ctrl_w,
                self.low_vram,
            )
            ControlBase.copy_to(self, copied)
            copied.previous_controlnet = self.previous_controlnet
            return copied

        def deepclone_multigpu(self, load_device, autoregister: bool = False):
            """Create an independent ControlNet module for another device."""

            cloned_controlnet = copy.deepcopy(self.controlnet)
            cloned_controlnet.to(load_device)

            cloned = type(self)(
                cloned_controlnet,
                self.control_context,
                self.strength,
                self.ctrl_h,
                self.ctrl_w,
                self.low_vram,
            )
            ControlBase.copy_to(self, cloned)

            if self.previous_controlnet is not None:
                cloned.previous_controlnet = self.previous_controlnet.deepclone_multigpu(
                    load_device,
                    autoregister=False,
                )

            if autoregister:
                self.multigpu_clones[load_device] = cloned
            return cloned

        def inference_memory_requirements(self, dtype):
            # Retain the extension's current memory accounting while the lifecycle
            # and clone traversal come from ControlBase.
            own = sum(parameter.numel() for parameter in self.controlnet.parameters()) * 2
            previous = 0
            if self.previous_controlnet is not None:
                previous = self.previous_controlnet.inference_memory_requirements(dtype)
            return own + previous

    ModernControlNetWrapper.__name__ = base_class.__name__
    ModernControlNetWrapper.__qualname__ = base_class.__qualname__
    ModernControlNetWrapper.__module__ = base_class.__module__
    setattr(ModernControlNetWrapper, _CONTROLBASE_MARKER, True)
    return ModernControlNetWrapper


__all__ = ["build_controlbase_wrapper"]
