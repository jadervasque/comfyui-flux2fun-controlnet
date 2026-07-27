"""Legacy compatibility shim for Flux2 Fun ControlNet.

Older releases replaced ``comfy.ldm.flux.model.Flux.forward_orig`` globally. The
current integration uses ComfyUI's official per-generation DIT replacement hooks in
``dit_injection.py`` and therefore must not modify the FLUX class at import time.
"""

from __future__ import annotations

_patched = False
_original_forward_orig = None
_notice_printed = False


def apply_patch() -> bool:
    """Preserve the old import call without applying a global ComfyUI monkey patch."""

    global _notice_printed
    if not _notice_printed:
        print("[Flux2 Fun] Global Flux.forward_orig patch disabled; using official DIT hooks")
        _notice_printed = True
    return False


def remove_patch() -> bool:
    """Remain compatible with callers from releases that exposed patch removal."""

    return False


__all__ = ["apply_patch", "remove_patch"]
