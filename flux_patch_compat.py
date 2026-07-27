"""Compatibility dispatcher for the Flux2 Fun global FLUX patch.

The original extension replaces ``Flux.forward_orig`` for every FLUX workflow,
even when no Flux2 Fun ControlNet is active. That makes unrelated workflows
inherit a copied implementation that can lag behind ComfyUI's runtime signature.

This dispatcher keeps the native ComfyUI implementation for ordinary workflows
and invokes the specialized Flux2 Fun implementation only when at least one
Flux2 Fun ControlNet is actually present in ``transformer_options``.
"""

from __future__ import annotations

from typing import Any

from . import flux_patch

_DISPATCH_MARKER = "_flux2_fun_compat_dispatch"


def _has_active_flux2_fun_controlnet(transformer_options: dict[str, Any] | None) -> bool:
    """Return whether the current model call contains an active Flux2 Fun ControlNet."""

    options = transformer_options or {}
    controlnets = options.get("flux2_fun_controlnets", ()) or ()
    return any(controlnet is not None for controlnet in controlnets)


def dispatch_forward_orig(
    self,
    img,
    img_ids,
    txt,
    txt_ids,
    timesteps,
    y,
    guidance=None,
    control=None,
    timestep_zero_index=None,
    transformer_options={},
    attn_mask=None,
    **kwargs,
):
    """Dispatch to native ComfyUI or the Flux2 Fun implementation.

    ``timestep_zero_index`` is part of current ComfyUI's FLUX API. The copied
    Flux2 Fun implementation does not yet implement the associated mixed
    modulation semantics. Ordinary workflows therefore use the native method,
    while the unsupported combination of an active Flux2 Fun ControlNet and an
    active timestep-zero reference receives a precise error instead of silently
    generating with incorrect modulation.
    """

    options = transformer_options or {}
    original = flux_patch._original_forward_orig
    if original is None:
        raise RuntimeError("[Flux2 Fun] Native Flux.forward_orig was not captured.")

    if not _has_active_flux2_fun_controlnet(options):
        return original(
            self,
            img,
            img_ids,
            txt,
            txt_ids,
            timesteps,
            y,
            guidance,
            control,
            timestep_zero_index=timestep_zero_index,
            transformer_options=options,
            attn_mask=attn_mask,
            **kwargs,
        )

    if timestep_zero_index is not None:
        raise RuntimeError(
            "[Flux2 Fun] Active Flux2 Fun ControlNet with ref_latents_method="
            "'index_timestep_zero' is not supported yet. Use reference method "
            "'index' or disable Flux2 Fun ControlNet for this generation."
        )

    if kwargs:
        unsupported = ", ".join(sorted(kwargs))
        raise RuntimeError(
            "[Flux2 Fun] The installed ComfyUI passed unsupported FLUX arguments "
            f"while Flux2 Fun ControlNet was active: {unsupported}."
        )

    return flux_patch.patched_forward_orig(
        self,
        img,
        img_ids,
        txt,
        txt_ids,
        timesteps,
        y,
        guidance,
        control,
        transformer_options=options,
        attn_mask=attn_mask,
    )


setattr(dispatch_forward_orig, _DISPATCH_MARKER, True)


def apply_compat_patch(flux_class=None) -> bool:
    """Install the compatibility dispatcher once and preserve removal support."""

    if flux_class is None:
        try:
            from comfy.ldm.flux.model import Flux as flux_class
        except ImportError as exc:
            print(f"[Flux2 Fun] Warning: Could not patch Flux model: {exc}")
            return False

    current = getattr(flux_class, "forward_orig", None)
    if not callable(current):
        print("[Flux2 Fun] Warning: Flux.forward_orig is unavailable")
        return False
    if getattr(current, _DISPATCH_MARKER, False):
        return False

    flux_patch._original_forward_orig = current
    flux_class.forward_orig = dispatch_forward_orig
    flux_patch._patched = True
    print("[Flux2 Fun] Conditional ControlNet compatibility patch applied")
    return True


__all__ = [
    "apply_compat_patch",
    "dispatch_forward_orig",
    "_has_active_flux2_fun_controlnet",
]
