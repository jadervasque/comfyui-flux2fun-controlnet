"""Official ComfyUI DIT-hook integration for Flux2 Fun ControlNet.

The extension used to replace ``Flux.forward_orig`` globally. That copied ComfyUI's
entire FLUX implementation and broke whenever the internal call signature evolved.
This module instead installs composable ``patches_replace['dit']`` callbacks in the
per-generation ``transformer_options`` dictionary supplied by ComfyUI's control path.

ComfyUI remains responsible for reference latents, ``timestep_zero_index`` and mixed
modulation. Flux2 Fun only generates hints from the main-image modulation branch and
adds them after the configured FLUX double-stream blocks.
"""

from __future__ import annotations

import math
from typing import Any

import torch

_CONTROLS_KEY = "_flux2_fun_controls"
_RUNTIME_STATE_KEY = "_flux2_fun_runtime_state"
_PATCH_MARKER = "_flux2_fun_dit_patch_layer"
_WRAPPER_MARKER = "_flux2_fun_official_control_wrapper"


def convert_pe_to_diffusers(pe):
    """Convert ComfyUI positional embeddings to the ``(cos, sin)`` format."""

    if pe is None:
        return None
    if isinstance(pe, tuple) and len(pe) == 2:
        return pe
    if pe.dim() == 6:
        pe = pe.squeeze(0).squeeze(0)
    if pe.dim() == 4:
        seq_len = pe.shape[0]
        cos = pe[:, :, 0, :].reshape(seq_len, -1)
        sin = pe[:, :, 1, :].reshape(seq_len, -1)
        return cos, sin
    return None


def _select_modulation_branch(tensor: torch.Tensor, branch: int = 0) -> torch.Tensor:
    """Select one mixed-timestep modulation branch while retaining broadcast shape."""

    if tensor.ndim >= 3 and tensor.shape[1] > 1:
        return tensor[:, branch : branch + 1]
    return tensor


def _modulation_out_to_tuple(value: Any, branch: int = 0):
    if hasattr(value, "shift"):
        return (
            _select_modulation_branch(value.shift, branch),
            _select_modulation_branch(value.scale, branch),
            _select_modulation_branch(value.gate, branch),
        )
    if isinstance(value, tuple) and len(value) == 3:
        return tuple(_select_modulation_branch(item, branch) for item in value)
    raise TypeError(f"Unsupported modulation value: {type(value)!r}")


def extract_main_modulation(vec: Any):
    """Return image/text modulation for the main-image timestep branch.

    Current FLUX.2 models use global modulation. When ``index_timestep_zero`` is
    active, ComfyUI passes two image modulation branches: index 0 for normal/main
    tokens and index 1 for zero-timestep reference tokens. Flux2 Fun processes only
    main image tokens, so it must use branch 0 while leaving ComfyUI's base model to
    apply the complete mixed modulation to all tokens.
    """

    if not (isinstance(vec, tuple) and len(vec) == 2):
        raise RuntimeError(
            "[Flux2 Fun] The loaded model does not expose FLUX.2 global modulation "
            "through the DIT patch API. Flux2 Fun supports FLUX.2 global-modulation models."
        )

    img_modulation, txt_modulation = vec
    if not (isinstance(img_modulation, tuple) and len(img_modulation) == 2):
        raise RuntimeError("[Flux2 Fun] Unexpected image modulation structure from ComfyUI.")
    if not (isinstance(txt_modulation, tuple) and len(txt_modulation) == 2):
        raise RuntimeError("[Flux2 Fun] Unexpected text modulation structure from ComfyUI.")

    image = tuple(_modulation_out_to_tuple(value, branch=0) for value in img_modulation)
    text = tuple(_modulation_out_to_tuple(value, branch=0) for value in txt_modulation)
    return image, text


def _repeat_to_batch(tensor: torch.Tensor, target_batch: int, name: str) -> torch.Tensor:
    if tensor.shape[0] == target_batch:
        return tensor
    if tensor.shape[0] <= 0 or target_batch % tensor.shape[0] != 0:
        raise RuntimeError(
            f"[Flux2 Fun] Cannot expand {name} batch {tensor.shape[0]} to model batch {target_batch}."
        )
    repeats = [target_batch // tensor.shape[0]] + [1] * (tensor.ndim - 1)
    return tensor.repeat(*repeats)


def _resize_hint_tokens(hint: torch.Tensor, target_tokens: int) -> torch.Tensor:
    if hint.shape[1] == target_tokens:
        return hint

    def find_hw(seq_len: int) -> tuple[int, int]:
        for height in range(int(math.sqrt(seq_len)), 0, -1):
            if seq_len % height == 0:
                return height, seq_len // height
        return 1, seq_len

    hint_h, hint_w = find_hw(hint.shape[1])
    target_h, target_w = find_hw(target_tokens)
    hint_2d = hint.permute(0, 2, 1).reshape(
        hint.shape[0], hint.shape[2], hint_h, hint_w
    )
    resized = torch.nn.functional.interpolate(
        hint_2d,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    )
    return resized.reshape(hint.shape[0], hint.shape[2], -1).permute(0, 2, 1)


def _prepare_hints(args: dict[str, Any], transformer_options: dict[str, Any]) -> None:
    controls = list((transformer_options.get(_CONTROLS_KEY) or {}).values())
    if not controls:
        transformer_options[_RUNTIME_STATE_KEY] = {"hints": {}, "last_layer": 0}
        return

    img = args["img"]
    txt = args["txt"]
    image_modulation, text_modulation = extract_main_modulation(args["vec"])
    image_rotary_emb = convert_pe_to_diffusers(args.get("pe"))
    hints_by_layer: dict[int, list[tuple[torch.Tensor, float, int]]] = {}

    for record in controls:
        controlnet = record["controlnet"]
        context = record["control_context"]
        strength = float(record["strength"])
        ctrl_h, ctrl_w = record["ctrl_dims"]
        low_vram = bool(record["low_vram"])
        main_tokens = int(ctrl_h) * int(ctrl_w)

        if main_tokens <= 0 or main_tokens > img.shape[1]:
            raise RuntimeError(
                "[Flux2 Fun] Control token dimensions do not match the main FLUX image tokens: "
                f"requested={main_tokens}, available={img.shape[1]}."
            )

        debug = not bool(getattr(controlnet, "_flux2_fun_logged", False))
        if low_vram:
            controlnet.to(img.device)

        try:
            context = context.to(device=img.device, dtype=img.dtype)
            context = _repeat_to_batch(context, img.shape[0], "control context")
            img_for_control = img[:, :main_tokens].clone()

            if debug and low_vram:
                print("[Flux2 Fun] Low VRAM mode enabled - using CPU offloading")
            if debug and image_rotary_emb is not None:
                cos, sin = image_rotary_emb
                print(f"[Flux2 Fun] RoPE: cos={cos.shape}, sin={sin.shape}")
                print(
                    f"[Flux2 Fun] img tokens: {img.shape[1]}, main tokens: {main_tokens}"
                )
                if img.shape[1] > main_tokens:
                    print(
                        "[Flux2 Fun] Reference latent tokens detected: "
                        f"{img.shape[1] - main_tokens}"
                    )

            control_hints = controlnet.forward_control(
                x=img_for_control,
                control_context=context,
                encoder_hidden_states=txt.clone(),
                temb_mod_params_img=image_modulation,
                temb_mod_params_txt=text_modulation,
                image_rotary_emb=image_rotary_emb,
                ctrl_h=ctrl_h,
                ctrl_w=ctrl_w,
                txt_seq_len=txt.shape[1],
                debug=debug,
            )
            del img_for_control

            for layer_index, hint_index in controlnet.control_layers_mapping.items():
                if hint_index >= len(control_hints):
                    continue
                hints_by_layer.setdefault(int(layer_index), []).append(
                    (control_hints[hint_index], strength, main_tokens)
                )

            if debug:
                print(
                    f"[Flux2 Fun] ControlNet generated {len(control_hints)} hints, "
                    f"strength={strength}"
                )
                setattr(controlnet, "_flux2_fun_logged", True)
        except Exception as exc:
            raise RuntimeError(f"[Flux2 Fun] Failed to generate ControlNet hints: {exc}") from exc
        finally:
            if low_vram:
                controlnet.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    transformer_options[_RUNTIME_STATE_KEY] = {
        "hints": hints_by_layer,
        "last_layer": max(hints_by_layer, default=0),
    }


def _call_composed_patch(existing, args, extra):
    if existing is None:
        return extra["original_block"](args)
    return existing(args, extra)


def _make_double_block_patch(layer_index: int, existing=None):
    def flux2_fun_double_block(args, extra):
        options = args.get("transformer_options")
        if options is None:
            options = {}

        if layer_index == 0:
            _prepare_hints(args, options)

        out = _call_composed_patch(existing, args, extra)
        state = options.get(_RUNTIME_STATE_KEY) or {}
        layer_hints = (state.get("hints") or {}).get(layer_index, ())

        for hint, strength, main_tokens in layer_hints:
            hint = hint.to(device=out["img"].device, dtype=out["img"].dtype)
            hint = _resize_hint_tokens(hint, main_tokens)
            hint = _repeat_to_batch(hint, out["img"].shape[0], "ControlNet hint")
            out["img"][:, :main_tokens] = (
                out["img"][:, :main_tokens] + hint * strength
            )

        if state and layer_index >= int(state.get("last_layer", 0)):
            options.pop(_RUNTIME_STATE_KEY, None)
        return out

    setattr(flux2_fun_double_block, _PATCH_MARKER, layer_index)
    return flux2_fun_double_block


def register_control(
    transformer_options: dict[str, Any],
    *,
    owner: Any,
    controlnet: Any,
    control_context: torch.Tensor,
    strength: float,
    ctrl_dims: tuple[int, int],
    low_vram: bool,
) -> None:
    """Register one ControlNet and composable DIT replacements for this model call."""

    controls = transformer_options.setdefault(_CONTROLS_KEY, {})
    controls[id(owner)] = {
        "controlnet": controlnet,
        "control_context": control_context,
        "strength": strength,
        "ctrl_dims": ctrl_dims,
        "low_vram": low_vram,
    }

    target_layers = {0}
    for record in controls.values():
        target_layers.update(int(index) for index in record["controlnet"].control_layers_mapping)

    patches_replace = transformer_options.setdefault("patches_replace", {})
    dit_patches = patches_replace.setdefault("dit", {})
    for layer_index in sorted(target_layers):
        key = ("double_block", layer_index)
        current = dit_patches.get(key)
        if getattr(current, _PATCH_MARKER, None) == layer_index:
            continue
        dit_patches[key] = _make_double_block_patch(layer_index, current)


def build_controlnet_wrapper(base_class):
    """Return a ControlNet wrapper using official per-call DIT patch registration."""

    if getattr(base_class, _WRAPPER_MARKER, False):
        return base_class

    class OfficialDITControlNetWrapper(base_class):
        def get_control(
            self,
            x_noisy,
            t,
            cond,
            batched_number,
            transformer_options=None,
        ):
            control_prev = None
            if self.previous_controlnet:
                control_prev = self.previous_controlnet.get_control(
                    x_noisy,
                    t,
                    cond,
                    batched_number,
                    transformer_options,
                )

            if transformer_options is not None:
                register_control(
                    transformer_options,
                    owner=self,
                    controlnet=self.controlnet,
                    control_context=self.control_context,
                    strength=self.strength,
                    ctrl_dims=(self.ctrl_h, self.ctrl_w),
                    low_vram=self.low_vram,
                )

            output = {"input": [], "output": []}
            if control_prev:
                output["input"] = control_prev.get("input", [])
                output["output"] = control_prev.get("output", [])
            return output

        def copy(self):
            copied = type(self)(
                self.controlnet,
                self.control_context,
                self.strength,
                self.ctrl_h,
                self.ctrl_w,
                self.low_vram,
            )
            copied.previous_controlnet = self.previous_controlnet
            return copied

    OfficialDITControlNetWrapper.__name__ = base_class.__name__
    OfficialDITControlNetWrapper.__qualname__ = base_class.__qualname__
    OfficialDITControlNetWrapper.__module__ = base_class.__module__
    setattr(OfficialDITControlNetWrapper, _WRAPPER_MARKER, True)
    return OfficialDITControlNetWrapper


__all__ = [
    "build_controlnet_wrapper",
    "convert_pe_to_diffusers",
    "extract_main_modulation",
    "register_control",
]
