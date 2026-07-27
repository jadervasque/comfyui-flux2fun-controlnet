from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

MODULE_PATH = Path(__file__).parents[1] / "dit_injection.py"
spec = importlib.util.spec_from_file_location("flux2_fun_dit_injection", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def modulation(batch=1, branches=1, hidden=4, value=0.0):
    tensor = torch.full((batch, branches, hidden), value)
    return SimpleNamespace(shift=tensor, scale=tensor + 1, gate=tensor + 2)


def global_vec(batch=1, image_branches=1, hidden=4):
    return (
        (modulation(batch, image_branches, hidden, 10), modulation(batch, image_branches, hidden, 20)),
        (modulation(batch, 1, hidden, 30), modulation(batch, 1, hidden, 40)),
    )


class FakeControlNet:
    control_layers_mapping = {0: 0, 2: 1}

    def __init__(self):
        self.received_image_modulation = None
        self.moves = []

    def to(self, device):
        self.moves.append(str(device))
        return self

    def forward_control(self, **kwargs):
        self.received_image_modulation = kwargs["temb_mod_params_img"]
        x = kwargs["x"]
        return [torch.ones_like(x), torch.full_like(x, 2.0)]


def original_block(args):
    return {"img": args["img"] + 1, "txt": args["txt"]}


def test_extracts_main_timestep_branch_for_index_timestep_zero():
    image, text = module.extract_main_modulation(global_vec(image_branches=2))

    assert image[0][0].shape == (1, 1, 4)
    assert torch.all(image[0][0] == 10)
    assert torch.all(image[1][0] == 20)
    assert text[0][0].shape == (1, 1, 4)


def test_dit_patches_generate_once_and_inject_after_target_blocks():
    options = {}
    controlnet = FakeControlNet()
    owner = object()
    context = torch.zeros((1, 4, 3))

    module.register_control(
        options,
        owner=owner,
        controlnet=controlnet,
        control_context=context,
        strength=0.5,
        ctrl_dims=(2, 2),
        low_vram=False,
    )

    patches = options["patches_replace"]["dit"]
    args = {
        "img": torch.zeros((1, 6, 4)),
        "txt": torch.zeros((1, 2, 4)),
        "vec": global_vec(image_branches=2),
        "pe": None,
        "transformer_options": options,
    }

    out0 = patches[("double_block", 0)](args, {"original_block": original_block})
    assert torch.all(out0["img"][:, :4] == 1.5)
    assert torch.all(out0["img"][:, 4:] == 1.0)
    assert torch.all(controlnet.received_image_modulation[0][0] == 10)

    args2 = dict(args, img=out0["img"])
    out2 = patches[("double_block", 2)](args2, {"original_block": original_block})
    assert torch.all(out2["img"][:, :4] == 3.5)
    assert "_flux2_fun_runtime_state" not in options


def test_composes_an_existing_dit_replacement():
    def existing(args, extra):
        out = extra["original_block"](args)
        out["img"] = out["img"] * 3
        return out

    options = {"patches_replace": {"dit": {("double_block", 0): existing}}}
    module.register_control(
        options,
        owner=object(),
        controlnet=FakeControlNet(),
        control_context=torch.zeros((1, 4, 3)),
        strength=1.0,
        ctrl_dims=(2, 2),
        low_vram=False,
    )

    args = {
        "img": torch.zeros((1, 4, 4)),
        "txt": torch.zeros((1, 2, 4)),
        "vec": global_vec(),
        "pe": None,
        "transformer_options": options,
    }
    out = options["patches_replace"]["dit"][("double_block", 0)](
        args, {"original_block": original_block}
    )
    assert torch.all(out["img"] == 4.0)


def test_wrapper_registers_without_modifying_comfyui_flux_class():
    class BaseWrapper:
        def __init__(self, controlnet, context, strength, h, w, low_vram=False):
            self.controlnet = controlnet
            self.control_context = context
            self.strength = strength
            self.ctrl_h = h
            self.ctrl_w = w
            self.low_vram = low_vram
            self.previous_controlnet = None

    Wrapper = module.build_controlnet_wrapper(BaseWrapper)
    WrapperAgain = module.build_controlnet_wrapper(Wrapper)
    assert WrapperAgain is Wrapper

    options = {}
    wrapper = Wrapper(FakeControlNet(), torch.zeros((1, 4, 3)), 0.75, 2, 2)
    assert wrapper.get_control(None, None, None, 1, options) == {"input": [], "output": []}
    assert ("double_block", 0) in options["patches_replace"]["dit"]
    assert wrapper.copy() is not wrapper
