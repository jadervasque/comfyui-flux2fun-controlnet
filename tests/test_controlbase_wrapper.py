from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).parents[1] / "controlbase_wrapper.py"
spec = importlib.util.spec_from_file_location("flux2_fun_controlbase_wrapper", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


class FakeControlBase:
    def __init__(self):
        self.multigpu_clones = {}
        self.previous_controlnet = None
        self.extra_hooks = None
        self.cond_hint_original = None
        self.strength = 1.0
        self.timestep_percent_range = (0.0, 1.0)
        self.global_average_pooling = False
        self.compression_ratio = 8
        self.upscale_algorithm = "nearest-exact"
        self.latent_format = None
        self.extra_args = {}
        self.vae = None
        self.extra_conds = []
        self.strength_type = object()
        self.concat_mask = False
        self.extra_concat_orig = []
        self.preprocess_image = lambda value: value
        self.pre_run_called = False
        self.cleaned = False

    def pre_run(self, model, percent_to_timestep_function):
        self.pre_run_called = True
        if self.previous_controlnet is not None:
            self.previous_controlnet.pre_run(model, percent_to_timestep_function)

    def cleanup(self):
        self.cleaned = True
        if self.previous_controlnet is not None:
            self.previous_controlnet.cleanup()
        for clone in self.multigpu_clones.values():
            clone.cleanup()

    def get_models(self):
        out = []
        for clone in self.multigpu_clones.values():
            out.extend(clone.get_models())
        if self.previous_controlnet is not None:
            out.extend(self.previous_controlnet.get_models())
        return out

    def get_extra_hooks(self):
        out = []
        if self.extra_hooks is not None:
            out.append(self.extra_hooks)
        if self.previous_controlnet is not None:
            out.extend(self.previous_controlnet.get_extra_hooks())
        return out

    def copy_to(self, copied):
        copied.strength = self.strength
        copied.timestep_percent_range = self.timestep_percent_range
        copied.extra_args = self.extra_args.copy()
        copied.extra_conds = self.extra_conds.copy()
        copied.extra_concat_orig = self.extra_concat_orig.copy()
        copied.extra_hooks = None


class LegacyWrapper:
    def __init__(self, controlnet, control_context, strength, ctrl_h, ctrl_w, low_vram=False):
        self.controlnet = controlnet
        self.control_context = control_context
        self.strength = strength
        self.ctrl_h = ctrl_h
        self.ctrl_w = ctrl_w
        self.low_vram = low_vram
        self.previous_controlnet = None


class FakeControlNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.last_device = None

    def to(self, device):
        self.last_device = str(device)
        return self


def install_fake_comfy(monkeypatch):
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    controlnet = types.ModuleType("comfy.controlnet")
    controlnet.ControlBase = FakeControlBase
    comfy.controlnet = controlnet
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.controlnet", controlnet)


def test_wrapper_initializes_current_controlbase_contract(monkeypatch):
    install_fake_comfy(monkeypatch)
    Wrapper = module.build_controlbase_wrapper(LegacyWrapper)
    wrapper = Wrapper(FakeControlNet(), torch.zeros((1, 4, 3)), 0.75, 2, 2)

    assert isinstance(wrapper, FakeControlBase)
    assert wrapper.multigpu_clones == {}
    assert wrapper.extra_hooks is None
    assert wrapper.strength == 0.75


def test_copy_preserves_flux2_fun_fields_and_control_chain(monkeypatch):
    install_fake_comfy(monkeypatch)
    Wrapper = module.build_controlbase_wrapper(LegacyWrapper)
    wrapper = Wrapper(FakeControlNet(), torch.zeros((1, 4, 3)), 0.5, 2, 2, True)
    previous = Wrapper(FakeControlNet(), torch.zeros((1, 4, 3)), 0.25, 2, 2)
    wrapper.previous_controlnet = previous

    copied = wrapper.copy()

    assert copied is not wrapper
    assert copied.controlnet is wrapper.controlnet
    assert copied.control_context is wrapper.control_context
    assert copied.previous_controlnet is previous
    assert copied.multigpu_clones == {}
    assert copied.low_vram is True


def test_deepclone_multigpu_registers_independent_controlnet(monkeypatch):
    install_fake_comfy(monkeypatch)
    Wrapper = module.build_controlbase_wrapper(LegacyWrapper)
    wrapper = Wrapper(FakeControlNet(), torch.zeros((1, 4, 3)), 1.0, 2, 2)

    cloned = wrapper.deepclone_multigpu(torch.device("cpu"), autoregister=True)

    assert cloned.controlnet is not wrapper.controlnet
    assert cloned.controlnet.last_device == "cpu"
    assert wrapper.multigpu_clones[torch.device("cpu")] is cloned


def test_wrapper_builder_is_idempotent(monkeypatch):
    install_fake_comfy(monkeypatch)
    Wrapper = module.build_controlbase_wrapper(LegacyWrapper)
    assert module.build_controlbase_wrapper(Wrapper) is Wrapper
