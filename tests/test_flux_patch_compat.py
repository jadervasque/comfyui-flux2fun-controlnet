from __future__ import annotations

import importlib.util
import pathlib
import sys
import types

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = "flux2fun_test_package"


def load_module():
    package = types.ModuleType(PACKAGE)
    package.__path__ = [str(ROOT)]
    sys.modules[PACKAGE] = package

    fake_flux_patch = types.ModuleType(f"{PACKAGE}.flux_patch")
    fake_flux_patch._original_forward_orig = None
    fake_flux_patch._patched = False
    fake_flux_patch.patched_forward_orig = None
    sys.modules[f"{PACKAGE}.flux_patch"] = fake_flux_patch

    spec = importlib.util.spec_from_file_location(
        f"{PACKAGE}.flux_patch_compat", ROOT / "flux_patch_compat.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, fake_flux_patch


def test_unrelated_flux_workflow_uses_native_comfyui_forward():
    module, flux_patch = load_module()
    calls = []

    class FakeFlux:
        def forward_orig(self, *args, **kwargs):
            calls.append((args, kwargs))
            return "native"

    flux_patch.patched_forward_orig = lambda *args, **kwargs: "patched"
    assert module.apply_compat_patch(FakeFlux) is True

    result = FakeFlux().forward_orig(
        "img",
        "img_ids",
        "txt",
        "txt_ids",
        "t",
        "y",
        timestep_zero_index=None,
        transformer_options={},
        attn_mask=None,
    )

    assert result == "native"
    assert len(calls) == 1


def test_active_flux2_fun_controlnet_uses_specialized_forward():
    module, flux_patch = load_module()

    class FakeFlux:
        def forward_orig(self, *args, **kwargs):
            return "native"

    flux_patch.patched_forward_orig = lambda *args, **kwargs: "patched"
    module.apply_compat_patch(FakeFlux)

    result = FakeFlux().forward_orig(
        "img",
        "img_ids",
        "txt",
        "txt_ids",
        "t",
        "y",
        transformer_options={"flux2_fun_controlnets": [object()]},
    )

    assert result == "patched"


def test_active_timestep_zero_reference_is_not_silently_ignored():
    module, flux_patch = load_module()

    class FakeFlux:
        def forward_orig(self, *args, **kwargs):
            return "native"

    flux_patch.patched_forward_orig = lambda *args, **kwargs: "patched"
    module.apply_compat_patch(FakeFlux)

    with pytest.raises(RuntimeError, match="index_timestep_zero"):
        FakeFlux().forward_orig(
            "img",
            "img_ids",
            "txt",
            "txt_ids",
            "t",
            "y",
            timestep_zero_index=[[10, 20]],
            transformer_options={"flux2_fun_controlnets": [object()]},
        )


def test_patch_is_idempotent():
    module, flux_patch = load_module()

    class FakeFlux:
        def forward_orig(self, *args, **kwargs):
            return "native"

    flux_patch.patched_forward_orig = lambda *args, **kwargs: "patched"
    assert module.apply_compat_patch(FakeFlux) is True
    first = FakeFlux.forward_orig
    assert module.apply_compat_patch(FakeFlux) is False
    assert FakeFlux.forward_orig is first
