from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "flux_patch.py"
spec = importlib.util.spec_from_file_location("flux2_fun_legacy_patch", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_legacy_apply_patch_is_a_noop():
    assert module.apply_patch() is False
    assert module._patched is False
    assert module._original_forward_orig is None


def test_legacy_remove_patch_is_a_noop():
    assert module.remove_patch() is False
