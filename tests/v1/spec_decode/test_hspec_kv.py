"""Target KV page borrowing must preserve request and position boundaries."""

import importlib.util
from pathlib import Path

import pytest
import torch

# Load the standalone adapter without importing CUDA-dependent vLLM modules.
_path = Path(__file__).resolve().parents[3] / "vllm/v1/spec_decode/hspec_kv.py"
_spec = importlib.util.spec_from_file_location("hspec_kv", _path)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
borrow_target_kv_window = _module.borrow_target_kv_window
borrow_target_kv_layers = _module.borrow_target_kv_layers


def test_borrow_window_uses_only_confirmed_positions_of_request():
    cache = torch.arange(6 * 2 * 1 * 4, dtype=torch.float32).reshape(6, 2, 1, 4)
    before = cache.clone()
    left = torch.tensor([4, 1, 5])
    right = torch.tensor([0, 3, 2])
    key, value = borrow_target_kv_window(cache, left, 5, head_dim=2, window=3)
    expected = torch.stack((cache[1, 0, 0], cache[1, 1, 0], cache[5, 0, 0]))
    torch.testing.assert_close(key[0, 0], expected[:, :2])
    torch.testing.assert_close(value[0, 0], expected[:, 2:])
    other_key, _ = borrow_target_kv_window(cache, right, 5, head_dim=2, window=3)
    assert not torch.equal(other_key, key)
    torch.testing.assert_close(cache, before)


def test_missing_page_and_unconfirmed_slot_fail_closed():
    cache = torch.zeros(2, 2, 1, 4)
    with pytest.raises(ValueError, match="does not cover"):
        borrow_target_kv_window(cache, torch.tensor([0]), 3, head_dim=2)
    with pytest.raises(ValueError, match="invalid physical page"):
        borrow_target_kv_window(cache, torch.tensor([0, -1]), 3, head_dim=2)
    with pytest.raises(ValueError, match="unquantized"):
        borrow_target_kv_window(cache[..., :2], torch.tensor([0]), 1, head_dim=2)


def test_one_based_layer_mapping_reads_verifier_pages():
    cache = torch.arange(8, dtype=torch.float32).reshape(1, 2, 1, 4)
    names = [f"model.layers.{i}.self_attn.attn" for i in (16, 24, 33)]
    pairs = borrow_target_kv_layers(
        dict.fromkeys(names, cache),
        {name: torch.tensor([0]) for name in names},
        2,
        head_dim=2,
    )
    assert len(pairs) == 3
    for key, value in pairs:
        torch.testing.assert_close(key[0, 0], cache[:, :, 0, :2].flatten(0, 1))
        torch.testing.assert_close(value[0, 0], cache[:, :, 0, 2:].flatten(0, 1))
