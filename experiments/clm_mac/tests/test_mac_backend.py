"""Offline checks with tiny random tensors/models; these do not assess CLM quality."""

import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location("mac_demo", Path(__file__).parents[1] / "demo.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def test_last_token_pooling_handles_padding_and_normalizes():
    hidden = torch.tensor(
        [[[1.0, 0.0], [0.0, 2.0], [9.0, 9.0]], [[9.0, 9.0], [1.0, 0.0], [3.0, 4.0]]]
    )
    mask = torch.tensor([[1, 1, 0], [0, 1, 1]])
    pooled = demo.last_token_embeddings(hidden, mask)
    torch.testing.assert_close(pooled, torch.tensor([[0.0, 1.0], [0.6, 0.8]]))


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        demo.last_token_embeddings(torch.zeros(1, 2, 4), torch.zeros(1, 2, dtype=torch.long))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Needs Apple GPU access")
def test_tiny_qwen3_forward_on_mps():
    from transformers import Qwen3Config, Qwen3Model

    config = Qwen3Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    config._attn_implementation = "sdpa"
    model = Qwen3Model(config).eval().to(device="mps", dtype=torch.bfloat16)
    ids = torch.tensor([[1, 2, 3]], device="mps")
    mask = torch.ones_like(ids)
    with torch.inference_mode():
        output = model(input_ids=ids, attention_mask=mask, use_cache=False)
        pooled = demo.last_token_embeddings(output.last_hidden_state, mask)
    assert pooled.shape == (1, 32)
    assert torch.isfinite(pooled).all()
    torch.testing.assert_close(pooled.norm(dim=-1).cpu(), torch.ones(1), atol=1e-5, rtol=1e-5)
