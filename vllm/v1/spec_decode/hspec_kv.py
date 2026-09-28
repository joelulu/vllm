# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness adapter for borrowing FlashAttention-format target KV pages.

The returned tensors are temporary windows. The target's paged cache is never
written to or registered as a drafter-owned cache. A serving implementation
needs a fused paged attention kernel to avoid this gather and its allocations.
"""

import torch


def borrow_target_kv_window(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    prefix_length: int,
    *,
    head_dim: int,
    window: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read a confirmed prefix window from one request's target KV cache.

    Args:
        kv_cache: FlashAttention cache [pages, page_size, kv_heads, 2 * head_dim].
        block_table: Physical page IDs in logical order for this request.
        prefix_length: Number of confirmed tokens; speculative slots are excluded.
        head_dim: Per-head width of both K and V.
        window: Maximum number of confirmed positions returned.

    Returns:
        K and V, each [1, kv_heads, min(prefix_length, window), head_dim].

    Raises:
        ValueError: On an incompatible cache, invalid block table, or length.

    """
    if kv_cache.ndim != 4 or kv_cache.shape[-1] != 2 * head_dim:
        raise ValueError("expected an unquantized FlashAttention target KV cache")
    if kv_cache.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("quantized target KV requires backend-specific dequantization")
    if block_table.ndim != 1 or block_table.device != kv_cache.device:
        raise ValueError("block table must be one-dimensional and on the cache device")
    if prefix_length < 1 or window < 1 or head_dim < 1:
        raise ValueError("prefix length, window, and head dimension must be positive")
    page_size = kv_cache.shape[1]
    if page_size < 1 or prefix_length > block_table.numel() * page_size:
        raise ValueError("block table does not cover the confirmed prefix")
    start = max(0, prefix_length - window)
    logical_positions = torch.arange(
        start, prefix_length, device=kv_cache.device, dtype=torch.long
    )
    page_ids = block_table[logical_positions // page_size].long()
    if torch.any((page_ids < 0) | (page_ids >= kv_cache.shape[0])).item():
        raise ValueError("confirmed prefix references an invalid physical page")
    packed = kv_cache[page_ids, logical_positions % page_size]
    key, value = packed.split(head_dim, dim=-1)
    return key.transpose(0, 1).unsqueeze(0), value.transpose(0, 1).unsqueeze(0)


def borrow_target_kv_layers(
    caches: dict[str, torch.Tensor],
    block_tables: dict[str, torch.Tensor],
    prefix_length: int,
    *,
    target_prefix: str = "model.layers",
    layer_numbers: tuple[int, ...] = (17, 25, 34),
    head_dim: int,
    window: int = 2048,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Map the paper's one-based Qwen3 layers to vLLM target cache names.

    Args:
        caches: Per-layer target KV tensors, keyed by attention layer name.
        block_tables: Per-layer block tables for the same request.
        prefix_length: Number of confirmed tokens shared with the verifier.
        target_prefix: Target decoder layer name prefix.
        layer_numbers: One-based target layer numbers.
        head_dim: Per-head target KV width.
        window: Maximum borrowed prefix length.

    Returns:
        Per-layer borrowed K/V windows in the given order.

    Raises:
        ValueError: On a duplicate or invalid layer number.
        KeyError: When a requested target layer or block table is missing.

    """
    if (
        not layer_numbers
        or min(layer_numbers) < 1
        or len(set(layer_numbers)) != len(layer_numbers)
    ):
        raise ValueError("target layer numbers must be unique and one-based")
    result = []
    for number in layer_numbers:
        name = f"{target_prefix}.{number - 1}.self_attn.attn"
        result.append(
            borrow_target_kv_window(
                caches[name],
                block_tables[name],
                prefix_length,
                head_dim=head_dim,
                window=window,
            )
        )
    return tuple(result)
