# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for CachedWeightProvider (LFRU expert cache)."""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.expert_weight_provider import (
    CachedWeightProvider,
    ExpertWeightResult,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA required"
)

NUM_EXPERTS = [8, 64]
DTYPES = [torch.bfloat16, torch.float16]
CAPACITIES = [1, 4]
HIDDEN = 16
INTERMEDIATE = 32


def _make_weights(num_experts: int, dtype: torch.dtype):
    w13 = torch.randn(num_experts, 2 * INTERMEDIATE, HIDDEN, dtype=dtype)
    w2 = torch.randn(num_experts, HIDDEN, INTERMEDIATE, dtype=dtype)
    return w13, w2


def _make_scales(num_experts: int):
    w13_s = torch.rand(num_experts, 1, dtype=torch.float32)
    w2_s = torch.rand(num_experts, 1, dtype=torch.float32)
    return w13_s, w2_s


def _make_provider(
    num_experts: int = 8,
    capacity: int = 4,
    dtype: torch.dtype = torch.bfloat16,
    with_scales: bool = False,
    **kwargs,
):
    set_random_seed(42)
    w13, w2 = _make_weights(num_experts, dtype)
    kw: dict = dict(capacity=capacity, w13_weight=w13, w2_weight=w2)
    scales = None
    if with_scales:
        w13_s, w2_s = _make_scales(num_experts)
        kw.update(w13_scale=w13_s, w2_scale=w2_s)
        scales = (w13_s, w2_s)
    kw.update(kwargs)
    return CachedWeightProvider(**kw), w13, w2, scales


def _make_provider_with_global(
    num_experts: int = 8,
    capacity: int = 2,
    global_num_experts: int = 8,
    dtype: torch.dtype = torch.bfloat16,
    **kwargs,
) -> CachedWeightProvider:
    """Create a provider with explicit global_num_experts (e.g. for EP)."""
    set_random_seed(42)
    w13, w2 = _make_weights(num_experts, dtype)
    kw: dict = dict(
        capacity=capacity,
        w13_weight=w13,
        w2_weight=w2,
        global_num_experts=global_num_experts,
    )
    kw.update(kwargs)
    return CachedWeightProvider(**kw)


def _topk(ids: list[int]) -> torch.Tensor:
    return torch.tensor(ids, dtype=torch.int32, device="cuda").unsqueeze(0)


def _prepare(provider: CachedWeightProvider,
             topk_ids: torch.Tensor) -> ExpertWeightResult:
    """Consume the ``prepare`` generator and return the single yielded result.

    Asserts exactly one chunk is yielded (i.e. unique experts <= capacity).
    """
    results = list(provider.prepare(topk_ids))
    assert len(results) == 1, f"Expected 1 chunk, got {len(results)}"
    return results[0]


# -- Core cache behavior --


@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("capacity", CAPACITIES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_cold_miss_and_warm_hit(
    num_experts: int, capacity: int, dtype: torch.dtype
):
    """Cold access misses, repeat access hits. GPU buffer matches source."""
    provider, w13, w2, _ = _make_provider(num_experts, capacity, dtype)
    expert_ids = list(range(min(capacity, num_experts)))

    # Cold miss
    result = _prepare(provider, _topk(expert_ids))
    assert provider.misses == len(expert_ids)
    assert provider.hits == 0
    assert isinstance(result, ExpertWeightResult)
    assert result.w1 is provider.buf_w13
    assert result.w2 is provider.buf_w2
    assert result.topk_ids.shape == (1, len(expert_ids))

    # Verify GPU buffer contents match source weights
    for eid in expert_ids:
        slot = provider._lru[eid][0]
        torch.testing.assert_close(result.w1[slot].cpu(), w13[eid])
        torch.testing.assert_close(result.w2[slot].cpu(), w2[eid])

    # Warm hit
    prev_misses = provider.misses
    _prepare(provider, _topk(expert_ids))
    assert provider.hits == len(expert_ids)
    assert provider.misses == prev_misses


@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_cache_full_equals_num_experts(
    num_experts: int, dtype: torch.dtype
):
    """When capacity == num_experts, all fit with zero evictions."""
    provider, _, _, _ = _make_provider(num_experts, capacity=num_experts,
                                       dtype=dtype)
    all_ids = list(range(num_experts))
    _prepare(provider, _topk(all_ids))
    assert provider.misses == num_experts
    assert len(provider._free_slots) == 0

    _prepare(provider, _topk(all_ids))
    assert provider.hits == num_experts


@pytest.mark.parametrize("capacity", CAPACITIES)
def test_topk_ids_remapping(capacity: int):
    """Remapped topk_ids point to the correct GPU buffer slots."""
    provider, _, _, _ = _make_provider(capacity=capacity)
    ids = list(range(min(capacity, 8)))
    result = _prepare(provider, _topk(ids))

    for eid, slot in zip(
        _topk(ids).squeeze(0).tolist(),
        result.topk_ids.squeeze(0).tolist(),
    ):
        assert provider._lru[eid][0] == slot


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_output_dtype_matches_input(dtype: torch.dtype):
    """Remapped topk_ids preserves input dtype."""
    provider, *_ = _make_provider()
    ids = torch.tensor([[0, 1]], dtype=dtype, device="cuda")
    result = _prepare(provider, ids)
    assert result.topk_ids.dtype == dtype


# -- LFRU eviction semantics --


def test_lfru_prefers_evicting_low_frequency():
    """LFRU evicts the expert with lowest freq/age score, not pure LRU.
    A accessed 5x, B accessed 1x. When C arrives, B is evicted, not A.
    """
    provider, w13, _, _ = _make_provider(capacity=2)
    _prepare(provider, _topk([0, 1]))
    for _ in range(4):
        _prepare(provider, _topk([0]))  # A freq=5
    _prepare(provider, _topk([1]))  # touch B for recency parity

    _prepare(provider, _topk([2]))  # evicts B (lower freq/age score)
    assert 0 in provider._lru, "High-frequency expert A should survive"
    assert 2 in provider._lru, "New expert C should be cached"
    assert 1 not in provider._lru, "Low-frequency expert B should be evicted"
    slot_c = provider._lru[2][0]
    torch.testing.assert_close(provider.buf_w13[slot_c].cpu(), w13[2])


def test_lfru_evicts_stale_high_freq_expert():
    """High historical freq but old last-access loses to recent low-freq.
    Distinguishes LFRU (score=freq/age) from pure frequency-based caching.
    """
    provider, _, _, _ = _make_provider(capacity=2)

    # Expert 0: accessed 11x early, then becomes stale
    _prepare(provider, _topk([0]))
    for _ in range(10):
        _prepare(provider, _topk([0]))
    # Expert 1: loaded later, accessed 51x (0 becomes very stale)
    _prepare(provider, _topk([1]))
    for _ in range(50):
        _prepare(provider, _topk([1]))

    # Expert 0: freq=11, age~62 -> score~0.18. Expert 1: freq=51, age=1 -> 51
    _prepare(provider, _topk([2]))
    assert 1 in provider._lru, "Recent high-freq expert should survive"
    assert 0 not in provider._lru, "Stale expert should be evicted"


def test_capacity_one_always_evicts():
    """With capacity=1, every new expert evicts the previous."""
    provider, *_ = _make_provider(capacity=1)
    for eid in range(5):
        _prepare(provider, _topk([eid]))
    assert provider.misses == 5
    assert provider.hits == 0
    assert len(provider._lru) == 1
    assert 4 in provider._lru


# -- GPU buffer correctness under eviction --


def test_gpu_buffer_correct_after_eviction():
    """After eviction, the reused slot contains the new expert's weights."""
    provider, w13, w2, _ = _make_provider(capacity=4)
    _prepare(provider, _topk([0, 1, 2, 3]))

    # Make 0 the eviction candidate (least recently used, lowest freq)
    _prepare(provider, _topk([1, 2, 3]))
    slot_for_0 = provider._lru[0][0]

    _prepare(provider, _topk([7]))
    assert provider._lru[7][0] == slot_for_0
    torch.testing.assert_close(provider.buf_w13[slot_for_0].cpu(), w13[7])
    torch.testing.assert_close(provider.buf_w2[slot_for_0].cpu(), w2[7])


# -- Scale buffer handling --


def test_scale_lifecycle():
    """Scales are allocated, copied on load, and updated on eviction."""
    if not current_platform.has_device_capability(89):
        pytest.skip("FP8 requires CUDA capability >= 89")

    provider, _, _, scales = _make_provider(
        capacity=4, dtype=torch.float8_e4m3fn, with_scales=True
    )
    w13_s, w2_s = scales

    # Buffers allocated on GPU
    assert provider.buf_w13_scale is not None
    assert provider.buf_w2_scale is not None
    assert provider.buf_w13_scale.device.type == "cuda"

    # Scales copied correctly on load
    result = _prepare(provider, _topk([3, 6]))
    for eid in [3, 6]:
        slot = provider._lru[eid][0]
        torch.testing.assert_close(result.w1_scale[slot].cpu(), w13_s[eid])
        torch.testing.assert_close(result.w2_scale[slot].cpu(), w2_s[eid])

    # Fill cache and evict: scales must be updated in evicted slot
    _prepare(provider, _topk([0, 1]))  # cache now full: [3, 6, 0, 1]
    _prepare(provider, _topk([3, 6, 0]))  # boost freq on 3,6,0; expert 1 stale

    result = _prepare(provider, _topk([7]))  # evicts 1
    assert 1 not in provider._lru
    slot_7 = provider._lru[7][0]
    torch.testing.assert_close(
        provider.buf_w13_scale[slot_7].cpu(), w13_s[7]
    )
    torch.testing.assert_close(
        provider.buf_w2_scale[slot_7].cpu(), w2_s[7]
    )


def test_no_scales_when_not_provided():
    """Without scale inputs, scale buffers remain None."""
    provider, *_ = _make_provider()
    assert provider.buf_w13_scale is None
    assert provider.buf_w2_scale is None
    result = _prepare(provider, _topk([0]))
    assert result.w1_scale is None
    assert result.w2_scale is None


# -- Invalidation --


def test_invalidate_frees_slot():
    """invalidate() removes an expert and returns its slot to the free list."""
    provider, *_ = _make_provider()
    _prepare(provider, _topk([0, 1, 2, 3]))
    old_slot = provider._lru[2][0]
    provider.invalidate(2)
    assert 2 not in provider._lru
    assert old_slot in provider._free_slots


def test_invalidate_noop_when_absent():
    """invalidate() on an uncached expert is a no-op."""
    provider, *_ = _make_provider()
    provider.invalidate(99)  # must not raise


# -- Overflow (unique experts > capacity) --


def test_overflow_yields_chunks():
    """When unique experts exceed capacity, generator yields multiple chunks.

    Each chunk carries ``token_indices`` and ``expert_map``.  Disjoint
    expert sets produce disjoint token sets (each token in exactly one
    chunk).
    """
    provider, *_ = _make_provider(capacity=2)
    # Two tokens whose expert sets are disjoint and each fits in capacity.
    topk_ids = torch.tensor(
        [[0, 1], [2, 3]], dtype=torch.int32, device="cuda"
    )

    results = list(provider.prepare(topk_ids))
    assert len(results) == 2, f"Expected 2 chunks, got {len(results)}"

    # Every result must carry an expert_map and reference live GPU buffers.
    for r in results:
        assert r.token_indices is not None
        assert r.expert_map is not None
        assert r.w1 is provider.buf_w13
        assert r.w2 is provider.buf_w2

    # Combined token coverage — with EP-style, disjoint sets should
    # give non-overlapping token_indices.
    all_indices = torch.cat([r.token_indices for r in results])
    assert set(all_indices.tolist()) == {0, 1}

    # Each chunk produces correctly shaped remapped ids.
    assert results[0].topk_ids.shape == (1, 2)
    assert results[1].topk_ids.shape == (1, 2)


def test_overflow_cross_chunk_token_ep_style():
    """Token whose experts span chunks appears in EVERY relevant chunk.

    EP-style: token [0, 3] needs experts 0 (chunk1: {0,1}) and 3
    (chunk2: {2,3}).  Both chunks process it.  Out-of-chunk experts
    are remapped to -1 in topk_ids so the caller can zero the weight.
    """
    provider, *_ = _make_provider(capacity=2)
    topk_ids = torch.tensor([[0, 3]], dtype=torch.int32, device="cuda")

    results = list(provider.prepare(topk_ids))
    # Token appears in both chunks.
    assert len(results) == 2, f"Expected 2 chunks for cross-chunk token"

    # Verify both chunks have the token.
    seen = set()
    for r in results:
        assert r.token_indices is not None
        assert r.expert_map is not None
        tok_list = r.token_indices.tolist()
        assert tok_list == [0]
        seen.update(tok_list)
    assert seen == {0}

    # Chunk 1 (experts {0,1}): expert 0 in-chunk, expert 3 out-of-chunk (-1).
    r0 = results[0]
    assert r0.topk_ids[0, 0].item() >= 0  # expert 0 → valid slot
    assert r0.topk_ids[0, 1].item() == -1  # expert 3 → masked

    # Chunk 2 (experts {2,3}): expert 0 out-of-chunk, expert 3 in-chunk.
    r1 = results[1]
    assert r1.topk_ids[0, 0].item() == -1  # expert 0 → masked
    assert r1.topk_ids[0, 1].item() >= 0  # expert 3 → valid slot

    # expert_map masks out-of-chunk experts.
    for expert_id in (0, 1):
        assert r0.expert_map[expert_id].item() >= 0
    for expert_id in (2, 3):
        assert r0.expert_map[expert_id].item() == -1


def test_overflow_all_tokens_covered():
    """Every token must appear in at least one chunk (EP-style)."""
    provider, *_ = _make_provider(capacity=2)
    num_tokens = 8
    # Each token uses a different pair of experts, forcing chunking.
    topk_ids = torch.tensor(
        [[i, i + 4] for i in range(num_tokens)],
        dtype=torch.int32,
        device="cuda",
    )

    results = list(provider.prepare(topk_ids))
    covered = torch.zeros(num_tokens, dtype=torch.bool, device="cuda")
    for r in results:
        if r.token_indices is not None:
            covered[r.token_indices] = True

    assert covered.all().item(), "All tokens must be covered"


def test_overflow_no_crash_single_token_many_experts():
    """Single token with more experts than capacity should not crash."""
    provider, *_ = _make_provider(capacity=2)
    result = list(provider.prepare(_topk([0, 1, 2, 3])))
    # EP-style: token spans 2 chunks ({0,1} and {2,3}), appears in both.
    assert len(result) >= 1
    for r in result:
        if r.token_indices is not None:
            assert r.topk_ids.shape == (1, 4)
            assert r.expert_map is not None


def test_overflow_chunk_truncation_legacy():
    """chunk_on_overflow=False uses old truncation behaviour."""
    provider, *_ = _make_provider(capacity=2, chunk_on_overflow=False)
    result = _prepare(provider, _topk([0, 1, 2, 3]))
    # Old behaviour: only last `capacity` experts (2,3) loaded.
    assert result.topk_ids.shape == (1, 4)
    assert result.token_indices is None
    assert result.expert_map is None
    assert len(provider._lru) <= 2


def test_overflow_expert_map_covers_global_experts():
    """expert_map has shape (global_num_experts,) with in-chunk slots."""
    global_exp = 8
    provider = _make_provider_with_global(global_num_experts=global_exp)
    topk_ids = torch.tensor(
        [[0, 1, 6, 7]], dtype=torch.int32, device="cuda"
    )
    results = list(provider.prepare(topk_ids))
    for r in results:
        assert r.expert_map is not None
        assert r.expert_map.shape == (global_exp,)
        assert r.expert_map.dtype == torch.int32
        assert r.expert_map.device.type == "cuda"
        # In-chunk experts have valid slots, out-of-chunk are -1.
        in_chunk = r.expert_map >= 0
        assert in_chunk.sum().item() <= provider.capacity


def test_overflow_no_expert_map_when_fits():
    """expert_map is None when unique <= capacity (single chunk)."""
    provider, *_ = _make_provider(capacity=4)
    result = _prepare(provider, _topk([0, 1, 2]))
    assert result.expert_map is None
    assert result.token_indices is None


# -- CPU pinned memory --


def test_cpu_backing_is_pinned():
    """CPU weight tensors must be pinned for async H2D copies."""
    provider, *_ = _make_provider()
    assert provider._cpu_w13.is_pinned()
    assert provider._cpu_w2.is_pinned()
