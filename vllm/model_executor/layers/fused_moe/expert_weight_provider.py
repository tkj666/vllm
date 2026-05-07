# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Generator
from dataclasses import dataclass

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class ExpertWeightResult:
    """GPU-resident expert weights ready for kernel consumption.

    When ``token_indices`` is not None, the caller must slice hidden states
    and topk_weights by these indices before invoking the kernel, then
    **accumulate** (``+=``) the kernel output back into the full result
    tensor at the same indices.  A token may appear in multiple chunks when
    its top-k experts span chunk boundaries (EP-like multi-pass).

    When ``expert_map`` is not None, the caller must pass it to the kernel
    in place of ``layer.expert_map``.  Experts not loaded in the current
    chunk are mapped to -1 so the kernel skips them.  Out-of-chunk entries
    in ``topk_ids`` are also set to -1; the caller must zero the
    corresponding ``topk_weights`` and clamp the sentinel to 0 before
    invoking the kernel.
    """

    w1: torch.Tensor
    w2: torch.Tensor
    topk_ids: torch.Tensor
    w1_scale: torch.Tensor | None = None
    w2_scale: torch.Tensor | None = None
    token_indices: torch.Tensor | None = None
    expert_map: torch.Tensor | None = None


class CachedWeightProvider:
    """GPU LRU cache backed by CPU pinned memory.

    Keeps capacity expert weight tensors in a fixed-size GPU scratch
    buffer. All expert weights reside in CPU pinned memory; only the N
    hottest experts are mirrored into the GPU buffer.

    Uses LFRU (frequency-weighted LRU) eviction: score = freq / age.
    This prevents early layers from monopolizing the cache — a known
    problem with pure LRU in sequential MoE execution where early
    layers always appear "recently used."

    On each forward pass, prepare() identifies which experts are needed,
    copies any misses from CPU to GPU (evicting the lowest-scored entry
    when the buffer is full), and yields ExpertWeightResult(s) with
    remapped topk_ids whose values are GPU-buffer slot indices.

    When unique experts exceed capacity and *chunk_on_overflow* is True,
    experts are split into capacity-sized chunks and yielded EP-style
    (one result per chunk).  Callers must accumulate partial results for
    tokens that span multiple chunks.
    """

    def __init__(
        self,
        capacity: int,
        w13_weight: torch.Tensor,
        w2_weight: torch.Tensor,
        w13_scale: torch.Tensor | None = None,
        w2_scale: torch.Tensor | None = None,
        global_num_experts: int | None = None,
    ) -> None:
        num_experts = w13_weight.size(0)
        self._global_num_experts = (
            global_num_experts if global_num_experts is not None else num_experts
        )

        self.capacity = capacity
        self._num_experts = num_experts
        self.hits = 0
        self.misses = 0
        self._overflow_warned = False

        if w13_weight.device.type == "cpu":
            cuda_device = torch.accelerator.current_accelerator()
            self._cpu_w13: torch.Tensor = (
                w13_weight if w13_weight.is_pinned() else w13_weight.pin_memory()
            )
            self._cpu_w2: torch.Tensor = (
                w2_weight if w2_weight.is_pinned() else w2_weight.pin_memory()
            )
        else:
            cuda_device = w13_weight.device
            self._cpu_w13 = w13_weight.cpu().pin_memory()
            self._cpu_w2 = w2_weight.cpu().pin_memory()

        self._buf_w13: torch.Tensor = torch.empty(
            capacity,
            *w13_weight.shape[1:],
            dtype=w13_weight.dtype,
            device=cuda_device,
        )
        self._buf_w2: torch.Tensor = torch.empty(
            capacity,
            *w2_weight.shape[1:],
            dtype=w2_weight.dtype,
            device=cuda_device,
        )

        if w13_scale is not None and w2_scale is not None:
            self._cpu_w13_scale: torch.Tensor | None = w13_scale.cpu()
            self._cpu_w2_scale: torch.Tensor | None = w2_scale.cpu()
            self._buf_w13_scale: torch.Tensor | None = torch.empty(
                capacity,
                *w13_scale.shape[1:],
                dtype=w13_scale.dtype,
                device=cuda_device,
            )
            self._buf_w2_scale: torch.Tensor | None = torch.empty(
                capacity,
                *w2_scale.shape[1:],
                dtype=w2_scale.dtype,
                device=cuda_device,
            )
        else:
            self._cpu_w13_scale = None
            self._cpu_w2_scale = None
            self._buf_w13_scale = None
            self._buf_w2_scale = None

        # LFRU state: {expert_id: [slot, freq, last_access_clock]}
        # Eviction score = freq / (clock - last_access + 1). Lower = evict first.
        self._lru: dict[int, list] = {}
        self._clock: int = 0
        self._free_slots: list[int] = list(range(capacity))

        # Persistent GPU mapping tensor: _mapping[expert_id] = slot.
        self._mapping: torch.Tensor = torch.zeros(
            num_experts, dtype=torch.int32, device=cuda_device
        )

    @property
    def buf_w13(self) -> torch.Tensor:
        return self._buf_w13

    @property
    def buf_w2(self) -> torch.Tensor:
        return self._buf_w2

    @property
    def buf_w13_scale(self) -> torch.Tensor | None:
        return self._buf_w13_scale

    @property
    def buf_w2_scale(self) -> torch.Tensor | None:
        return self._buf_w2_scale

    def invalidate(self, expert_id: int) -> None:
        """Remove *expert_id* from the cache, returning its slot to the free
        list.  No-op if the expert is not currently cached."""
        if expert_id in self._lru:
            entry = self._lru.pop(expert_id)
            self._free_slots.append(entry[0])

    def _ensure_expert(self, expert_id: int, requied_expert_id: list) -> None:
        """Load *expert_id* into the GPU buffer if not already cached.

        On a miss, copies weights from CPU pinned memory into a free or
        evicted slot using LFRU eviction policy and updates the mapping.
        """
        if expert_id in self._lru:
            # Cache hit
            self._clock += 1
            entry = self._lru[expert_id]
            entry[1] += 1  # freq
            entry[2] = self._clock  # last access
            self.hits += 1
            return

        # Cache miss
        if self._free_slots:
            slot = self._free_slots.pop()
        else:
            # Evict entry with lowest freq/age score
            best_key = None
            best_score = float("inf")
            for k, (s, freq, last) in self._lru.items():
                if k in requied_expert_id:
                    continue
                age = self._clock - last + 1
                score = freq / age
                if score < best_score:
                    best_score = score
                    best_key = k
            slot = self._lru.pop(best_key)[0]

        # Copy expert weights from CPU to GPU slot
        self._buf_w13[slot].copy_(self._cpu_w13[expert_id], non_blocking=True)
        self._buf_w2[slot].copy_(self._cpu_w2[expert_id], non_blocking=True)
        if self._buf_w13_scale is not None:
            assert self._cpu_w13_scale is not None
            assert self._cpu_w2_scale is not None
            assert self._buf_w2_scale is not None
            self._buf_w13_scale[slot].copy_(
                self._cpu_w13_scale[expert_id], non_blocking=True
            )
            self._buf_w2_scale[slot].copy_(
                self._cpu_w2_scale[expert_id], non_blocking=True
            )

        self._clock += 1
        self._lru[expert_id] = [slot, 1, self._clock]
        self._mapping[expert_id] = slot
        self.misses += 1

    def _log_stats(self) -> None:
        """Log cache hit/miss statistics at debug level."""
        total = self.hits + self.misses
        if total > 0:
            logger.debug(
                "Expert cache: %d hits, %d misses (%.1f%% hit rate)",
                self.hits,
                self.misses,
                100.0 * self.hits / total,
            )

    @torch.compiler.disable
    def prepare(
        self, topk_ids: torch.Tensor
    ) -> Generator[ExpertWeightResult, None, None]:
        """Populate the GPU buffer and yield slot-remapped expert IDs.

        When the number of unique experts fits within *capacity*, yields a
        single ``ExpertWeightResult`` covering all tokens (both
        ``token_indices`` and ``expert_map`` are ``None``).

        When unique experts exceed capacity and ``chunk_on_overflow`` is
        ``True``, experts are split into *capacity*-sized chunks and
        processed EP-style.  Each chunk yields an ``ExpertWeightResult``
        with:

        * ``expert_map`` — maps global expert → buffer slot (-1 if not in
          this chunk).  Callers must pass this to the kernel.
        * ``token_indices`` — **all** tokens that reference any expert in
          this chunk (may overlap between chunks).
        * ``topk_ids`` — remapped via ``expert_map``; out-of-chunk entries
          are set to -1.  Callers must **zero the corresponding
          ``topk_weights``** and clamp sentinels to 0 before invoking the
          kernel.
        * Outputs for tokens that span multiple chunks must be
          **accumulated** (``+=``).

        When ``chunk_on_overflow`` is ``False``, the old truncation
        behaviour is used: only the last *capacity* unique experts are
        kept, ``token_indices`` and ``expert_map`` are ``None``.
        """
        unique_ids = topk_ids.unique().tolist()

        if len(unique_ids) <= self.capacity:
            # Single chunk: all experts fit in the buffer.
            for expert_id in unique_ids:
                self._ensure_expert(expert_id, unique_ids)
            remapped_ids = self._mapping[topk_ids.long()].to(
                dtype=topk_ids.dtype
            )
            self._log_stats()
            yield ExpertWeightResult(
                w1=self._buf_w13,
                w2=self._buf_w2,
                topk_ids=remapped_ids,
                w1_scale=self._buf_w13_scale,
                w2_scale=self._buf_w2_scale,
            )
            return

        # --- Overflow path ---
        if not self._overflow_warned:
            logger.warning(
                "CachedWeightProvider.prepare() called with %d unique "
                "experts but capacity is only %d.  "
                "%s.  This is expected during prefill with large batches.",
                len(unique_ids),
                self.capacity,
                (
                    "Yielding %d EP-style chunk(s)"
                    % ((len(unique_ids) + self.capacity - 1) // self.capacity)
                ),
            )
            self._overflow_warned = True

        # EP-style chunking: each chunk processes all tokens that reference
        # any of its experts.  Tokens whose top-k spans multiple chunks are
        # processed in every relevant chunk; the caller accumulates results.
        global_num_experts = self._global_num_experts
        device = topk_ids.device

        for i in range(0, len(unique_ids), self.capacity):
            chunk = unique_ids[i : i + self.capacity]

            # Load this chunk's experts into the GPU buffer.
            for expert_id in chunk:
                self._ensure_expert(expert_id, chunk)

            # Build expert_map for this chunk.
            expert_map = torch.full(
                (global_num_experts,), -1, dtype=torch.int32, device=device
            )
            for expert_id in chunk:
                expert_map[expert_id] = self._lru[expert_id][0]

            # Find ALL tokens that reference any expert in this chunk.
            chunk_tensor = torch.tensor(chunk, device=device)
            in_chunk = torch.isin(topk_ids, chunk_tensor)  # [T, top_k]
            token_mask = in_chunk.any(dim=1)  # [T]

            if not token_mask.any():
                self._log_stats()
                continue

            token_indices = token_mask.nonzero(as_tuple=True)[0]

            # Remap topk_ids: expert_map gives buffer slot for in-chunk
            # experts, -1 for out-of-chunk experts.
            sliced = topk_ids[token_indices].long()
            remapped_ids = expert_map[sliced]  # [n_sel, top_k]
            remapped_ids = remapped_ids.to(dtype=topk_ids.dtype)

            self._log_stats()
            yield ExpertWeightResult(
                w1=self._buf_w13,
                w2=self._buf_w2,
                topk_ids=remapped_ids,
                w1_scale=self._buf_w13_scale,
                w2_scale=self._buf_w2_scale,
                token_indices=token_indices,
                expert_map=expert_map,
            )
