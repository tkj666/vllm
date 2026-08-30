# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import gc
import threading
import time
import weakref
from contextlib import nullcontext
from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.model_executor.layers.fused_moe.expert_cache import (
    ExpertCacheBindingError,
    ExpertCacheError,
    ExpertCacheLoadError,
    ExpertCacheStaleHandleError,
    ExpertLoadHandle,
    ExpertSlotSnapshot,
    ExpertSlotState,
    ExpertWeightBundle,
    FIFOExpertCachePolicy,
    LayerBinding,
    StreamedExpertCache,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEKernelModularImpl,
    StreamedMoEBuffers,
)
from vllm.model_executor.model_loader.utils import device_loading_context

pytestmark = pytest.mark.cpu_test


@dataclass
class _FakeEvent:
    completed: bool = True
    synchronize_calls: int = 0
    query_calls: int = 0
    failure: Exception | None = None

    def query(self) -> bool:
        self.query_calls += 1
        return self.completed

    def synchronize(self) -> None:
        self.synchronize_calls += 1
        if self.failure is not None:
            raise self.failure
        self.completed = True


class _RaisingQueryEvent(_FakeEvent):
    def query(self) -> bool:
        self.query_calls += 1
        raise RuntimeError("event query failed")


@dataclass(frozen=True)
class _Submission:
    label: str
    wait_for: _FakeEvent | None
    destination_ptrs: tuple[int, ...]
    ready: _FakeEvent


class _RecordingCoordinator:
    def __init__(self, *, complete_copies: bool = True) -> None:
        self.complete_copies = complete_copies
        self.submissions: list[_Submission] = []
        self.compute_waits: list[tuple[_FakeEvent, object | None]] = []
        self.last_use_events: list[_FakeEvent] = []
        self.fail_next_after_copy = False

    def submit_copy(
        self,
        source: ExpertWeightBundle,
        destination: ExpertWeightBundle,
        *,
        wait_for: _FakeEvent | None,
        label: str,
    ) -> _FakeEvent:
        destination.copy_from_(source, non_blocking=False)
        if self.fail_next_after_copy:
            self.fail_next_after_copy = False
            raise RuntimeError("copy submission failed")
        ready = _FakeEvent(completed=self.complete_copies)
        self.submissions.append(
            _Submission(label, wait_for, destination.data_ptrs, ready)
        )
        return ready

    def wait_ready(
        self,
        event: _FakeEvent,
        compute_stream: object | None,
    ) -> None:
        self.compute_waits.append((event, compute_stream))

    def record_last_use(self, compute_stream: object | None) -> _FakeEvent:
        del compute_stream
        event = _FakeEvent(completed=False)
        self.last_use_events.append(event)
        return event


class _BlockingEvent:
    def __init__(self) -> None:
        self._completed = threading.Event()
        self.synchronize_entered = threading.Event()
        self.synchronize_calls = 0

    def query(self) -> bool:
        return self._completed.is_set()

    def synchronize(self) -> None:
        self.synchronize_calls += 1
        self.synchronize_entered.set()
        if not self._completed.wait(timeout=5):
            raise TimeoutError("controlled expert copy did not complete")

    def complete(self) -> None:
        self._completed.set()


@dataclass(frozen=True)
class _ControlledSubmission:
    label: str
    wait_for: _FakeEvent | _BlockingEvent | None
    ready: _BlockingEvent


class _ControlledCoordinator:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._block_next_submission = False
        self._submit_release = threading.Event()
        self.submit_entered = threading.Event()
        self.submissions: list[_ControlledSubmission] = []
        self.compute_waits: list[tuple[_BlockingEvent, object | None]] = []

    def block_next_submission(self) -> None:
        with self._condition:
            self._block_next_submission = True
            self._submit_release.clear()
            self.submit_entered.clear()

    def release_submission(self) -> None:
        self._submit_release.set()

    def submit_copy(
        self,
        source: ExpertWeightBundle,
        destination: ExpertWeightBundle,
        *,
        wait_for: _FakeEvent | _BlockingEvent | None,
        label: str,
    ) -> _BlockingEvent:
        destination.copy_from_(source, non_blocking=False)
        with self._condition:
            block = self._block_next_submission
            self._block_next_submission = False
        self.submit_entered.set()
        if block and not self._submit_release.wait(timeout=5):
            raise TimeoutError("controlled expert submission was not released")
        ready = _BlockingEvent()
        with self._condition:
            self.submissions.append(_ControlledSubmission(label, wait_for, ready))
            self._condition.notify_all()
        return ready

    def wait_for_submissions(
        self,
        count: int,
        *,
        timeout: float = 5,
    ) -> tuple[_ControlledSubmission, ...]:
        with self._condition:
            if not self._condition.wait_for(
                lambda: len(self.submissions) >= count,
                timeout=timeout,
            ):
                raise TimeoutError(f"expected {count} controlled submissions")
            return tuple(self.submissions)

    def wait_ready(
        self,
        event: _BlockingEvent,
        compute_stream: object | None,
    ) -> None:
        self.compute_waits.append((event, compute_stream))

    def record_last_use(self, compute_stream: object | None) -> _FakeEvent:
        del compute_stream
        return _FakeEvent(completed=False)


def test_queued_cuda_event_is_not_consumed_before_native_issue() -> None:
    from vllm.model_executor.layers.fused_moe.expert_cache import (
        _QueuedCudaExpertCacheEvent,
    )

    class NativeScheduler:
        def __init__(self) -> None:
            self.issued = False
            self.query_calls = 0
            self.wait_calls = 0

        def query_urgent_issued(self, cookie: int) -> bool:
            assert cookie == 7
            self.query_calls += 1
            return self.issued

        def wait_urgent_issued(self, cookie: int) -> None:
            assert cookie == 7
            self.wait_calls += 1
            self.issued = True

    class CudaEvent:
        cuda_event = 123

        def __init__(self) -> None:
            self.query_calls = 0
            self.synchronize_calls = 0

        def query(self) -> bool:
            self.query_calls += 1
            return True

        def synchronize(self) -> None:
            self.synchronize_calls += 1

    class Stream:
        def __init__(self) -> None:
            self.waited_events: list[object] = []

        def wait_event(self, event: object) -> None:
            self.waited_events.append(event)

    scheduler = NativeScheduler()
    cuda_event = CudaEvent()
    event = _QueuedCudaExpertCacheEvent(cuda_event, scheduler, 7)

    assert not event.query()
    assert cuda_event.query_calls == 0
    scheduler.issued = True
    assert event.query()
    assert cuda_event.query_calls == 1

    scheduler = NativeScheduler()
    cuda_event = CudaEvent()
    event = _QueuedCudaExpertCacheEvent(cuda_event, scheduler, 7)
    stream = Stream()
    event.wait_on_stream(stream)

    assert scheduler.wait_calls == 1
    assert stream.waited_events == [cuda_event]
    assert cuda_event.synchronize_calls == 0


class _NewestFirstPolicy:
    def __init__(self) -> None:
        self.calls: list[tuple[int, ...]] = []

    def select_slot(
        self,
        candidates: tuple[ExpertSlotSnapshot, ...],
        *,
        key: tuple[int, int, str],
    ) -> int | None:
        del key
        self.calls.append(tuple(slot.index for slot in candidates))
        if not candidates:
            return None
        return max(candidates, key=lambda slot: slot.fifo_age or -1).index


def _bundle(value: int, *, format_class: str = "bf16-triton") -> ExpertWeightBundle:
    return ExpertWeightBundle(
        format_class,
        {
            "w13_weight": torch.full((2, 3), value, dtype=torch.bfloat16),
            "w2_weight": torch.full((3, 1), -value, dtype=torch.bfloat16),
        },
    )


def test_native_hard_submit_allocates_queued_event_before_enqueue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.layers.fused_moe import expert_cache
    from vllm.model_executor.layers.fused_moe.expert_cache import (
        CudaExpertTransferCoordinator,
    )

    class Scheduler:
        def __init__(self) -> None:
            self.jobs = []

        def enqueue_urgent(self, job) -> None:
            self.jobs.append(job)

    source = _bundle(1)
    destination = _bundle(0)
    ready = SimpleNamespace(cuda_event=123)
    scheduler = Scheduler()
    coordinator = CudaExpertTransferCoordinator.__new__(CudaExpertTransferCoordinator)
    coordinator._native_ready_events = {destination.data_ptrs: ready}
    coordinator._next_native_cookie = 0
    coordinator._native_copy_scheduler = scheduler

    def fail_queued_event_allocation(*args, **kwargs):
        del args, kwargs
        raise MemoryError("queued event allocation failed")

    monkeypatch.setattr(
        expert_cache,
        "_QueuedCudaExpertCacheEvent",
        fail_queued_event_allocation,
    )

    with pytest.raises(MemoryError, match="queued event allocation failed"):
        coordinator._submit_native_copy(
            source,
            destination,
            wait_for=None,
            label="expert_cache:hard:0:1",
        )

    assert scheduler.jobs == []


@pytest.mark.parametrize("native_enabled", [False, True])
def test_submit_copy_arms_and_unwraps_queued_dependency(
    native_enabled: bool,
) -> None:
    from vllm.model_executor.layers.fused_moe.expert_cache import (
        CudaExpertTransferCoordinator,
        _QueuedCudaExpertCacheEvent,
    )

    class Scheduler:
        def wait_urgent_issued(self, cookie: int) -> None:
            actions.append(("arm", cookie))

    class CudaEvent:
        cuda_event = 456

    actions = []
    scheduler = Scheduler()
    cuda_event = CudaEvent()
    queued = _QueuedCudaExpertCacheEvent(cuda_event, scheduler, 7)
    ordinary = CudaEvent()

    def capture_submit(source, destination, *, wait_for, label):
        del source, destination, label
        actions.append(("submit", wait_for))
        return _FakeEvent()

    coordinator = CudaExpertTransferCoordinator.__new__(CudaExpertTransferCoordinator)
    coordinator.device = torch.device("cpu")
    coordinator._submission_lock = threading.Lock()
    coordinator._native_copy_scheduler = object() if native_enabled else None
    coordinator._native_submission_pause_depth = 0
    coordinator._copy_scheduler = None
    coordinator._submit_native_copy = capture_submit
    coordinator._submit_copy = capture_submit
    source = SimpleNamespace(is_pinned=True)
    destination = SimpleNamespace(tensors={"weight": torch.empty(1)})

    coordinator.submit_copy(
        source,
        destination,
        wait_for=queued,
        label="queued",
    )
    coordinator.submit_copy(
        source,
        destination,
        wait_for=ordinary,
        label="ordinary",
    )

    assert actions == [
        ("arm", 7),
        ("submit", cuda_event),
        ("submit", ordinary),
    ]


def _binding(
    layer_id: int,
    reserved_slot_indices: tuple[int, ...],
    *,
    num_experts: int = 4,
) -> LayerBinding:
    return LayerBinding(
        layer_id=layer_id,
        format_class="bf16-triton",
        host_bundles={i: _bundle(10 * layer_id + i + 1) for i in range(num_experts)},
        reserved_slot_indices=reserved_slot_indices,
    )


def _wait_for_prefetch_candidates_discarded(
    cache: StreamedExpertCache,
    count: int,
    *,
    timeout: float = 5,
) -> None:
    deadline = time.monotonic() + timeout
    while (
        cache.stats.prefetch_candidates_discarded != count
        and time.monotonic() < deadline
    ):
        time.sleep(0.001)
    assert cache.stats.prefetch_candidates_discarded == count


def test_ordered_unique_bulk_converts_routing_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.layers.fused_moe.cached_expert_layer import (
        _ordered_unique,
    )

    routing_ids = torch.tensor([[4, 2, 4], [1, 2, 3]], dtype=torch.int32)

    def reject_elementwise_iteration(self: torch.Tensor):
        raise AssertionError("routing IDs must be converted to host scalars in bulk")

    monkeypatch.setattr(torch.Tensor, "__iter__", reject_elementwise_iteration)

    assert _ordered_unique(routing_ids) == (4, 2, 1, 3)


def test_shared_host_storage_keeps_layer_expert_bundles_distinct() -> None:
    storage = torch.UntypedStorage(256, device="cpu")

    def view(
        storage: torch.UntypedStorage,
        offset: int,
        shape: tuple[int, ...],
    ) -> torch.Tensor:
        return torch.empty(0, dtype=torch.bfloat16).set_(storage, offset, shape)

    layer0_w13 = torch.nn.Parameter(view(storage, 0, (2, 2, 3)), requires_grad=False)
    layer0_w2 = torch.nn.Parameter(view(storage, 16, (2, 3, 1)), requires_grad=False)
    layer1_w13 = torch.nn.Parameter(view(storage, 32, (2, 2, 3)), requires_grad=False)
    layer1_w2 = torch.nn.Parameter(view(storage, 48, (2, 3, 1)), requires_grad=False)
    parameters = (layer0_w13, layer0_w2, layer1_w13, layer1_w2)
    assert len({tensor.untyped_storage().data_ptr() for tensor in parameters}) == 1

    for layer_id, (w13, w2) in enumerate(
        ((layer0_w13, layer0_w2), (layer1_w13, layer1_w2))
    ):
        for expert_id in range(2):
            value = 10 * (layer_id + 1) + expert_id
            w13.data[expert_id].fill_(value)
            w2.data[expert_id].fill_(-value)

    def bundles(
        w13: torch.nn.Parameter, w2: torch.nn.Parameter
    ) -> dict[int, ExpertWeightBundle]:
        return {
            expert_id: ExpertWeightBundle(
                "bf16-triton",
                {
                    "w13_weight": w13.data[expert_id],
                    "w2_weight": w2.data[expert_id],
                },
            )
            for expert_id in range(2)
        }

    layer0_bundles = bundles(layer0_w13, layer0_w2)
    layer1_bundles = bundles(layer1_w13, layer1_w2)
    parameter_refs = tuple(weakref.ref(parameter) for parameter in parameters)
    del parameters, layer0_w13, layer0_w2, layer1_w13, layer1_w2, w13, w2, storage
    gc.collect()
    assert all(parameter_ref() is None for parameter_ref in parameter_refs)

    cache = StreamedExpertCache([_bundle(0)], shared_slot_indices=(0,))
    layers = (
        cache.bind(
            LayerBinding(0, "bf16-triton", layer0_bundles, reserved_slot_indices=())
        ),
        cache.bind(
            LayerBinding(1, "bf16-triton", layer1_bundles, reserved_slot_indices=())
        ),
    )

    for layer_id, expert_id in ((0, 1), (1, 0), (0, 0), (1, 1)):
        expected = 10 * (layer_id + 1) + expert_id
        with layers[layer_id].acquire(expert_id) as lease:
            assert torch.equal(
                lease.bundle.tensors["w13_weight"],
                torch.full((2, 3), expected, dtype=torch.bfloat16),
            )
            assert torch.equal(
                lease.bundle.tensors["w2_weight"],
                torch.full((3, 1), -expected, dtype=torch.bfloat16),
            )


def test_bundle_layout_and_arena_boundaries_are_immutable() -> None:
    bundle = _bundle(1)
    assert bundle.nbytes == sum(
        tensor.numel() * tensor.element_size() for tensor in bundle.tensors.values()
    )
    with pytest.raises(TypeError):
        bundle.tensors["replacement"] = torch.empty(1)  # type: ignore[index]

    cache = StreamedExpertCache(
        [_bundle(0) for _ in range(4)],
        shared_slot_indices=(2, 3),
    )
    cache.bind(_binding(0, (0,)))
    cache.bind(_binding(1, (1,)))

    snapshots = cache.slot_snapshots()
    assert [slot.reserved_layer_id for slot in snapshots] == [0, 1, None, None]
    assert [slot.is_shared for slot in snapshots] == [False, False, True, True]
    assert all(slot.state is ExpertSlotState.ABSENT for slot in snapshots)


def test_binding_rejects_reserved_overlap_and_shared_slots() -> None:
    cache = StreamedExpertCache(
        [_bundle(0) for _ in range(3)],
        shared_slot_indices=(2,),
    )
    cache.bind(_binding(0, (0,)))

    with pytest.raises(ExpertCacheBindingError, match="already reserved"):
        cache.bind(_binding(1, (0,)))
    with pytest.raises(ExpertCacheBindingError, match="is shared"):
        cache.bind(_binding(2, (2,)))


def test_placement_order_and_fifo_keep_reserved_ranges_private() -> None:
    coordinator = _RecordingCoordinator()
    cache = StreamedExpertCache(
        [_bundle(0) for _ in range(4)],
        shared_slot_indices=(2, 3),
        coordinator=coordinator,
    )
    layer0 = cache.bind(_binding(0, (0,)))
    layer1 = cache.bind(_binding(1, (1,)))

    assert layer0.request(0).slot_index == 0
    assert layer0.request(1).slot_index == 2
    assert layer1.request(0).slot_index == 1
    assert layer1.request(1).slot_index == 3

    # A hit does not refresh FIFO age. Shared victims precede reserved victims.
    assert layer0.request(1).slot_index == 2
    assert layer1.request(2).slot_index == 2
    assert layer0.slot_for(1) is None
    assert layer0.slot_for(0) == 0

    shared_lease0 = layer1.acquire(2)
    shared_lease1 = layer1.acquire(1)
    # With both shared slots leased, layer 1 can only evict its own reserved slot.
    assert layer1.request(3).slot_index == 1
    assert layer0.slot_for(0) == 0
    shared_lease0.release()
    shared_lease1.release()


def test_wave_claim_protects_shared_hit_from_later_miss() -> None:
    coordinator = _RecordingCoordinator()
    cache = StreamedExpertCache(
        [_bundle(0) for _ in range(4)],
        shared_slot_indices=(2, 3),
        coordinator=coordinator,
    )
    layer = cache.bind(_binding(0, (0, 1), num_experts=5))

    for expert_id in range(4):
        layer.request(expert_id)

    hit = layer.claim(2)
    miss = layer.claim(4)

    assert hit.slot_index == 2
    assert miss.slot_index == 3
    assert layer.slot_for(2) == 2
    assert layer.slot_for(4) == 3
    assert coordinator.compute_waits == []

    leases = (hit.acquire("compute-stream"), miss.acquire("compute-stream"))
    assert len(coordinator.compute_waits) == 2
    for lease in leases:
        lease.release()


def test_claim_resident_protects_routed_hits_until_a_slot_is_released() -> None:
    coordinator = _RecordingCoordinator()
    cache = StreamedExpertCache(
        [_bundle(0) for _ in range(3)],
        coordinator=coordinator,
    )
    binding = cache.bind(_binding(0, (0, 1, 2), num_experts=4))
    for expert_id in (1, 2, 3):
        binding.request(expert_id)

    resident_claims, pending, pending_claims = binding.claim_resident((1, 2, 3, 0))

    assert tuple(claim.key[1] for claim in resident_claims) == (1, 2, 3)
    assert pending == (0,)
    assert pending_claims == ()
    assert binding.try_claim(0) is None

    reusable_slot = resident_claims[0].slot_index
    last_use = _FakeEvent(completed=False)
    resident_claims[0].acquire().release(last_use_event=last_use)
    cold_claim = binding.try_claim(0)

    assert cold_claim is not None
    assert cold_claim.slot_index == reusable_slot
    assert coordinator.submissions[-1].wait_for is last_use
    for claim in resident_claims[1:]:
        claim.acquire().release(last_use_event=_FakeEvent(completed=False))
    cold_claim.release()
    assert cache.stats.requests == 7
    assert cache.stats.hits == 3
    assert cache.stats.misses == 4
    assert cache.stats.loads == 4


def test_claim_resident_starts_first_pending_load_during_classification() -> None:
    coordinator = _RecordingCoordinator()
    cache = StreamedExpertCache(
        [_bundle(0), _bundle(0)],
        coordinator=coordinator,
    )
    binding = cache.bind(_binding(0, (0, 1), num_experts=3))
    binding.request(1)

    resident_claims, pending, pending_claims = binding.claim_resident((1, 0, 2))

    assert tuple(claim.key[1] for claim in resident_claims) == (1,)
    assert pending == (0, 2)
    assert tuple(claim.key[1] for claim in pending_claims) == (0,)
    assert coordinator.submissions[-1].label == "expert_cache:hard:0:0"
    assert cache.stats.requests == 3
    assert cache.stats.hits == 1
    assert cache.stats.misses == 2
    assert cache.stats.loads == 2
    resident_claims[0].release()
    pending_claims[0].release()


def test_claim_resident_bounds_initial_pending_runway() -> None:
    coordinator = _RecordingCoordinator()
    cache = StreamedExpertCache(
        [_bundle(0) for _ in range(4)],
        coordinator=coordinator,
    )
    binding = cache.bind(_binding(0, (0, 1, 2, 3), num_experts=4))

    resident, pending, pending_claims = binding.claim_resident(
        (0, 1, 2, 3),
        max_pending_claims=2,
    )

    assert resident == ()
    assert pending == (0, 1, 2, 3)
    assert tuple(claim.key[1] for claim in pending_claims) == (0, 1)
    assert [submission.label for submission in coordinator.submissions] == [
        "expert_cache:hard:0:0",
        "expert_cache:hard:0:1",
    ]
    for claim in pending_claims:
        claim.release()
    with pytest.raises(ValueError, match="max_pending_claims"):
        binding.claim_resident((2,), max_pending_claims=0)


def test_claim_resident_refresh_is_layer_scoped_and_includes_shared() -> None:
    coordinator = _RecordingCoordinator(complete_copies=False)
    cache = StreamedExpertCache(
        [_bundle(0), _bundle(0), _bundle(0)],
        shared_slot_indices=(2,),
        coordinator=coordinator,
    )
    current = cache.bind(_binding(0, (0,), num_experts=1))
    unrelated = cache.bind(_binding(1, (1,), num_experts=2))
    unrelated.request(0)
    reserved_ready = coordinator.submissions[-1].ready
    unrelated.request(1)
    shared_ready = coordinator.submissions[-1].ready
    shared_ready.completed = True
    reserved_query_calls = reserved_ready.query_calls
    shared_query_calls = shared_ready.query_calls

    resident, pending, pending_claims = current.claim_resident(
        (0,),
        max_pending_claims=1,
    )

    assert resident == ()
    assert pending == (0,)
    assert len(pending_claims) == 1
    assert reserved_ready.query_calls == reserved_query_calls
    assert shared_ready.query_calls == shared_query_calls + 1
    pending_claims[0].release()


def test_acquire_refreshes_only_its_record_slot() -> None:
    coordinator = _RecordingCoordinator(complete_copies=False)
    cache = StreamedExpertCache(
        [_bundle(0), _bundle(0)],
        coordinator=coordinator,
    )
    current = cache.bind(_binding(0, (0,), num_experts=1))
    unrelated = cache.bind(_binding(1, (1,), num_experts=1))
    handle = current.request(0)
    current_ready = coordinator.submissions[-1].ready
    unrelated.request(0)
    unrelated_ready = coordinator.submissions[-1].ready
    assert current_ready.query_calls == unrelated_ready.query_calls == 1

    lease = handle.acquire()

    assert current_ready.query_calls == 2
    assert unrelated_ready.query_calls == 1
    lease.release()


def test_claim_readiness_is_nonblocking_and_rolls_back_query_failure() -> None:
    coordinator = _RecordingCoordinator(complete_copies=False)
    cache = StreamedExpertCache(
        [_bundle(0), _bundle(0)],
        coordinator=coordinator,
    )
    layer = cache.bind(_binding(0, (0, 1), num_experts=2))
    pending = layer.claim(0)
    pending_ready = coordinator.submissions[-1].ready

    assert not pending.is_ready()
    assert pending_ready.synchronize_calls == 0
    pending_ready.completed = True
    assert pending.is_ready()
    pending.release()

    failed = layer.claim(1)
    failed_ready = _RaisingQueryEvent(completed=False)
    cache._slots[failed.slot_index].ready_event = failed_ready

    with pytest.raises(ExpertCacheLoadError, match="event failed"):
        failed.is_ready()
    assert failed.released
    assert cache.slot_snapshots()[failed.slot_index].state is ExpertSlotState.ABSENT
    assert cache.stats.load_failures == 1


def test_custom_policy_controls_victim_within_placement_tier() -> None:
    policy = _NewestFirstPolicy()
    cache = StreamedExpertCache(
        [_bundle(0), _bundle(0)],
        shared_slot_indices=(0, 1),
        policy=policy,
    )
    layer = cache.bind(_binding(0, ()))

    assert layer.request(0).slot_index == 0
    assert layer.request(1).slot_index == 1
    assert layer.request(2).slot_index == 1
    assert policy.calls[-1] == (0, 1)


def test_fifo_subclass_uses_custom_policy_path() -> None:
    class ReverseFIFO(FIFOExpertCachePolicy):
        def __init__(self) -> None:
            self.calls: list[tuple[int, ...]] = []

        def select_slot(
            self,
            candidates: tuple[ExpertSlotSnapshot, ...],
            *,
            key: tuple[int, int, str],
        ) -> int | None:
            del key
            self.calls.append(tuple(slot.index for slot in candidates))
            return max(slot.index for slot in candidates)

    policy = ReverseFIFO()
    cache = StreamedExpertCache(
        [_bundle(0), _bundle(0)],
        shared_slot_indices=(0, 1),
        policy=policy,
    )
    layer = cache.bind(_binding(0, ()))
    layer.request(0)
    layer.request(1)

    assert layer.request(2).slot_index == 1
    assert policy.calls == [(0, 1)]


def test_prefetch_is_promoted_and_duplicate_requests_coalesce() -> None:
    coordinator = _RecordingCoordinator(complete_copies=False)
    cache = StreamedExpertCache([_bundle(0)], coordinator=coordinator)
    layer = cache.bind(_binding(0, (0,)))

    prefetched = layer.prefetch([0, 0])
    assert len(prefetched) == 1
    assert len(coordinator.submissions) == 1
    assert cache.slot_snapshots()[0].state is ExpertSlotState.LOADING

    hard = layer.request(0)
    assert hard is prefetched[0]
    assert len(coordinator.submissions) == 1

    lease = hard.acquire("compute-stream")
    assert cache.slot_snapshots()[0].state is ExpertSlotState.LEASED
    assert coordinator.compute_waits == [
        (coordinator.submissions[0].ready, "compute-stream")
    ]
    assert torch.equal(
        lease.bundle.tensors["w13_weight"],
        _bundle(1).tensors["w13_weight"],
    )
    lease.release()
    assert cache.slot_snapshots()[0].state is ExpertSlotState.RESIDENT
    assert cache.stats.prefetch_promotions == 1


def test_load_record_does_not_retain_unused_handle() -> None:
    cache = StreamedExpertCache([_bundle(0)])
    layer = cache.bind(_binding(0, (0,), num_experts=1))
    handle = layer.request(0)
    slot_index = handle.slot_index
    handle_ref = weakref.ref(handle)
    assert layer.request(0) is handle

    del handle

    assert handle_ref() is None
    assert layer.request(0).slot_index == slot_index


def test_native_prefetch_reservation_reconciles_only_committed_prefix() -> None:
    cache = StreamedExpertCache(
        [_bundle(0), _bundle(0), _bundle(0)],
        shared_slot_indices=(2,),
    )
    layer = cache.bind(_binding(0, (0, 1), num_experts=5))
    layer.request(0)
    layer.request(1)
    layer.request(2)
    ready_events = {0: _FakeEvent(completed=False), 1: _FakeEvent(completed=False)}

    reservation = layer.reserve_prefetch(ready_events)
    assert reservation is not None
    assert [copy.expert_id for copy in reservation.copies] == [3, 4]
    assert [copy.slot_index for copy in reservation.copies] == [0, 1]
    assert layer.slot_for(0) is None
    assert layer.slot_for(1) is None
    assert layer.slot_for(2) == 2
    assert [snapshot.state for snapshot in cache.slot_snapshots()[:2]] == [
        ExpertSlotState.PREFETCH_RESERVED,
        ExpertSlotState.PREFETCH_RESERVED,
    ]

    reservation.reconcile((0,))

    assert layer.slot_for(0) is None
    assert layer.slot_for(1) == 1
    assert layer.slot_for(2) == 2
    assert layer.slot_for(3) == 0
    assert layer.slot_for(4) is None
    assert cache.slot_snapshots()[0].state is ExpertSlotState.LOADING
    stats = cache.stats
    assert stats.prefetch_requests == 1
    assert stats.prefetch_started == 1
    assert stats.prefetch_candidates_discarded == 1
    assert stats.evictions == 1

    hard = layer.request(3)
    assert hard.ready_event is ready_events[0]
    assert cache.stats.prefetch_promotions == 1


def test_native_prefetch_reservation_limits_work_to_available_slots() -> None:
    cache = StreamedExpertCache([_bundle(0)], coordinator=_RecordingCoordinator())
    layer = cache.bind(_binding(0, (0,), num_experts=4))
    ready = _FakeEvent(completed=False)

    reservation = layer.reserve_prefetch({0: ready})
    assert reservation is not None
    assert [copy.expert_id for copy in reservation.copies] == [0]
    assert {copy.slot_index for copy in reservation.copies} == {0}

    reservation.reconcile((0,))

    assert layer.slot_for(0) == 0
    assert layer.slot_for(1) is None
    assert layer.slot_for(2) is None
    assert layer.slot_for(3) is None
    assert cache.stats.loads == 1
    assert cache.stats.evictions == 0
    assert cache.stats.prefetch_candidates_discarded == 0


def test_prefetch_refreshes_unrequested_bound_slots_before_dedup() -> None:
    cache = StreamedExpertCache([_bundle(0), _bundle(0)])
    layer = cache.bind(_binding(0, (0, 1), num_experts=3))
    resident = _FakeEvent(completed=True)
    failed = _RaisingQueryEvent(completed=False)
    initial = layer.reserve_prefetch({0: resident, 1: failed})
    assert initial is not None
    initial.reconcile((0, 1))

    replacement = layer.reserve_prefetch({0: _FakeEvent(completed=False)})

    assert replacement is not None
    assert [copy.expert_id for copy in replacement.copies] == [1]
    assert [copy.slot_index for copy in replacement.copies] == [0]
    assert failed.query_calls == 1
    assert cache.stats.load_failures == 1
    replacement.abort()


def test_native_prefetch_reservation_fills_unique_reserved_and_shared_slots() -> None:
    cache = StreamedExpertCache(
        [_bundle(0) for _ in range(4)],
        shared_slot_indices=(2, 3),
    )
    layer = cache.bind(_binding(0, (0, 1), num_experts=6))
    ready_events = {slot_index: _FakeEvent(completed=False) for slot_index in range(4)}

    reservation = layer.reserve_prefetch(ready_events)

    assert reservation is not None
    assert [copy.expert_id for copy in reservation.copies] == [0, 1, 2, 3]
    assert [copy.slot_index for copy in reservation.copies] == [0, 1, 2, 3]
    assert len({copy.expert_id for copy in reservation.copies}) == 4
    assert len({copy.slot_index for copy in reservation.copies}) == 4
    reservation.abort()


def test_native_prefetch_reservation_skips_claimed_shared_slots() -> None:
    cache = StreamedExpertCache(
        [_bundle(0) for _ in range(4)],
        shared_slot_indices=(2, 3),
    )
    current = cache.bind(_binding(0, (0,), num_experts=3))
    target = cache.bind(_binding(1, (1,), num_experts=4))
    current.request(0)
    current.request(1)
    claimed = current.claim(1)
    assert claimed.slot_index == 2
    ready_events = {slot_index: _FakeEvent(completed=False) for slot_index in (1, 2, 3)}

    reservation = target.reserve_prefetch(ready_events)

    assert reservation is not None
    assert [copy.slot_index for copy in reservation.copies] == [1, 3]
    assert current.slot_for(1) == 2
    assert current.try_claim(1) is None
    reservation.abort()
    claimed.release()


def test_native_prefetch_reservation_abort_restores_exact_slot_state() -> None:
    coordinator = _RecordingCoordinator()
    cache = StreamedExpertCache([_bundle(0)], coordinator=coordinator)
    layer = cache.bind(_binding(0, (0,), num_experts=2))
    handle = layer.request(1)
    original_snapshot = cache.slot_snapshots()[0]

    reservation = layer.reserve_prefetch({0: _FakeEvent(completed=False)})
    assert reservation is not None
    reservation.abort()

    restored = cache.slot_snapshots()[0]
    assert restored == original_snapshot
    assert layer.request(1) is handle
    assert cache.stats.prefetch_started == 0


def test_native_prefetch_reservation_rejects_nonprefix_snapshot() -> None:
    cache = StreamedExpertCache([_bundle(0)])
    layer = cache.bind(_binding(0, (0,), num_experts=3))
    reservation = layer.reserve_prefetch({0: _FakeEvent(completed=False)})
    assert reservation is not None

    with pytest.raises(ExpertCacheError, match="ascending copy prefix"):
        reservation.reconcile((1,))

    assert reservation.active
    reservation.abort()


def test_native_prefetch_register_precomputes_shared_only_copy_templates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.layers.fused_moe import cached_expert_layer
    from vllm.model_executor.layers.fused_moe.cached_expert_layer import (
        _NativeDummyExpertPrefetcher,
    )
    from vllm.utils.native_cuda_copy_scheduler import CopyJob, CopySegment

    class Event:
        next_handle = 100

        def __init__(self) -> None:
            self.cuda_event = Event.next_handle
            Event.next_handle += 1
            self.recorded_stream = None

        def record(self, stream: object) -> None:
            self.recorded_stream = stream

    class Binding:
        reserved_slot_indices = ()
        prefetch_slot_indices = (2, 4)

        def prefetch_copy_layouts(self):
            return (
                (0, 2, "source-0", "destination-2"),
                (0, 9, "source-0", "shared-destination"),
                (1, 4, "source-1", "destination-4"),
                (0, 4, "source-0", "destination-4"),
                (1, 2, "source-1", "destination-2"),
            )

    class Scheduler:
        def close(self) -> None:
            pass

    copy_stream = SimpleNamespace(device=torch.device("cpu"))
    calls = []

    def copy_job_template(layer_id, expert_id, source, destination, ready_event):
        calls.append((layer_id, expert_id, source, destination, ready_event))
        return CopyJob(
            cookie=0,
            segments=(CopySegment(expert_id, ready_event.cuda_event, 1),),
            done_event=ready_event.cuda_event,
            label=f"template:{expert_id}",
        )

    monkeypatch.setattr(cached_expert_layer.torch.cuda, "Event", Event)
    monkeypatch.setattr(
        _NativeDummyExpertPrefetcher,
        "_copy_job_template",
        staticmethod(copy_job_template),
    )
    prefetcher = _NativeDummyExpertPrefetcher(
        (0,),
        SimpleNamespace(),
        Scheduler(),
        copy_stream,
    )

    try:
        binding = Binding()
        prefetcher.register(0, binding, 2)
        registration = prefetcher._bindings[0]

        assert [(call[1], call[3]) for call in calls] == [
            (0, "destination-2"),
            (1, "destination-4"),
            (0, "destination-4"),
            (1, "destination-2"),
        ]
        assert set(registration.copy_job_templates) == {
            (0, 2),
            (0, 4),
            (1, 2),
            (1, 4),
        }
        assert all(
            event.recorded_stream is copy_stream
            for event in registration.ready_events.values()
        )
        templates: Any = registration.copy_job_templates
        with pytest.raises(TypeError):
            templates[(0, 2)] = CopyJob(0, ())
    finally:
        prefetcher.close()


@pytest.mark.parametrize(
    ("layouts", "num_experts", "error"),
    [
        (((0, 2, "source", "destination"),), 2, "does not cover"),
        (
            (
                (0, 2, "source", "destination"),
                (0, 2, "source", "destination"),
            ),
            1,
            "duplicate",
        ),
    ],
)
def test_native_prefetch_register_rejects_invalid_template_coverage(
    monkeypatch: pytest.MonkeyPatch,
    layouts: tuple[tuple[int, int, str, str], ...],
    num_experts: int,
    error: str,
) -> None:
    from vllm.model_executor.layers.fused_moe import cached_expert_layer
    from vllm.model_executor.layers.fused_moe.cached_expert_layer import (
        _NativeDummyExpertPrefetcher,
    )
    from vllm.utils.native_cuda_copy_scheduler import CopyJob

    class Event:
        cuda_event = 1

        def record(self, stream: object) -> None:
            pass

    class Binding:
        reserved_slot_indices = (2,)

        def prefetch_copy_layouts(self):
            return layouts

    class Scheduler:
        def close(self) -> None:
            pass

    monkeypatch.setattr(cached_expert_layer.torch.cuda, "Event", Event)
    monkeypatch.setattr(
        _NativeDummyExpertPrefetcher,
        "_copy_job_template",
        staticmethod(lambda *args: CopyJob(0, ())),
    )
    prefetcher = _NativeDummyExpertPrefetcher(
        (0,),
        SimpleNamespace(),
        Scheduler(),
        SimpleNamespace(device=torch.device("cpu")),
    )

    try:
        with pytest.raises(ValueError, match=error):
            prefetcher.register(0, Binding(), num_experts)
        assert 0 not in prefetcher._bindings
    finally:
        prefetcher.close()


def test_native_prefetch_copy_job_reuses_static_template_fields() -> None:
    from vllm.model_executor.layers.fused_moe.cached_expert_layer import (
        _NativeDummyExpertPrefetcher,
    )
    from vllm.utils.native_cuda_copy_scheduler import CopyJob, CopySegment

    class StaticField:
        def __getattribute__(self, name: str):
            raise AssertionError(f"static reservation field was inspected: {name}")

    segments = (CopySegment(11, 22, 33),)
    template = CopyJob(
        cookie=0,
        segments=segments,
        done_event=44,
        label="template-label",
    )
    copy = SimpleNamespace(
        cookie=7,
        wait_for=SimpleNamespace(cuda_event=55),
        source=StaticField(),
        destination=StaticField(),
        ready_event=StaticField(),
    )

    job = _NativeDummyExpertPrefetcher._copy_job(copy, template)

    assert job.cookie == 7
    assert job.segments is segments
    assert job.wait_event == 55
    assert job.done_event == 44
    assert job.label == "template-label"


def test_native_prefetch_planning_is_async_and_cancelable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.layers.fused_moe import cached_expert_layer
    from vllm.model_executor.layers.fused_moe.cached_expert_layer import (
        _NativeDummyExpertPrefetcher,
        _NativeDummyPrefetchBinding,
    )

    reserve_started = threading.Event()
    allow_reserve = threading.Event()

    class Reservation:
        copies: tuple[()] = ()
        active = True
        aborted = False

        def abort(self) -> None:
            self.active = False
            self.aborted = True

    reservation = Reservation()

    class Binding:
        def reserve_prefetch(self, ready_events: object) -> Reservation:
            del ready_events
            reserve_started.set()
            assert allow_reserve.wait(timeout=5)
            return reservation

    class Scheduler:
        closed = False

        def prepare_window(self, jobs: object) -> int:
            pytest.fail(f"canceled planner prepared jobs: {jobs}")

        def close(self) -> None:
            self.closed = True

    class Cache:
        def fail_closed(self, error: Exception) -> None:
            pytest.fail(f"cancelable planner poisoned the cache: {error}")

    monkeypatch.setattr(
        cached_expert_layer.torch.cuda,
        "device",
        lambda device: nullcontext(),
    )
    monkeypatch.setattr(
        cached_expert_layer.torch.cuda.nvtx,
        "range",
        lambda label: nullcontext(),
    )
    scheduler = Scheduler()
    prefetcher = _NativeDummyExpertPrefetcher(
        (0, 1),
        Cache(),
        scheduler,
        SimpleNamespace(device=torch.device("cpu")),
    )
    prefetcher._bindings[1] = _NativeDummyPrefetchBinding(
        Binding(),
        {},
        MappingProxyType({}),
    )
    cancel_done = threading.Event()

    try:
        request = prefetcher.start_after(0)
        assert request is not None
        assert reserve_started.wait(timeout=5)

        threading.Thread(
            target=lambda: (
                prefetcher.cancel_for_demand(1),
                cancel_done.set(),
            ),
            daemon=True,
        ).start()
        assert not cancel_done.wait(timeout=0.05)
        allow_reserve.set()
        assert cancel_done.wait(timeout=5)

        assert request.done.is_set()
        assert reservation.aborted
        assert prefetcher.active_layer_id is None
    finally:
        allow_reserve.set()
        prefetcher.close()
    assert scheduler.closed


def test_native_prefetch_demand_cancels_prepared_inactive_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.layers.fused_moe import cached_expert_layer
    from vllm.model_executor.layers.fused_moe.cached_expert_layer import (
        _NativeDummyExpertPrefetcher,
        _NativeDummyPrefetchBinding,
    )
    from vllm.utils.native_cuda_copy_scheduler import CopyWindowSnapshot

    class Reservation:
        copies: tuple[()] = ()
        active = True
        reconciled = False

        def abort(self) -> None:
            self.active = False

        def reconcile(self, issued: object) -> None:
            assert tuple(issued) == ()
            self.active = False
            self.reconciled = True

    reservation = Reservation()

    class Binding:
        def reserve_prefetch(self, ready_events: object) -> Reservation:
            del ready_events
            return reservation

    class Scheduler:
        released: list[int] = []

        def prepare_window(self, jobs: object) -> int:
            assert tuple(jobs) == ()
            return 13

        def activate_window(
            self,
            handle: int,
            *,
            start_event: int,
            stop_event: int,
        ) -> None:
            pytest.fail(
                f"inactive window {handle} activated with events "
                f"{start_event}, {stop_event}"
            )

        def cancel_and_snapshot(self, handle: int) -> CopyWindowSnapshot:
            return CopyWindowSnapshot(handle, (), (), (), (), (), (), None)

        def release_window(self, handle: int) -> None:
            self.released.append(handle)

        def close(self) -> None:
            pass

    class Cache:
        def fail_closed(self, error: Exception) -> None:
            pytest.fail(f"prepared window cleanup poisoned the cache: {error}")

    monkeypatch.setattr(
        cached_expert_layer.torch.cuda,
        "device",
        lambda device: nullcontext(),
    )
    monkeypatch.setattr(
        cached_expert_layer.torch.cuda.nvtx,
        "range",
        lambda label: nullcontext(),
    )
    scheduler = Scheduler()
    prefetcher = _NativeDummyExpertPrefetcher(
        (0, 1),
        Cache(),
        scheduler,
        SimpleNamespace(device=torch.device("cpu")),
    )
    prefetcher._bindings[1] = _NativeDummyPrefetchBinding(
        Binding(),
        {},
        MappingProxyType({}),
    )

    try:
        request = prefetcher.prepare_after(0)
        assert request is not None
        assert request.done.wait(timeout=5)
        assert request.window is not None
        assert prefetcher.active_layer_id == 1

        prefetcher.cancel_for_demand(1)

        assert reservation.reconciled
        assert scheduler.released == [13]
        assert prefetcher.active_layer_id is None
    finally:
        prefetcher.close()


@pytest.mark.parametrize("activate_before_prepare", [False, True])
@pytest.mark.parametrize("failure_stage", [None, "activation", "record"])
def test_native_prefetch_uses_pending_stop_event_and_releases_window(
    monkeypatch: pytest.MonkeyPatch,
    activate_before_prepare: bool,
    failure_stage: str | None,
) -> None:
    from vllm.model_executor.layers.fused_moe import cached_expert_layer
    from vllm.model_executor.layers.fused_moe.cached_expert_layer import (
        _NativeDummyExpertPrefetcher,
        _NativeDummyPrefetchBinding,
    )
    from vllm.utils.native_cuda_copy_scheduler import CopyWindowSnapshot

    prepare_started = threading.Event()
    allow_prepare = threading.Event()

    class Reservation:
        copies: tuple[()] = ()
        active = True

        def abort(self) -> None:
            self.active = False

        def reconcile(self, issued: object) -> None:
            assert tuple(issued) == ()
            self.active = False

    reservation = Reservation()

    class Binding:
        def reserve_prefetch(self, ready_events: object) -> Reservation:
            del ready_events
            return reservation

    class Scheduler:
        activated: tuple[int, int, int] | None = None
        activations = 0
        released: list[int] = []
        closed = False

        def prepare_window(self, jobs: object) -> int:
            assert tuple(jobs) == ()
            prepare_started.set()
            assert allow_prepare.wait(timeout=5)
            return 11

        def activate_window(
            self,
            handle: int,
            *,
            start_event: int,
            stop_event: int,
        ) -> None:
            self.activations += 1
            self.activated = (handle, start_event, stop_event)
            if failure_stage == "activation":
                raise RuntimeError("injected activation failure")

        def cancel_and_snapshot(self, handle: int) -> CopyWindowSnapshot:
            return CopyWindowSnapshot(handle, (), (), (), (), (), (), None)

        def release_window(self, handle: int) -> None:
            self.released.append(handle)

        def close(self) -> None:
            self.closed = True

    class Cache:
        failed: Exception | None = None
        prefetch_windows = 0

        def fail_closed(self, error: Exception) -> None:
            self.failed = error

        def record_prefetch_window(self) -> None:
            if failure_stage == "record":
                raise RuntimeError("injected statistics failure")
            self.prefetch_windows += 1

    monkeypatch.setattr(
        cached_expert_layer.torch.cuda,
        "device",
        lambda device: nullcontext(),
    )
    monkeypatch.setattr(
        cached_expert_layer.torch.cuda.nvtx,
        "range",
        lambda label: nullcontext(),
    )
    scheduler = Scheduler()
    cache = Cache()
    copy_stream = SimpleNamespace(device=torch.device("cpu"), cuda_stream=123)
    prefetcher = _NativeDummyExpertPrefetcher(
        (0, 1),
        cache,
        scheduler,
        copy_stream,
    )
    prefetcher._bindings[1] = _NativeDummyPrefetchBinding(
        Binding(),
        {},
        MappingProxyType({}),
    )

    try:
        request = prefetcher.prepare_after(0)
        assert request is not None
        assert prepare_started.wait(timeout=5)
        prefetcher.set_stop_event(1, SimpleNamespace(cuda_event=99))
        start_event = SimpleNamespace(cuda_event=77)
        if activate_before_prepare:
            prefetcher.activate_prepared(request, start_event)
        allow_prepare.set()
        assert request.done.wait(timeout=5)
        if not activate_before_prepare:
            assert scheduler.activated is None
            if failure_stage is not None:
                with pytest.raises(ExpertCacheLoadError):
                    prefetcher.activate_prepared(request, start_event)
            else:
                prefetcher.activate_prepared(request, start_event)

        assert scheduler.activated == (11, 77, 99)
        assert scheduler.activations == 1
        if failure_stage is not None:
            assert isinstance(cache.failed, ExpertCacheLoadError)
            assert scheduler.released == [11]
            assert not reservation.active
            assert prefetcher.active_layer_id is None
        else:
            assert cache.failed is None
            assert cache.prefetch_windows == 1
            prefetcher.activate_prepared(request, start_event)
            assert scheduler.activations == 1
            prefetcher.cancel_window(request)
            prefetcher.cancel_window(request)
            assert scheduler.released == [11]
    finally:
        allow_prepare.set()
        prefetcher.close()
    assert scheduler.closed


@pytest.mark.parametrize(
    "failure_stage",
    ("snapshot", "reconcile", "release"),
)
def test_native_prefetch_cancel_failure_poison_cache_and_release_slots(
    failure_stage: str,
) -> None:
    from vllm.model_executor.layers.fused_moe.cached_expert_layer import (
        _NativeDummyExpertPrefetcher,
        _NativeDummyPrefetchWindow,
    )
    from vllm.utils.native_cuda_copy_scheduler import CopyWindowSnapshot

    cache = StreamedExpertCache([_bundle(0)])
    layer = cache.bind(_binding(0, (0,), num_experts=3))
    reservation = layer.reserve_prefetch({0: _FakeEvent(completed=False)})
    assert reservation is not None

    class Scheduler:
        released: list[int] = []

        def cancel_and_snapshot(self, handle: int) -> CopyWindowSnapshot:
            if failure_stage == "snapshot":
                raise RuntimeError("injected snapshot failure")
            issued = (1,) if failure_stage == "reconcile" else ()
            return CopyWindowSnapshot(handle, (), (), issued, (), (), (), None)

        def release_window(self, handle: int) -> None:
            self.released.append(handle)
            if failure_stage == "release":
                raise RuntimeError("injected release failure")

        def close(self) -> None:
            pass

    scheduler = Scheduler()
    prefetcher = _NativeDummyExpertPrefetcher(
        (0,),
        cache,
        scheduler,
        SimpleNamespace(device=torch.device("cpu")),
    )
    window = _NativeDummyPrefetchWindow(0, 17, reservation)
    prefetcher._window = window

    try:
        with pytest.raises(
            ExpertCacheLoadError,
            match=f"native expert prefetch {failure_stage} failed for layer 0",
        ):
            prefetcher.cancel_window(window)

        assert scheduler.released == [17]
        assert window.released
        assert not reservation.active
        assert cache.slot_snapshots()[0].state is ExpertSlotState.ABSENT
        with pytest.raises(ExpertCacheLoadError, match=failure_stage):
            layer.request(0)
    finally:
        prefetcher.close()


@pytest.mark.parametrize("rollback_succeeds", (True, False))
def test_native_prefetch_pause_failure_is_transactional(
    rollback_succeeds: bool,
) -> None:
    from vllm.model_executor.layers.fused_moe.cached_expert_layer import (
        _NativeDummyExpertPrefetcher,
        _StreamedExpertCacheRuntime,
    )

    cache = StreamedExpertCache([_bundle(0)])
    layer = cache.bind(_binding(0, (0,), num_experts=1))

    class Scheduler:
        pause_depth = 0

        def pause_and_drain(self) -> None:
            self.pause_depth += 1
            raise RuntimeError("injected pause failure")

        def resume(self) -> None:
            if not rollback_succeeds:
                raise RuntimeError("injected resume failure")
            assert self.pause_depth == 1
            self.pause_depth -= 1

        def close(self) -> None:
            pass

    class Coordinator:
        pause_depth = 0

        def pause_native_submissions(self) -> None:
            self.pause_depth += 1

        def resume_native_submissions(self) -> None:
            assert self.pause_depth == 1
            self.pause_depth -= 1

    scheduler = Scheduler()
    coordinator = Coordinator()
    prefetcher = _NativeDummyExpertPrefetcher(
        (0,),
        cache,
        scheduler,
        SimpleNamespace(device=torch.device("cpu")),
    )
    runtime = object.__new__(_StreamedExpertCacheRuntime)
    runtime._coordinator = coordinator
    runtime._dummy_prefetcher = prefetcher

    try:
        expected_error = RuntimeError if rollback_succeeds else ExpertCacheLoadError
        with pytest.raises(expected_error, match="pause failure"):
            runtime.pause_dummy_prefetch_and_drain()

        assert coordinator.pause_depth == 0
        assert prefetcher._pause_depth == 0
        if rollback_succeeds:
            assert scheduler.pause_depth == 0
            with pytest.raises(RuntimeError, match="not paused"):
                prefetcher.resume()
            assert layer.request(0).key == (0, 0, "bf16-triton")
        else:
            assert scheduler.pause_depth == 1
            with pytest.raises(ExpertCacheLoadError, match="pause failure"):
                layer.request(0)
            prefetcher.resume()
    finally:
        prefetcher.close()


def test_dummy_prefetch_uses_layer_order_and_bounded_runway() -> None:
    from vllm.model_executor.layers.fused_moe.cached_expert_layer import (
        _DummyExpertPrefetcher,
    )

    coordinator = _ControlledCoordinator()
    cache = StreamedExpertCache(
        [_bundle(0) for _ in range(9)],
        coordinator=coordinator,
    )
    layer_ids = (2, 5, 9)
    bindings = {
        layer_id: cache.bind(
            _binding(layer_id, tuple(range(offset, offset + 3)), num_experts=3)
        )
        for layer_id, offset in zip(layer_ids, (0, 3, 6))
    }
    prefetcher = _DummyExpertPrefetcher(layer_ids, cache)
    for layer_id, binding in bindings.items():
        prefetcher.register(layer_id, binding, 3)

    try:
        prefetcher.start_after(2)
        submissions = coordinator.wait_for_submissions(2)
        assert [submission.label for submission in submissions] == [
            "expert_cache:prefetch:5:0",
            "expert_cache:prefetch:5:1",
        ]
        assert prefetcher.pending_count == 1

        submissions[0].ready.complete()
        submissions = coordinator.wait_for_submissions(3)
        assert [submission.label for submission in submissions] == [
            "expert_cache:prefetch:5:0",
            "expert_cache:prefetch:5:1",
            "expert_cache:prefetch:5:2",
        ]
        for submission in submissions[1:]:
            submission.ready.complete()

        prefetcher.cancel_for_demand(5)
        prefetcher.start_after(5)
        submissions = coordinator.wait_for_submissions(5)
        targets_two = submissions[-2:]
        assert [submission.label for submission in targets_two] == [
            "expert_cache:prefetch:9:0",
            "expert_cache:prefetch:9:1",
        ]
        prefetcher.cancel_for_demand(9)
        for submission in targets_two:
            submission.ready.complete()
        prefetcher.start_after(9)
        assert prefetcher.active_layer_id is None
        assert len(coordinator.submissions) == 5
        assert cache.stats.prefetch_windows == 2
    finally:
        for submission in coordinator.submissions:
            submission.ready.complete()
        prefetcher.close()


def test_dummy_prefetch_cancel_discards_only_unsubmitted_candidates() -> None:
    from vllm.model_executor.layers.fused_moe.cached_expert_layer import (
        _DummyExpertPrefetcher,
    )

    coordinator = _ControlledCoordinator()
    cache = StreamedExpertCache(
        [_bundle(0) for _ in range(4)],
        coordinator=coordinator,
    )
    binding = cache.bind(_binding(1, (0, 1, 2, 3), num_experts=4))
    prefetcher = _DummyExpertPrefetcher((0, 1), cache)
    prefetcher.register(1, binding, 4)

    try:
        prefetcher.start_after(0)
        submissions = coordinator.wait_for_submissions(2)

        prefetcher.cancel_for_demand(1)
        assert prefetcher.active_layer_id is None
        assert prefetcher.pending_count == 0
        _wait_for_prefetch_candidates_discarded(cache, 2)

        for submission in submissions:
            submission.ready.complete()
        prefetcher.close()
        assert [submission.label for submission in coordinator.submissions] == [
            "expert_cache:prefetch:1:0",
            "expert_cache:prefetch:1:1",
        ]
    finally:
        for submission in coordinator.submissions:
            submission.ready.complete()
        prefetcher.close()


def test_dummy_prefetch_capacity_one_refills_after_each_completion() -> None:
    from vllm.model_executor.layers.fused_moe.cached_expert_layer import (
        _DummyExpertPrefetcher,
    )

    coordinator = _ControlledCoordinator()
    cache = StreamedExpertCache([_bundle(0)], coordinator=coordinator)
    binding = cache.bind(_binding(1, (0,), num_experts=3))
    prefetcher = _DummyExpertPrefetcher((0, 1), cache)
    prefetcher.register(1, binding, 3)

    try:
        prefetcher.start_after(0)
        first = coordinator.wait_for_submissions(1)[0]
        assert first.label == "expert_cache:prefetch:1:0"
        assert prefetcher.pending_count == 2

        first.ready.complete()
        second = coordinator.wait_for_submissions(2)[-1]
        assert second.label == "expert_cache:prefetch:1:1"
        second.ready.complete()
        third = coordinator.wait_for_submissions(3)[-1]
        assert third.label == "expert_cache:prefetch:1:2"
        third.ready.complete()
    finally:
        for submission in coordinator.submissions:
            submission.ready.complete()
        prefetcher.close()


def test_dummy_prefetch_pause_drains_copy_and_blocks_new_windows() -> None:
    from vllm.model_executor.layers.fused_moe.cached_expert_layer import (
        _DummyExpertPrefetcher,
    )

    coordinator = _ControlledCoordinator()
    cache = StreamedExpertCache(
        [_bundle(0) for _ in range(6)],
        coordinator=coordinator,
    )
    layer_one = cache.bind(_binding(1, (0, 1, 2), num_experts=3))
    layer_two = cache.bind(_binding(2, (3, 4, 5), num_experts=3))
    prefetcher = _DummyExpertPrefetcher((0, 1, 2), cache)
    prefetcher.register(1, layer_one, 3)
    prefetcher.register(2, layer_two, 3)
    pause_returned = threading.Event()

    def pause() -> None:
        prefetcher.pause_and_drain()
        pause_returned.set()

    pause_thread = threading.Thread(target=pause, daemon=True)
    try:
        prefetcher.start_after(0)
        first, second = coordinator.wait_for_submissions(2)

        pause_thread.start()
        assert first.ready.synchronize_entered.wait(timeout=5)
        assert prefetcher.active_layer_id is None
        assert not pause_returned.is_set()

        prefetcher.start_after(1)
        assert prefetcher.active_layer_id is None
        assert len(coordinator.submissions) == 2

        first.ready.complete()
        assert second.ready.synchronize_entered.wait(timeout=5)
        second.ready.complete()
        assert pause_returned.wait(timeout=5)
        assert len(coordinator.submissions) == 2
        assert cache.stats.prefetch_candidates_discarded == 1

        prefetcher.resume()
        prefetcher.start_after(1)
        targets_two = coordinator.wait_for_submissions(4)[-2:]
        assert [submission.label for submission in targets_two] == [
            "expert_cache:prefetch:2:0",
            "expert_cache:prefetch:2:1",
        ]
        prefetcher.cancel_for_demand(2)
        for submission in targets_two:
            submission.ready.complete()
    finally:
        for submission in coordinator.submissions:
            submission.ready.complete()
        prefetcher.close()
        pause_thread.join(timeout=5)


def test_dummy_prefetch_pause_linearizes_with_first_submission() -> None:
    from vllm.model_executor.layers.fused_moe.cached_expert_layer import (
        _DummyExpertPrefetcher,
    )

    coordinator = _ControlledCoordinator()
    coordinator.block_next_submission()
    cache = StreamedExpertCache(
        [_bundle(0) for _ in range(3)],
        coordinator=coordinator,
    )
    binding = cache.bind(_binding(1, (0, 1, 2), num_experts=3))
    prefetcher = _DummyExpertPrefetcher((0, 1), cache)
    prefetcher.register(1, binding, 3)
    pause_returned = threading.Event()
    start_returned = threading.Event()

    def start() -> None:
        prefetcher.start_after(0)
        start_returned.set()

    def pause() -> None:
        prefetcher.pause_and_drain()
        pause_returned.set()

    start_thread = threading.Thread(target=start, daemon=True)
    pause_thread = threading.Thread(target=pause, daemon=True)
    try:
        start_thread.start()
        assert coordinator.submit_entered.wait(timeout=5)
        pause_thread.start()
        assert not start_returned.is_set()
        assert not pause_returned.is_set()
        assert prefetcher.active_layer_id is None

        coordinator.release_submission()
        assert start_returned.wait(timeout=5)
        submission = coordinator.wait_for_submissions(1)[0]
        assert submission.ready.synchronize_entered.wait(timeout=5)
        assert not pause_returned.is_set()

        submission.ready.complete()
        assert pause_returned.wait(timeout=5)
        assert cache.stats.prefetch_candidates_discarded == 2
        prefetcher.resume()
    finally:
        coordinator.release_submission()
        for submission in coordinator.submissions:
            submission.ready.complete()
        prefetcher.close()
        start_thread.join(timeout=5)
        pause_thread.join(timeout=5)
        assert not start_thread.is_alive()
        assert not pause_thread.is_alive()


def test_prefetch_suspension_resumes_after_capture_failure(monkeypatch) -> None:
    import vllm.model_executor.layers.fused_moe.cached_expert_layer as cache_layer

    calls: list[str] = []

    class _Runtime:
        def pause_dummy_prefetch_and_drain(self) -> None:
            calls.append("pause")

        def resume_dummy_prefetch(self) -> None:
            calls.append("resume")

    runtime = _Runtime()
    monkeypatch.setattr(cache_layer, "_RUNTIMES", {0: runtime})

    with (
        pytest.raises(RuntimeError, match="capture failed"),
        cache_layer.suspend_streamed_expert_cache_prefetch(),
    ):
        calls.append("capture")
        raise RuntimeError("capture failed")

    assert calls == ["pause", "capture", "resume"]


@pytest.mark.parametrize(
    "cancel_queued_window",
    [False, True],
    ids=("resume-next-window", "cancel-next-window"),
)
def test_dummy_prefetch_keeps_submitted_copies_across_windows(
    cancel_queued_window: bool,
) -> None:
    from vllm.model_executor.layers.fused_moe.cached_expert_layer import (
        _DummyExpertPrefetcher,
    )

    coordinator = _ControlledCoordinator()
    cache = StreamedExpertCache(
        [_bundle(0) for _ in range(6)],
        coordinator=coordinator,
    )
    layer_one = cache.bind(_binding(1, (0, 1, 2), num_experts=3))
    layer_two = cache.bind(_binding(2, (3, 4, 5), num_experts=3))
    prefetcher = _DummyExpertPrefetcher((0, 1, 2), cache)
    prefetcher.register(1, layer_one, 3)
    prefetcher.register(2, layer_two, 3)

    try:
        prefetcher.start_after(0)
        old_copies = coordinator.wait_for_submissions(2)
        assert [copy.label for copy in old_copies] == [
            "expert_cache:prefetch:1:0",
            "expert_cache:prefetch:1:1",
        ]

        prefetcher.cancel_for_demand(1)
        _wait_for_prefetch_candidates_discarded(cache, 1)
        discarded_after_layer_one = cache.stats.prefetch_candidates_discarded
        assert discarded_after_layer_one == 1
        prefetcher.start_after(1)

        assert prefetcher.active_layer_id == 2
        assert prefetcher.pending_count == 3
        assert len(coordinator.submissions) == 2

        if cancel_queued_window:
            prefetcher.cancel_for_demand(2)
            assert prefetcher.active_layer_id is None
            _wait_for_prefetch_candidates_discarded(
                cache,
                discarded_after_layer_one + 3,
            )
            for copy in old_copies:
                copy.ready.complete()
            time.sleep(0.02)
            assert len(coordinator.submissions) == 2
        else:
            old_copies[0].ready.complete()
            next_copy = coordinator.wait_for_submissions(3)[-1]
            assert next_copy.label == "expert_cache:prefetch:2:0"
            prefetcher.cancel_for_demand(2)
            old_copies[1].ready.complete()
            next_copy.ready.complete()
    finally:
        for submission in coordinator.submissions:
            submission.ready.complete()
        prefetcher.close()


def test_dummy_prefetch_inflight_load_is_promoted_without_another_copy() -> None:
    from vllm.model_executor.layers.fused_moe.cached_expert_layer import (
        _DummyExpertPrefetcher,
    )

    coordinator = _ControlledCoordinator()
    cache = StreamedExpertCache([_bundle(0)], coordinator=coordinator)
    binding = cache.bind(_binding(1, (0,), num_experts=3))
    prefetcher = _DummyExpertPrefetcher((0, 1), cache)
    prefetcher.register(1, binding, 3)

    try:
        prefetcher.start_after(0)
        coordinator.wait_for_submissions(1)
        prefetcher.cancel_for_demand(1)

        claim = binding.claim(0)
        assert claim.key == (1, 0, "bf16-triton")
        assert len(coordinator.submissions) == 1
        assert cache.stats.prefetch_promotions == 1
        claim.release()
    finally:
        for submission in coordinator.submissions:
            submission.ready.complete()
        prefetcher.close()


def test_dummy_prefetch_demand_cancel_does_not_wait_for_active_issue() -> None:
    from vllm.model_executor.layers.fused_moe.cached_expert_layer import (
        _DummyExpertPrefetcher,
    )

    coordinator = _ControlledCoordinator()
    coordinator.block_next_submission()
    cache = StreamedExpertCache(
        [_bundle(0) for _ in range(3)],
        coordinator=coordinator,
    )
    binding = cache.bind(_binding(1, (0, 1, 2), num_experts=3))
    prefetcher = _DummyExpertPrefetcher((0, 1), cache)
    prefetcher.register(1, binding, 3)
    cancel_returned = threading.Event()
    start_returned = threading.Event()

    def start() -> None:
        prefetcher.start_after(0)
        start_returned.set()

    def cancel() -> None:
        prefetcher.cancel_for_demand(1)
        cancel_returned.set()

    start_thread = threading.Thread(target=start, daemon=True)
    cancel_thread = threading.Thread(target=cancel, daemon=True)
    try:
        start_thread.start()
        assert coordinator.submit_entered.wait(timeout=5)
        cancel_thread.start()
        assert cancel_returned.wait(timeout=5)
        assert not start_returned.is_set()
        assert prefetcher.active_layer_id is None

        coordinator.release_submission()
        assert start_returned.wait(timeout=5)
        submission = coordinator.wait_for_submissions(1)[0]
        assert submission.label == "expert_cache:prefetch:1:0"
        _wait_for_prefetch_candidates_discarded(cache, 2)
        submission.ready.complete()
        prefetcher.close()
        assert len(coordinator.submissions) == 1
    finally:
        coordinator.release_submission()
        for submission in coordinator.submissions:
            submission.ready.complete()
        prefetcher.close()
        start_thread.join(timeout=5)
        cancel_thread.join(timeout=5)
        assert not start_thread.is_alive()
        assert not cancel_thread.is_alive()


def test_reuse_waits_on_last_use_without_host_synchronization() -> None:
    coordinator = _RecordingCoordinator()
    cache = StreamedExpertCache([_bundle(0)], coordinator=coordinator)
    layer = cache.bind(_binding(0, (0,)))
    arena_ptrs = cache.slot_snapshots()[0]
    first = layer.request(0)
    first_ptrs = coordinator.submissions[0].destination_ptrs

    lease = first.acquire("compute-stream")
    lease.release(compute_stream="compute-stream")
    last_use = coordinator.last_use_events[-1]
    second = layer.request(1)

    assert second.slot_index == first.slot_index == arena_ptrs.index
    assert coordinator.submissions[-1].wait_for is last_use
    assert last_use.synchronize_calls == 0
    assert coordinator.submissions[-1].destination_ptrs == first_ptrs
    with pytest.raises(ExpertCacheStaleHandleError):
        first.acquire()


def test_hard_waiter_preempts_prefetch_and_capacity_one_progresses() -> None:
    coordinator = _RecordingCoordinator()
    cache = StreamedExpertCache([_bundle(0)], coordinator=coordinator)
    layer = cache.bind(_binding(0, (0,), num_experts=3))
    lease = layer.acquire(0)

    result: dict[str, ExpertLoadHandle] = {}
    thread_started = threading.Event()

    def request_hard_expert() -> None:
        thread_started.set()
        result["handle"] = layer.request(1)

    thread = threading.Thread(target=request_hard_expert, daemon=True)
    thread.start()
    assert thread_started.wait(timeout=1)

    deadline = time.monotonic() + 1
    while cache.hard_waiter_count == 0 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert cache.hard_waiter_count == 1
    assert layer.prefetch([2]) == ()
    assert cache.stats.prefetch_dropped_for_hard_demand == 1

    last_use = _FakeEvent(completed=False)
    lease.release(last_use_event=last_use)
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert result["handle"].slot_index == 0
    assert coordinator.submissions[-1].wait_for is last_use
    assert last_use.synchronize_calls == 0


def test_hard_miss_waits_on_one_inflight_load_event() -> None:
    coordinator = _RecordingCoordinator(complete_copies=False)
    cache = StreamedExpertCache([_bundle(0)], coordinator=coordinator)
    layer = cache.bind(_binding(0, (0,)))

    prefetched = layer.prefetch([0])[0]
    ready = coordinator.submissions[0].ready
    assert not ready.completed

    hard = layer.request(1)
    assert ready.synchronize_calls == 1
    assert hard.slot_index == prefetched.slot_index == 0
    assert len(coordinator.submissions) == 2


def test_failed_transfer_invalidates_victim_and_allows_retry() -> None:
    coordinator = _RecordingCoordinator()
    cache = StreamedExpertCache([_bundle(0)], coordinator=coordinator)
    layer = cache.bind(_binding(0, (0,)))
    old_handle = layer.request(0)

    coordinator.fail_next_after_copy = True
    with pytest.raises(ExpertCacheLoadError, match="failed to load expert"):
        layer.request(1)

    snapshot = cache.slot_snapshots()[0]
    assert snapshot.state is ExpertSlotState.ABSENT
    assert snapshot.owner is None
    assert layer.slot_for(0) is None
    assert layer.slot_for(1) is None
    assert cache.stats.load_failures == 1
    with pytest.raises(ExpertCacheStaleHandleError):
        old_handle.acquire()

    retry = layer.request(1)
    assert retry.slot_index == 0
    assert cache.slot_snapshots()[0].state is ExpertSlotState.RESIDENT


def test_streamed_empty_batch_does_not_request_a_zero_byte_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Prepare:
        def supports_async(self) -> bool:
            return False

        def prepare(self, hidden_states, topk_weights, topk_ids, *args, **kwargs):
            del args, kwargs
            return hidden_states, None, None, topk_ids, topk_weights

    class _Experts:
        quant_config = object()
        expects_unquantized_inputs = False

        def moe_problem_size(self, a1q, w1, w2, topk_ids):
            del w1, w2
            return 2, a1q.shape[0], 6, a1q.shape[1], topk_ids.shape[1]

        def workspace_dtype(self, dtype):
            return dtype

        def workspace_shapes(self, *args, **kwargs):
            del args, kwargs
            return (0,), (0,), (0,)

    impl = FusedMoEKernelModularImpl.__new__(FusedMoEKernelModularImpl)
    impl.prepare_finalize = _Prepare()
    impl.fused_experts = _Experts()
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.modular_kernel.current_workspace_manager",
        lambda: pytest.fail("empty streamed batches must not request a workspace"),
    )

    batch = impl.prepare_streamed(
        hidden_states=torch.empty((0, 4), dtype=torch.bfloat16),
        w1=torch.empty((2, 6, 4), dtype=torch.bfloat16),
        w2=torch.empty((2, 4, 3), dtype=torch.bfloat16),
        topk_ids=torch.empty((0, 2), dtype=torch.int32),
        topk_weights=torch.empty((0, 2), dtype=torch.float32),
        activation=object(),
        global_num_experts=2,
        apply_router_weight_on_input=False,
        shared_experts=None,
        shared_experts_input=None,
    )

    assert batch.output.shape == (0, 4)
    assert batch.route_output.shape == (0, 2, 4)
    assert batch.workspace13.numel() == 0
    assert batch.workspace2.numel() == 0


def test_streamed_prepare_reuses_caller_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Prepare:
        def supports_async(self) -> bool:
            return False

        def prepare(self, hidden_states, topk_weights, topk_ids, *args, **kwargs):
            del args, kwargs
            return hidden_states, None, None, topk_ids, topk_weights

    class _Experts:
        quant_config = object()
        expects_unquantized_inputs = False

        def moe_problem_size(self, a1q, w1, w2, topk_ids):
            del w1, w2
            return 2, a1q.shape[0], 6, a1q.shape[1], topk_ids.shape[1]

        def workspace_dtype(self, dtype):
            return dtype

        def workspace_shapes(self, m, n, k, topk, *args, **kwargs):
            del n, args, kwargs
            return (m, topk, k), (m, topk, 6), (m, k)

    impl = FusedMoEKernelModularImpl.__new__(FusedMoEKernelModularImpl)
    impl.prepare_finalize = _Prepare()
    impl.fused_experts = _Experts()
    buffers = StreamedMoEBuffers(
        workspace13=torch.empty(64, dtype=torch.bfloat16),
        workspace2=torch.empty(96, dtype=torch.bfloat16),
        route_output=torch.empty((8, 2, 4), dtype=torch.bfloat16),
        output=torch.empty((8, 4), dtype=torch.bfloat16),
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.modular_kernel.current_workspace_manager",
        lambda: pytest.fail("caller buffers must bypass the workspace manager"),
    )

    data_ptrs = tuple(
        tensor.data_ptr()
        for tensor in (
            buffers.workspace13,
            buffers.workspace2,
            buffers.route_output,
            buffers.output,
        )
    )
    for num_tokens in (3, 1):
        batch = impl.prepare_streamed(
            hidden_states=torch.empty((num_tokens, 4), dtype=torch.bfloat16),
            w1=torch.empty((2, 6, 4), dtype=torch.bfloat16),
            w2=torch.empty((2, 4, 3), dtype=torch.bfloat16),
            topk_ids=torch.empty((num_tokens, 2), dtype=torch.int32),
            topk_weights=torch.empty((num_tokens, 2), dtype=torch.float32),
            activation=object(),
            global_num_experts=2,
            apply_router_weight_on_input=False,
            shared_experts=None,
            shared_experts_input=None,
            buffers=buffers,
        )

        assert (
            batch.workspace13.data_ptr(),
            batch.workspace2.data_ptr(),
            batch.route_output.data_ptr(),
            batch.output.data_ptr(),
        ) == data_ptrs


@pytest.mark.parametrize(
    ("fail_hard_demand", "fail_finish_forward"),
    ((False, False), (True, False), (False, True)),
)
def test_scheduler_orders_dummy_prefetch_lifecycle_around_demand(
    monkeypatch: pytest.MonkeyPatch,
    fail_hard_demand: bool,
    fail_finish_forward: bool,
) -> None:
    from vllm.model_executor.layers.fused_moe import cached_expert_layer
    from vllm.model_executor.layers.fused_moe.cached_expert_layer import (
        CachedExpertLayer,
        _NativeDummyPrefetchRequest,
    )

    events: list[str] = []

    class ForwardLock:
        locked = True

        def release(self) -> None:
            assert self.locked
            events.append("forward_lock.release")
            self.locked = False

    class RoutingReady:
        def synchronize(self) -> None:
            events.append("routing.synchronize.begin")
            events.append("routing.synchronize.end")

    class Claim:
        key = (7, 0, "bf16-triton")
        slot_index = 0

        def release(self) -> None:
            pass

    class Binding:
        layer_id = 7

        def claim_resident(self, expert_ids, *, max_pending_claims):
            assert tuple(expert_ids) == (0,)
            assert max_pending_claims == 2
            events.append("claim_resident")
            if fail_hard_demand:
                raise ExpertCacheLoadError("injected hard-demand failure")
            return (), (0,), (Claim(),)

        def claim(self, expert_id: int):
            pytest.fail(f"unexpected duplicate claim for expert {expert_id}")

        def try_claim(self, expert_id: int):
            pytest.fail(f"unexpected lookahead claim for expert {expert_id}")

    class Kernel:
        def finalize_streamed(self, batch: object) -> object:
            events.append("finalize")
            return batch

    forward_lock = ForwardLock()
    prepare_stream = object()
    layer = CachedExpertLayer.__new__(CachedExpertLayer)
    next_request = _NativeDummyPrefetchRequest(8)

    def cancel_dummy_prefetch_for(layer_id: int) -> None:
        assert layer_id == 7
        assert events[-1] == "routing.synchronize.end"
        events.extend(("cancel.begin", "cancel.end"))

    def finish_forward(stream: object) -> None:
        assert stream is prepare_stream
        assert forward_lock.locked
        events.append("finish_forward")
        if fail_finish_forward:
            raise RuntimeError("injected finish-forward failure")

    def prepare_dummy_prefetch_after(
        layer_id: int,
    ) -> _NativeDummyPrefetchRequest:
        assert layer_id == 7
        assert forward_lock.locked
        assert layer._prepare_stream is prepare_stream
        assert events[-1] == "claim_resident"
        events.append("prepare_next_prefetch")
        return next_request

    def activate_dummy_prefetch(
        request: _NativeDummyPrefetchRequest,
        start_event: object | None,
    ) -> None:
        assert request is next_request
        assert start_event is None
        assert events[-1] == "prepare_next_prefetch"
        events.append("activate_next_prefetch")

    def cancel_dummy_prefetch_window(request: object) -> None:
        assert request is next_request
        events.append("cancel_next_prefetch")

    layer.runtime = SimpleNamespace(
        _forward_lock=forward_lock,
        cache=SimpleNamespace(record_wave=lambda: events.append("record_wave")),
        device=torch.device("cpu"),
        cancel_dummy_prefetch_for=cancel_dummy_prefetch_for,
        cancel_dummy_prefetch_window=cancel_dummy_prefetch_window,
        finish_forward=finish_forward,
        prepare_dummy_prefetch_after=prepare_dummy_prefetch_after,
        activate_dummy_prefetch=activate_dummy_prefetch,
        start_dummy_prefetch_after=lambda layer_id: pytest.fail(
            f"native prefetch for layer {layer_id} was prepared twice"
        ),
    )
    layer.binding = Binding()
    layer.num_experts = 1
    layer._kernel = Kernel()
    layer._prepare_stream = prepare_stream
    layer._prefetch_start = SimpleNamespace(
        record=lambda stream: pytest.fail(
            f"early native prefetch recorded an event on {stream}"
        )
    )
    monkeypatch.setattr(
        cached_expert_layer.torch.cuda,
        "current_stream",
        lambda device: prepare_stream,
    )
    monkeypatch.setattr(cached_expert_layer, "_stream_identity", id)

    def acquire_wave(claims) -> tuple[object, ...]:
        assert tuple(claim.key[1] for claim in claims) == (0,)
        events.append("acquire_wave")
        return (SimpleNamespace(slot_index=0),)

    layer._acquire_wave = acquire_wave

    def stage_wave_map(
        expert_ids,
        slot_indices,
    ) -> None:
        assert tuple(expert_ids) == (0,)
        assert tuple(slot_indices) == (0,)

    def invoke_wave(kernel, batch) -> None:
        del kernel, batch
        events.append("execute_wave")

    layer._stage_wave_map = stage_wave_map
    layer._invoke_wave = invoke_wave
    layer._release_wave = lambda leases: (
        events.append("release_wave") if leases else None
    )
    prepared = SimpleNamespace(
        routing_ready=RoutingReady(),
        routing_ids=torch.tensor([[0]], dtype=torch.int32),
        kernel_batch=object(),
    )

    if fail_hard_demand:
        with pytest.raises(ExpertCacheLoadError, match="hard-demand"):
            layer.execute(prepared)
    elif fail_finish_forward:
        with pytest.raises(RuntimeError, match="finish-forward"):
            layer.execute(prepared)
    else:
        assert layer.execute(prepared) is prepared.kernel_batch

    demand_events = [
        "routing.synchronize.begin",
        "routing.synchronize.end",
        "cancel.begin",
        "cancel.end",
        "claim_resident",
    ]
    assert not forward_lock.locked
    if fail_hard_demand:
        assert events == demand_events + [
            "finish_forward",
            "forward_lock.release",
        ]
    else:
        expected = demand_events + [
            "prepare_next_prefetch",
            "activate_next_prefetch",
            "acquire_wave",
            "execute_wave",
            "release_wave",
            "record_wave",
            "finalize",
            "finish_forward",
        ]
        if fail_finish_forward:
            expected.append("cancel_next_prefetch")
        assert events == expected + ["forward_lock.release"]


def test_scheduler_cancels_activated_prefetch_after_finalize_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.layers.fused_moe import cached_expert_layer
    from vllm.model_executor.layers.fused_moe.cached_expert_layer import (
        CachedExpertLayer,
    )

    events: list[str] = []
    from vllm.model_executor.layers.fused_moe.cached_expert_layer import (
        _NativeDummyPrefetchRequest,
    )

    prefetch_request = _NativeDummyPrefetchRequest(8)

    class ForwardLock:
        locked = True

        def release(self) -> None:
            assert self.locked
            events.append("forward_lock.release")
            self.locked = False

    class Claim:
        key = (7, 0, "bf16-triton")
        slot_index = 0

        def release(self) -> None:
            pass

    class Binding:
        layer_id = 7

        def claim_resident(self, expert_ids, *, max_pending_claims):
            assert tuple(expert_ids) == (0,)
            assert max_pending_claims == 2
            return (), (0,), (Claim(),)

    class Kernel:
        def finalize_streamed(self, batch: object) -> object:
            events.append("finalize")
            raise RuntimeError("injected finalize failure")

    forward_lock = ForwardLock()
    prepare_stream = object()
    layer = CachedExpertLayer.__new__(CachedExpertLayer)

    def prepare_dummy_prefetch_after(layer_id: int) -> object:
        assert layer_id == 7
        events.append("prepare_next_prefetch")
        return prefetch_request

    def cancel_dummy_prefetch_window(window: object) -> None:
        assert window is prefetch_request
        events.append("cancel_next_prefetch")

    def activate_dummy_prefetch(
        request: object,
        start_event: object | None,
    ) -> None:
        assert request is prefetch_request
        assert start_event is None
        events.append("activate_next_prefetch")

    layer.runtime = SimpleNamespace(
        _forward_lock=forward_lock,
        cache=SimpleNamespace(record_wave=lambda: events.append("record_wave")),
        device=torch.device("cpu"),
        cancel_dummy_prefetch_for=lambda layer_id: events.append("cancel_demand"),
        cancel_dummy_prefetch_window=cancel_dummy_prefetch_window,
        finish_forward=lambda stream: events.append("finish_forward"),
        prepare_dummy_prefetch_after=prepare_dummy_prefetch_after,
        activate_dummy_prefetch=activate_dummy_prefetch,
        start_dummy_prefetch_after=lambda layer_id: pytest.fail(
            f"native prefetch for layer {layer_id} used fallback start"
        ),
    )
    layer.binding = Binding()
    layer.num_experts = 1
    layer._kernel = Kernel()
    layer._prepare_stream = prepare_stream
    layer._prefetch_start = SimpleNamespace(
        record=lambda stream: pytest.fail(
            f"early native prefetch recorded an event on {stream}"
        )
    )
    monkeypatch.setattr(
        cached_expert_layer.torch.cuda,
        "current_stream",
        lambda device: prepare_stream,
    )
    monkeypatch.setattr(cached_expert_layer, "_stream_identity", id)

    def acquire_wave(claims) -> tuple[object, ...]:
        events.append("acquire_wave")
        return (SimpleNamespace(slot_index=0),)

    layer._acquire_wave = acquire_wave

    def stage_wave_map(
        expert_ids: object,
        slot_indices: object,
    ) -> None:
        assert tuple(expert_ids) == (0,)
        assert tuple(slot_indices) == (0,)

    def invoke_wave(*args: object) -> None:
        events.append("execute_wave")

    layer._stage_wave_map = stage_wave_map
    layer._invoke_wave = invoke_wave
    layer._release_wave = lambda leases: (
        events.append("release_wave") if leases else None
    )
    prepared = SimpleNamespace(
        routing_ready=_FakeEvent(),
        routing_ids=torch.tensor([[0]], dtype=torch.int32),
        kernel_batch=object(),
    )

    with pytest.raises(RuntimeError, match="finalize failure"):
        layer.execute(prepared)

    assert events == [
        "cancel_demand",
        "prepare_next_prefetch",
        "activate_next_prefetch",
        "acquire_wave",
        "execute_wave",
        "release_wave",
        "record_wave",
        "finalize",
        "cancel_next_prefetch",
        "finish_forward",
        "forward_lock.release",
    ]


def test_scheduler_activates_prefetch_after_all_hard_enqueue_before_ready_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.layers.fused_moe import cached_expert_layer
    from vllm.model_executor.layers.fused_moe.cached_expert_layer import (
        CachedExpertLayer,
        _NativeDummyPrefetchRequest,
        _PreparedExpertDemand,
    )

    events: list[str] = []

    class Claim:
        def __init__(self, expert_id: int) -> None:
            self.expert_id = expert_id

        @property
        def key(self) -> tuple[int, int, str]:
            return (7, self.expert_id, "bf16-triton")

        @property
        def slot_index(self) -> int:
            return self.expert_id

        def is_ready(self) -> bool:
            return False

        def release(self) -> None:
            pass

    class Binding:
        layer_id = 7

        def claim_resident(self, expert_ids, *, max_pending_claims):
            assert tuple(expert_ids) == (0, 1, 2)
            assert max_pending_claims == 2
            events.extend(("enqueue:0", "enqueue:1"))
            return (), (0, 1, 2), (Claim(0), Claim(1))

        def try_claim(self, expert_id: int) -> Claim:
            assert expert_id == 2
            events.append("enqueue:2")
            return Claim(2)

        def claim(self, expert_id: int) -> Claim:
            pytest.fail(f"unexpected blocking claim for expert {expert_id}")

    class Kernel:
        def finalize_streamed(self, batch: object) -> object:
            events.append("finalize")
            return batch

    forward_lock = threading.Lock()
    forward_lock.acquire()
    prepare_stream = object()
    request = _NativeDummyPrefetchRequest(8)
    layer = CachedExpertLayer.__new__(CachedExpertLayer)

    def activate_dummy_prefetch(
        window: object,
        start_event: object | None,
    ) -> None:
        assert window is request
        assert start_event is None
        events.append("activate_prefetch")

    layer.runtime = SimpleNamespace(
        _forward_lock=forward_lock,
        cache=SimpleNamespace(record_wave=lambda: None),
        device=torch.device("cpu"),
        cancel_dummy_prefetch_for=lambda layer_id: None,
        cancel_dummy_prefetch_window=lambda window: None,
        finish_forward=lambda stream: None,
        prepare_dummy_prefetch_after=lambda layer_id: pytest.fail(
            f"prepared request for layer {layer_id} was prepared twice"
        ),
        activate_dummy_prefetch=activate_dummy_prefetch,
        start_dummy_prefetch_after=lambda layer_id: pytest.fail(
            f"native prefetch for layer {layer_id} used fallback start"
        ),
    )
    layer.binding = Binding()
    layer.num_experts = 3
    layer._kernel = Kernel()
    layer._prepare_stream = prepare_stream
    layer._prefetch_start = SimpleNamespace(
        record=lambda stream: pytest.fail(
            f"early native prefetch recorded an event on {stream}"
        )
    )
    monkeypatch.setattr(
        cached_expert_layer.torch.cuda,
        "current_stream",
        lambda device: prepare_stream,
    )
    monkeypatch.setattr(cached_expert_layer, "_stream_identity", id)

    def acquire_wave(claims) -> tuple[object, ...]:
        (claim,) = claims
        events.append(f"ready_wait:{claim.key[1]}")
        return (object(),)

    def execute_wave(
        kernel,
        batch,
        expert_ids,
        leases,
    ) -> None:
        del kernel, batch, leases
        events.append(f"execute:{expert_ids[0]}")

    layer._acquire_wave = acquire_wave
    layer._execute_wave = execute_wave
    layer._release_wave = lambda leases: None
    events.extend(("enqueue:0", "enqueue:1"))
    prepared = SimpleNamespace(
        routing_ready=_FakeEvent(),
        routing_ids=torch.tensor([[0, 1, 2]], dtype=torch.int32),
        kernel_batch=object(),
        demand=_PreparedExpertDemand(
            (0, 1, 2),
            (),
            (0, 1, 2),
            (Claim(0), Claim(1)),
        ),
        next_prefetch_request=request,
    )

    assert layer.execute(prepared) is prepared.kernel_batch
    assert events.index("execute:0") < events.index("enqueue:2")
    assert events.index("enqueue:2") < events.index("activate_prefetch")
    assert events.index("activate_prefetch") < events.index("ready_wait:1")
    assert events.index("ready_wait:1") < events.index("execute:1")
    assert events.index("activate_prefetch") < events.index("finalize")
    assert events.count("activate_prefetch") == 1
    assert not forward_lock.locked()


def test_wave_updates_expert_map_before_fused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.layers.fused_moe import cached_expert_layer
    from vllm.model_executor.layers.fused_moe.cached_expert_layer import (
        CachedExpertLayer,
    )

    events: list[str] = []
    wave_expert_map = object()
    layer = CachedExpertLayer.__new__(CachedExpertLayer)
    layer.wave_expert_map = wave_expert_map

    def update_expert_map(
        expert_map: object,
        expert_ids: list[int],
        slot_indices: list[int],
    ) -> None:
        assert expert_map is wave_expert_map
        assert expert_ids == [2, 0]
        assert slot_indices == [1, 3]
        events.append("map_update")

    monkeypatch.setattr(
        cached_expert_layer.ops,
        "moe_update_expert_map",
        update_expert_map,
    )
    layer._invoke_wave = lambda kernel, batch: events.append("fused_wave")

    layer._execute_wave(
        object(),
        object(),
        (2, 0),
        (SimpleNamespace(slot_index=1), SimpleNamespace(slot_index=3)),
    )

    assert events == ["map_update", "fused_wave"]


def test_scheduler_activates_native_prefetch_before_empty_finalize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.layers.fused_moe import cached_expert_layer
    from vllm.model_executor.layers.fused_moe.cached_expert_layer import (
        CachedExpertLayer,
        _NativeDummyPrefetchRequest,
    )

    events: list[str] = []
    stream = object()
    forward_lock = threading.Lock()
    forward_lock.acquire()
    request = _NativeDummyPrefetchRequest(8)

    def activate_dummy_prefetch(
        item: object,
        start_event: object | None,
    ) -> None:
        assert item is request
        assert start_event is None
        events.append("activate_prefetch")

    class Kernel:
        def finalize_streamed(self, batch: object) -> object:
            events.append("finalize")
            return batch

    layer = CachedExpertLayer.__new__(CachedExpertLayer)
    layer.runtime = SimpleNamespace(
        _forward_lock=forward_lock,
        device=torch.device("cpu"),
        finish_forward=lambda selected_stream: events.append("finish_forward"),
        prepare_dummy_prefetch_after=lambda layer_id: (
            events.append("prepare_prefetch") or request
        ),
        activate_dummy_prefetch=activate_dummy_prefetch,
        start_dummy_prefetch_after=lambda layer_id: pytest.fail(
            f"native prefetch for layer {layer_id} used fallback start"
        ),
        cancel_dummy_prefetch_window=lambda item: pytest.fail(
            f"successful empty routing canceled prefetch {item}"
        ),
    )
    layer.binding = SimpleNamespace(layer_id=7)
    layer._kernel = Kernel()
    layer._prepare_stream = stream
    layer._prefetch_start = SimpleNamespace(
        record=lambda selected_stream: pytest.fail(
            f"empty-routing native prefetch recorded an event on {selected_stream}"
        )
    )
    monkeypatch.setattr(
        cached_expert_layer.torch.cuda,
        "current_stream",
        lambda device: stream,
    )
    monkeypatch.setattr(cached_expert_layer, "_stream_identity", id)
    batch = object()
    prepared = SimpleNamespace(
        demand=SimpleNamespace(
            expert_ids=(),
            resident_claims=(),
            pending_ids=(),
            pending_claims=(),
        ),
        kernel_batch=batch,
    )

    assert layer.execute(prepared) is batch
    assert events == [
        "prepare_prefetch",
        "activate_prefetch",
        "finalize",
        "finish_forward",
    ]
    assert not forward_lock.locked()


def test_scheduler_cleans_up_when_early_prefetch_activation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.layers.fused_moe import cached_expert_layer
    from vllm.model_executor.layers.fused_moe.cached_expert_layer import (
        CachedExpertLayer,
        _NativeDummyPrefetchRequest,
        _NativeDummyPrefetchWindow,
    )

    events: list[str] = []
    stream = object()
    forward_lock = threading.Lock()
    forward_lock.acquire()
    prepared_window = _NativeDummyPrefetchWindow(8, 17, SimpleNamespace(active=True))
    request = _NativeDummyPrefetchRequest(8, window=prepared_window)

    class Claim:
        key = (7, 0, "bf16-triton")
        release_calls = 0
        released = False

        @property
        def slot_index(self) -> int:
            return 0

        def release(self) -> None:
            if self.released:
                return
            self.released = True
            self.release_calls += 1
            events.append("claim.release")

    claim = Claim()

    def activate_dummy_prefetch(
        item: object,
        start_event: object | None,
    ) -> None:
        assert item is request
        assert request.window is prepared_window
        assert start_event is None
        events.append("activate_prefetch")
        raise RuntimeError("injected prefetch activation failure")

    cancel_calls = 0

    def cancel_dummy_prefetch_window(item: object) -> None:
        nonlocal cancel_calls
        assert item is request
        assert request.window is prepared_window
        cancel_calls += 1
        request.canceled = True
        prepared_window.released = True
        events.append("cancel_prefetch")

    layer = CachedExpertLayer.__new__(CachedExpertLayer)
    layer.runtime = SimpleNamespace(
        _forward_lock=forward_lock,
        cache=SimpleNamespace(
            record_wave=lambda: pytest.fail("failed wave was recorded")
        ),
        device=torch.device("cpu"),
        finish_forward=lambda selected_stream: events.append("finish_forward"),
        prepare_dummy_prefetch_after=lambda layer_id: request,
        activate_dummy_prefetch=activate_dummy_prefetch,
        start_dummy_prefetch_after=lambda layer_id: pytest.fail(
            f"native prefetch for layer {layer_id} used fallback start"
        ),
        cancel_dummy_prefetch_window=cancel_dummy_prefetch_window,
    )
    layer.binding = SimpleNamespace(layer_id=7)
    layer._kernel = SimpleNamespace(
        finalize_streamed=lambda batch: pytest.fail("failed wave was finalized")
    )
    layer._prepare_stream = stream
    layer._prefetch_start = SimpleNamespace(
        record=lambda selected_stream: pytest.fail(
            f"early native prefetch recorded an event on {selected_stream}"
        )
    )
    monkeypatch.setattr(
        cached_expert_layer.torch.cuda,
        "current_stream",
        lambda device: stream,
    )
    monkeypatch.setattr(cached_expert_layer, "_stream_identity", id)

    layer._acquire_wave = lambda claims: pytest.fail(
        f"waited for hard claims {claims} after prefetch activation failed"
    )
    prepared = SimpleNamespace(
        demand=SimpleNamespace(
            expert_ids=(0,),
            resident_claims=(),
            pending_ids=(0,),
            pending_claims=(claim,),
        ),
        kernel_batch=object(),
    )

    with pytest.raises(RuntimeError, match="injected prefetch activation failure"):
        layer.execute(prepared)

    assert events == [
        "activate_prefetch",
        "claim.release",
        "cancel_prefetch",
        "finish_forward",
    ]
    assert cancel_calls == 1
    assert claim.release_calls == 1
    assert request.canceled
    assert prepared_window.released
    assert not forward_lock.locked()


def test_scheduler_releases_current_leases_when_lookahead_load_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.layers.fused_moe import cached_expert_layer
    from vllm.model_executor.layers.fused_moe.cached_expert_layer import (
        CachedExpertLayer,
    )

    forward_lock = threading.Lock()
    forward_lock.acquire()
    cache = SimpleNamespace(record_wave=lambda: None)
    layer = CachedExpertLayer.__new__(CachedExpertLayer)
    layer.runtime = SimpleNamespace(
        capacity_per_wave=2,
        _forward_lock=forward_lock,
        cache=cache,
        device=torch.device("cpu"),
        finish_forward=lambda stream: None,
        cancel_dummy_prefetch_for=lambda layer_id: None,
        start_dummy_prefetch_after=lambda layer_id: None,
    )

    def try_claim(expert_id: int):
        assert expert_id == 1
        raise ExpertCacheLoadError("injected lookahead failure")

    resident_claim = SimpleNamespace(
        key=(0, 0, "bf16-triton"),
        release=lambda: None,
    )
    layer.binding = SimpleNamespace(
        layer_id=0,
        claim=lambda expert_id: pytest.fail(
            f"unexpected blocking claim for expert {expert_id}"
        ),
        claim_resident=lambda expert_ids, max_pending_claims: (
            ((resident_claim,), (1,), ())
            if max_pending_claims == 2
            else pytest.fail(f"unexpected hard runway: {max_pending_claims}")
        ),
        try_claim=try_claim,
    )
    layer.num_experts = 2
    layer._kernel = object()
    layer._prepare_stream = object()
    monkeypatch.setattr(
        cached_expert_layer.torch.cuda,
        "current_stream",
        lambda device: layer._prepare_stream,
    )
    monkeypatch.setattr(cached_expert_layer, "_stream_identity", id)

    lease = object()
    released: list[object] = []
    layer._acquire_wave = lambda handles: (lease,)
    layer._execute_wave = lambda *args: None
    monkeypatch.setattr(
        CachedExpertLayer,
        "_release_wave",
        staticmethod(lambda leases: released.extend(leases)),
    )
    prepared = SimpleNamespace(
        routing_ready=_FakeEvent(),
        routing_ids=torch.tensor([[0, 1]], dtype=torch.int32),
        kernel_batch=object(),
    )

    with pytest.raises(ExpertCacheLoadError, match="lookahead"):
        layer.execute(prepared)

    assert released == [lease]
    assert not forward_lock.locked()


def test_scheduler_does_not_block_current_dispatch_on_saturated_lookahead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.layers.fused_moe import cached_expert_layer
    from vllm.model_executor.layers.fused_moe.cached_expert_layer import (
        CachedExpertLayer,
    )

    class Kernel:
        def finalize_streamed(self, batch: object) -> object:
            return batch

    coordinator = _RecordingCoordinator(complete_copies=False)
    cache = StreamedExpertCache(
        [_bundle(0), _bundle(0)],
        coordinator=coordinator,
    )
    binding = cache.bind(_binding(0, (0, 1), num_experts=3))
    prefetched_a, prefetched_b = binding.prefetch([0, 1])
    ready_b = prefetched_b.ready_event
    assert isinstance(ready_b, _FakeEvent)
    assert [slot.state for slot in cache.slot_snapshots()] == [
        ExpertSlotState.LOADING,
        ExpertSlotState.LOADING,
    ]

    forward_lock = threading.Lock()
    forward_lock.acquire()
    layer = CachedExpertLayer.__new__(CachedExpertLayer)
    layer.runtime = SimpleNamespace(
        capacity_per_wave=2,
        _forward_lock=forward_lock,
        cache=cache,
        device=torch.device("cpu"),
        finish_forward=lambda stream: None,
        cancel_dummy_prefetch_for=lambda layer_id: None,
        start_dummy_prefetch_after=lambda layer_id: None,
    )
    layer.binding = binding
    layer.num_experts = 3
    layer._kernel = Kernel()
    layer._prepare_stream = object()
    monkeypatch.setattr(
        cached_expert_layer.torch.cuda,
        "current_stream",
        lambda device: layer._prepare_stream,
    )
    monkeypatch.setattr(cached_expert_layer, "_stream_identity", id)

    executions: list[tuple[tuple[int, ...], int]] = []

    def execute_wave(kernel, batch, expert_ids, leases) -> None:
        del kernel, batch, leases
        executions.append((tuple(expert_ids), ready_b.synchronize_calls))

    def release_wave(leases) -> None:
        completion = _FakeEvent(completed=False)
        for lease in leases:
            lease.release(last_use_event=completion)

    layer._execute_wave = execute_wave
    layer._release_wave = release_wave
    prepared = SimpleNamespace(
        routing_ready=_FakeEvent(),
        routing_ids=torch.tensor([[0, 2]], dtype=torch.int32),
        kernel_batch=object(),
    )

    result = layer.execute(prepared)

    assert result is prepared.kernel_batch
    assert executions == [((0,), 0), ((2,), 0)]
    assert prefetched_a.ready_event is coordinator.submissions[0].ready
    assert cache.stats.requests == 4
    assert cache.stats.hits == 1
    assert cache.stats.misses == 3
    assert cache.stats.waves == 2
    assert not forward_lock.locked()


def test_scheduler_capacity_one_progresses_from_resident_through_cold_demands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.layers.fused_moe import cached_expert_layer
    from vllm.model_executor.layers.fused_moe.cached_expert_layer import (
        CachedExpertLayer,
    )

    events: list[tuple[str, int]] = []

    class SchedulingCoordinator(_RecordingCoordinator):
        trace_submissions = False

        def submit_copy(
            self,
            source: ExpertWeightBundle,
            destination: ExpertWeightBundle,
            *,
            wait_for: _FakeEvent | None,
            label: str,
        ) -> _FakeEvent:
            ready = super().submit_copy(
                source,
                destination,
                wait_for=wait_for,
                label=label,
            )
            if self.trace_submissions:
                events.append(("submit", int(label.rsplit(":", 1)[-1])))
            return ready

    class Kernel:
        def __init__(self) -> None:
            self.finalize_calls = 0

        def finalize_streamed(self, batch: object) -> object:
            self.finalize_calls += 1
            return batch

    coordinator = SchedulingCoordinator()
    cache = StreamedExpertCache([_bundle(0)], coordinator=coordinator)
    binding = cache.bind(_binding(0, (0,), num_experts=3))
    binding.request(0)
    coordinator.complete_copies = False
    coordinator.trace_submissions = True

    forward_lock = threading.Lock()
    forward_lock.acquire()
    kernel = Kernel()
    layer = CachedExpertLayer.__new__(CachedExpertLayer)
    layer.runtime = SimpleNamespace(
        capacity_per_wave=1,
        _forward_lock=forward_lock,
        cache=cache,
        device=torch.device("cpu"),
        finish_forward=lambda stream: None,
        cancel_dummy_prefetch_for=lambda layer_id: None,
        start_dummy_prefetch_after=lambda layer_id: None,
    )
    layer.binding = binding
    layer.num_experts = 3
    layer._kernel = kernel
    layer._prepare_stream = object()
    monkeypatch.setattr(
        cached_expert_layer.torch.cuda,
        "current_stream",
        lambda device: layer._prepare_stream,
    )
    monkeypatch.setattr(cached_expert_layer, "_stream_identity", id)

    released_leases = []
    last_use_events: list[_FakeEvent] = []

    def execute_wave(kernel, batch, expert_ids, leases) -> None:
        del kernel, batch, leases
        assert len(expert_ids) == 1
        events.append(("execute", expert_ids[0]))

    def release_wave(leases) -> None:
        last_use = _FakeEvent(completed=False)
        last_use_events.append(last_use)
        for lease in leases:
            lease.release(last_use_event=last_use)
            released_leases.append(lease)

    layer._execute_wave = execute_wave
    layer._release_wave = release_wave
    routing_ready = _FakeEvent()
    prepared = SimpleNamespace(
        routing_ready=routing_ready,
        routing_ids=torch.tensor([[0, 1, 2]], dtype=torch.int32),
        kernel_batch=object(),
    )

    result = layer.execute(prepared)

    assert result is prepared.kernel_batch
    assert events == [
        ("execute", 0),
        ("submit", 1),
        ("execute", 1),
        ("submit", 2),
        ("execute", 2),
    ]
    assert kernel.finalize_calls == 1
    assert routing_ready.synchronize_calls == 1
    assert len(released_leases) == 3
    assert all(lease.released for lease in released_leases)
    assert all(event.synchronize_calls == 0 for event in last_use_events)
    assert [submission.wait_for for submission in coordinator.submissions[1:]] == (
        last_use_events[:2]
    )
    assert cache.hard_waiter_count == 0
    assert cache.stats.requests == 4
    assert cache.stats.hits == 1
    assert cache.stats.misses == 3
    assert cache.stats.loads == 3
    assert cache.stats.waves == 3
    assert cache.stats.evictions == 2
    assert cache.slot_snapshots()[0].owner == (0, 2, "bf16-triton")
    reclaim = binding.try_claim(2)
    assert reclaim is not None
    reclaim.release()
    assert reclaim.released
    assert not forward_lock.locked()


def test_scheduler_bounds_hard_runway_before_first_wave(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.layers.fused_moe import cached_expert_layer
    from vllm.model_executor.layers.fused_moe.cached_expert_layer import (
        CachedExpertLayer,
    )

    events: list[tuple[str, object]] = []

    class Claim:
        def __init__(self, expert_id: int) -> None:
            self.expert_id = expert_id

        @property
        def key(self) -> tuple[int, int, str]:
            return (0, self.expert_id, "bf16-triton")

        def acquire(self, stream: object) -> SimpleNamespace:
            del stream
            events.append(("acquire", self.expert_id))
            return SimpleNamespace(expert_id=self.expert_id)

        def is_ready(self) -> bool:
            return False

        def release(self) -> None:
            events.append(("release_claim", self.expert_id))

    class Kernel:
        def finalize_streamed(self, batch: object) -> object:
            events.append(("finalize", batch))
            return batch

    def claim(expert_id: int) -> Claim:
        events.append(("claim", expert_id))
        return Claim(expert_id)

    forward_lock = threading.Lock()
    forward_lock.acquire()
    layer = CachedExpertLayer.__new__(CachedExpertLayer)
    layer.runtime = SimpleNamespace(
        capacity_per_wave=3,
        _forward_lock=forward_lock,
        cache=SimpleNamespace(record_wave=lambda: events.append(("record_wave", None))),
        device=torch.device("cpu"),
        finish_forward=lambda stream: None,
        cancel_dummy_prefetch_for=lambda layer_id: None,
        start_dummy_prefetch_after=lambda layer_id: events.append(
            ("prefetch", layer_id)
        ),
    )
    layer.binding = SimpleNamespace(
        layer_id=0,
        claim=claim,
        claim_resident=lambda expert_ids, max_pending_claims: (
            (
                (),
                tuple(expert_ids),
                (claim(tuple(expert_ids)[0]),),
            )
            if max_pending_claims == 2
            else pytest.fail(f"unexpected hard runway: {max_pending_claims}")
        ),
        try_claim=claim,
    )
    layer.num_experts = 4
    layer._kernel = Kernel()
    layer._prepare_stream = object()
    monkeypatch.setattr(
        cached_expert_layer.torch.cuda,
        "current_stream",
        lambda device: layer._prepare_stream,
    )
    monkeypatch.setattr(cached_expert_layer, "_stream_identity", id)

    layer._execute_wave = lambda kernel, batch, expert_ids, leases: events.append(
        ("execute", tuple(expert_ids))
    )
    layer._release_wave = lambda leases: events.extend(
        ("release_lease", lease.expert_id) for lease in leases
    )
    prepared = SimpleNamespace(
        routing_ready=_FakeEvent(),
        routing_ids=torch.tensor([[3, 1, 3, 2]], dtype=torch.int32),
        kernel_batch=object(),
    )

    result = layer.execute(prepared)

    assert result is prepared.kernel_batch
    assert [value for event, value in events if event == "claim"] == [3, 1, 2]
    assert [value for event, value in events if event == "execute"] == [
        (3,),
        (1,),
        (2,),
    ]
    assert events.index(("execute", (3,))) < events.index(("claim", 2))
    assert events.index(("execute", (3,))) < events.index(("acquire", 1))
    assert events.index(("acquire", 2)) < events.index(
        ("finalize", prepared.kernel_batch)
    )
    assert events.index(("finalize", prepared.kernel_batch)) < events.index(
        ("prefetch", 0)
    )
    assert sum(event == "record_wave" for event, _ in events) == 3
    assert sum(event == "finalize" for event, _ in events) == 1
    assert not forward_lock.locked()


def test_scheduler_groups_ready_claims_without_waiting_for_older_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.layers.fused_moe import cached_expert_layer
    from vllm.model_executor.layers.fused_moe.cached_expert_layer import (
        CachedExpertLayer,
    )

    class Claim:
        def __init__(self, expert_id: int, *, ready: bool) -> None:
            self.expert_id = expert_id
            self.ready = ready
            self.released = False

        @property
        def key(self) -> tuple[int, int, str]:
            return (0, self.expert_id, "bf16-triton")

        @property
        def slot_index(self) -> int:
            return self.expert_id

        def is_ready(self) -> bool:
            return self.ready

        def release(self) -> None:
            self.released = True

    claims = {
        0: Claim(0, ready=False),
        1: Claim(1, ready=True),
        2: Claim(2, ready=True),
    }
    executed: list[tuple[int, ...]] = []
    acquired: list[tuple[int, ...]] = []
    forward_lock = threading.Lock()
    forward_lock.acquire()
    layer = CachedExpertLayer.__new__(CachedExpertLayer)
    layer.runtime = SimpleNamespace(
        _forward_lock=forward_lock,
        cache=SimpleNamespace(record_wave=lambda: None),
        device=torch.device("cpu"),
        finish_forward=lambda stream: None,
        start_dummy_prefetch_after=lambda layer_id: None,
    )
    layer.binding = SimpleNamespace(
        layer_id=0,
        claim=lambda expert_id: pytest.fail(
            f"unexpected blocking claim for expert {expert_id}"
        ),
        claim_resident=lambda expert_ids, max_pending_claims: (
            (),
            tuple(expert_ids),
            (claims[0], claims[1]),
        ),
        try_claim=lambda expert_id: claims[expert_id],
    )
    layer._kernel = SimpleNamespace(
        finalize_streamed=lambda batch: batch,
    )
    layer._prepare_stream = object()
    monkeypatch.setattr(
        cached_expert_layer.torch.cuda,
        "current_stream",
        lambda device: layer._prepare_stream,
    )
    monkeypatch.setattr(cached_expert_layer, "_stream_identity", id)

    def acquire_wave(wave_claims) -> tuple[SimpleNamespace, ...]:
        expert_ids = tuple(claim.expert_id for claim in wave_claims)
        acquired.append(expert_ids)
        return tuple(
            SimpleNamespace(expert_id=expert_id, slot_index=expert_id)
            for expert_id in expert_ids
        )

    def execute_wave(kernel, batch, expert_ids, leases) -> None:
        del kernel, batch, leases
        executed.append(tuple(expert_ids))
        if tuple(expert_ids) == (1,):
            claims[0].ready = True

    layer._acquire_wave = acquire_wave
    layer._execute_wave = execute_wave
    layer._release_wave = lambda leases: None
    prepared = SimpleNamespace(
        demand=SimpleNamespace(
            expert_ids=(0, 1, 2),
            resident_claims=(),
            pending_ids=(0, 1, 2),
            pending_claims=(claims[0], claims[1]),
        ),
        kernel_batch=object(),
    )

    assert layer.execute(prepared) is prepared.kernel_batch
    assert acquired == executed == [(1,), (0, 2)]
    assert not forward_lock.locked()


def test_scheduler_runs_resident_hits_before_waiting_for_cold_misses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.layers.fused_moe import cached_expert_layer
    from vllm.model_executor.layers.fused_moe.cached_expert_layer import (
        CachedExpertLayer,
    )

    events: list[tuple[str, object]] = []

    class SchedulingCoordinator(_RecordingCoordinator):
        def submit_copy(
            self,
            source: ExpertWeightBundle,
            destination: ExpertWeightBundle,
            *,
            wait_for: _FakeEvent | None,
            label: str,
        ) -> _FakeEvent:
            ready = super().submit_copy(
                source,
                destination,
                wait_for=wait_for,
                label=label,
            )
            if not self.complete_copies:
                events.append(("submit", int(label.rsplit(":", 1)[-1])))
            return ready

        def wait_ready(
            self,
            event: _FakeEvent,
            compute_stream: object | None,
        ) -> None:
            submission = next(
                submission
                for submission in self.submissions
                if submission.ready is event
            )
            events.append(("wait", int(submission.label.rsplit(":", 1)[-1])))
            super().wait_ready(event, compute_stream)

    class Kernel:
        def finalize_streamed(self, batch: object) -> object:
            events.append(("finalize", batch))
            return batch

    coordinator = SchedulingCoordinator()
    cache = StreamedExpertCache(
        [_bundle(0) for _ in range(4)],
        coordinator=coordinator,
    )
    binding = cache.bind(_binding(0, (0, 1, 2, 3)))
    binding.request(1)
    binding.request(2)
    coordinator.complete_copies = False

    forward_lock = threading.Lock()
    forward_lock.acquire()
    layer = CachedExpertLayer.__new__(CachedExpertLayer)
    layer.runtime = SimpleNamespace(
        capacity_per_wave=4,
        _forward_lock=forward_lock,
        cache=cache,
        device=torch.device("cpu"),
        finish_forward=lambda stream: None,
        cancel_dummy_prefetch_for=lambda layer_id: None,
        start_dummy_prefetch_after=lambda layer_id: None,
    )
    layer.binding = binding
    layer.num_experts = 4
    layer._kernel = Kernel()
    layer._prepare_stream = object()
    monkeypatch.setattr(
        cached_expert_layer.torch.cuda,
        "current_stream",
        lambda device: layer._prepare_stream,
    )
    monkeypatch.setattr(cached_expert_layer, "_stream_identity", id)

    def execute_wave(kernel, batch, expert_ids, leases) -> None:
        del kernel, batch, leases
        events.append(("execute", tuple(expert_ids)))

    def release_wave(leases) -> None:
        completion = _FakeEvent(completed=False)
        for lease in leases:
            lease.release(last_use_event=completion)

    layer._execute_wave = execute_wave
    layer._release_wave = release_wave
    prepared = SimpleNamespace(
        routing_ready=_FakeEvent(),
        routing_ids=torch.tensor([[0, 1, 2, 3]], dtype=torch.int32),
        kernel_batch=object(),
    )

    result = layer.execute(prepared)

    assert result is prepared.kernel_batch
    executions = [value for event, value in events if event == "execute"]
    assert executions == [(1, 2), (0,), (3,)]
    first_execution = ("execute", (1, 2))
    assert events.index(("submit", 0)) < events.index(first_execution)
    assert events.index(first_execution) < events.index(("wait", 0))
    assert [expert_id for execution in executions for expert_id in execution] == [
        1,
        2,
        0,
        3,
    ]
    assert cache.stats.waves == 3
    assert sum(event == "finalize" for event, _ in events) == 1
    assert not forward_lock.locked()


def test_shared_expert_output_uses_fixed_staging_storage() -> None:
    from vllm.model_executor.layers.fused_moe.cached_expert_layer import (
        CachedExpertLayer,
    )

    layer = CachedExpertLayer.__new__(CachedExpertLayer)
    layer.runtime = SimpleNamespace(device=torch.device("cpu"))
    layer.max_num_tokens = 4
    layer._shared_output_staging = torch.empty((4, 3), dtype=torch.bfloat16)
    staging_ptr = layer._shared_output_staging.data_ptr()

    first = torch.arange(6, dtype=torch.float32).to(torch.bfloat16).view(2, 3)
    first_staged = layer.stage_shared_output(first)
    assert first_staged.data_ptr() == staging_ptr
    assert torch.equal(first_staged, first)

    second = torch.full((1, 3), 7, dtype=torch.bfloat16)
    second_staged = layer.stage_shared_output(second)
    assert second_staged.data_ptr() == staging_ptr
    assert torch.equal(second_staged, second)


@pytest.mark.parametrize("fail_kernel_prepare", [False, True])
def test_prepare_starts_next_prefetch_before_kernel_prepare(
    monkeypatch: pytest.MonkeyPatch,
    fail_kernel_prepare: bool,
) -> None:
    from vllm.model_executor.layers.fused_moe import cached_expert_layer
    from vllm.model_executor.layers.fused_moe.cached_expert_layer import (
        CachedExpertLayer,
        _NativeDummyPrefetchRequest,
    )

    events: list[str] = []
    device = torch.device("cuda")

    class FakeTensor:
        def __init__(
            self,
            name: str,
            shape: tuple[int, ...],
            dtype: torch.dtype,
            data_ptr: int,
            values: list[int] | None = None,
        ) -> None:
            self.name = name
            self.shape = shape
            self.dtype = dtype
            self.device = device
            self._data_ptr = data_ptr
            self._values = values

        @property
        def ndim(self) -> int:
            return len(self.shape)

        def __getitem__(self, key: object) -> FakeTensor:
            del key
            return self

        def copy_(self, source: FakeTensor, *, non_blocking: bool) -> None:
            assert non_blocking
            events.append(f"{self.name}.copy_from.{source.name}")

        def data_ptr(self) -> int:
            return self._data_ptr

        def view(self, *shape: int) -> FakeTensor:
            assert shape == (-1,)
            return self

        def tolist(self) -> list[int]:
            assert self._values is not None
            return self._values

    class FakeEvent:
        def __init__(self, name: str) -> None:
            self.name = name

        def record(self, stream: object) -> None:
            assert stream is current_stream
            events.append(f"{self.name}.record")

        def synchronize(self) -> None:
            events.append(f"{self.name}.synchronize")

    class Binding:
        layer_id = 7

        def claim_resident(self, expert_ids, *, max_pending_claims):
            assert expert_ids == (2, 1, 0)
            assert max_pending_claims == 2
            events.append("claim_resident")
            return (), expert_ids, ()

    class Kernel:
        def prepare_streamed(self, **kwargs: object) -> object:
            events.append("kernel.prepare_streamed")
            if fail_kernel_prepare:
                raise RuntimeError("injected kernel prepare failure")
            return SimpleNamespace(
                a1q=kwargs["hidden_states"],
                a1q_scale=None,
                topk_weights=kwargs["topk_weights"],
                topk_ids=kwargs["topk_ids"],
            )

    current_stream = object()
    routing_available = FakeEvent("routing_available")
    routing_ready = FakeEvent("routing_ready")
    hidden = FakeTensor("hidden", (2, 4), torch.bfloat16, 1)
    topk_weights = FakeTensor("topk_weights", (2, 2), torch.float32, 2)
    topk_ids = FakeTensor("topk_ids", (2, 2), torch.int32, 3)
    hidden_staging = FakeTensor("hidden_staging", (2, 4), torch.bfloat16, 4)
    weights_staging = FakeTensor("weights_staging", (2, 2), torch.float32, 5)
    ids_staging = FakeTensor("ids_staging", (2, 2), torch.int32, 6)
    routing_staging = FakeTensor(
        "routing_staging",
        (2, 2),
        torch.int32,
        7,
        [2, 1, 2, 0],
    )

    def set_stop_event(layer_id: int, event: FakeEvent) -> None:
        assert layer_id == 7
        assert event is routing_available
        events.append("set_stop_event")

    def cancel_current_prefetch(layer_id: int) -> None:
        assert layer_id == 7
        events.append("cancel_current_prefetch")

    next_prefetch_request = _NativeDummyPrefetchRequest(8)

    def prepare_next_prefetch(layer_id: int) -> _NativeDummyPrefetchRequest:
        assert layer_id == 7
        events.append("prepare_next_prefetch")
        return next_prefetch_request

    def cancel_next_prefetch(request: object) -> None:
        assert request is next_prefetch_request
        events.append("cancel_next_prefetch")

    layer = CachedExpertLayer.__new__(CachedExpertLayer)
    layer.runtime = SimpleNamespace(
        device=device,
        w13=object(),
        w2=object(),
        set_dummy_prefetch_stop_event=set_stop_event,
        cancel_dummy_prefetch_for=cancel_current_prefetch,
        prepare_dummy_prefetch_after=prepare_next_prefetch,
        cancel_dummy_prefetch_window=cancel_next_prefetch,
    )
    layer.binding = Binding()
    layer.num_experts = 4
    layer.top_k = 2
    layer.max_num_tokens = 2
    layer._hidden_staging = hidden_staging
    layer._topk_weights_staging = weights_staging
    layer._topk_ids_staging = ids_staging
    layer._routing_staging = routing_staging
    layer._routing_available = routing_available
    layer._routing_ready = routing_ready
    layer._execution_buffers = object()
    layer.set_kernel = lambda kernel: None
    monkeypatch.setattr(
        cached_expert_layer.torch.cuda,
        "current_stream",
        lambda selected_device: current_stream,
    )

    if fail_kernel_prepare:
        with pytest.raises(RuntimeError, match="injected kernel prepare failure"):
            layer._prepare_locked(
                Kernel(),
                hidden,
                topk_weights,
                topk_ids,
                activation=object(),
                global_num_experts=4,
                apply_router_weight_on_input=False,
                shared_experts=None,
                shared_experts_input=None,
            )
    else:
        prepared = layer._prepare_locked(
            Kernel(),
            hidden,
            topk_weights,
            topk_ids,
            activation=object(),
            global_num_experts=4,
            apply_router_weight_on_input=False,
            shared_experts=None,
            shared_experts_input=None,
        )
        assert prepared.next_prefetch_request is next_prefetch_request

    expected = [
        "routing_available.record",
        "set_stop_event",
        "routing_staging.copy_from.topk_ids",
        "routing_ready.record",
        "routing_ready.synchronize",
        "cancel_current_prefetch",
        "claim_resident",
        "prepare_next_prefetch",
        "hidden_staging.copy_from.hidden",
        "weights_staging.copy_from.topk_weights",
        "ids_staging.copy_from.topk_ids",
        "kernel.prepare_streamed",
    ]
    if fail_kernel_prepare:
        expected.append("cancel_next_prefetch")
    assert events == expected


def test_host_routing_staging_uses_one_pinned_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.layers.fused_moe.cached_expert_layer import (
        _StreamedExpertCacheRuntime,
    )

    runtime = _StreamedExpertCacheRuntime.__new__(_StreamedExpertCacheRuntime)
    runtime._shared_host_staging = None
    runtime._shared_host_num_experts = None
    pinned_tensors: list[torch.Tensor] = []

    def record_pin_memory(tensor: torch.Tensor) -> torch.Tensor:
        pinned_tensors.append(tensor)
        return tensor

    monkeypatch.setattr(torch.Tensor, "pin_memory", record_pin_memory)

    routing = runtime.get_host_staging(
        max_num_tokens=8,
        top_k=2,
        num_experts=4,
    )
    same_routing = runtime.get_host_staging(
        max_num_tokens=8,
        top_k=2,
        num_experts=4,
    )

    assert len(pinned_tensors) == 1
    assert routing.shape == (8, 2)
    assert same_routing.data_ptr() == routing.data_ptr()

    with pytest.raises(ValueError, match="routing staging geometry"):
        runtime.get_host_staging(max_num_tokens=9, top_k=2, num_experts=4)
    with pytest.raises(ValueError, match="routing staging geometry"):
        runtime.get_host_staging(max_num_tokens=8, top_k=2, num_experts=5)


def test_streamed_runtime_allocates_and_copies_generic_quantized_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.layers.fused_moe import cached_expert_layer
    from vllm.model_executor.layers.fused_moe.cached_expert_layer import (
        _StreamedExpertCacheRuntime,
    )
    from vllm.model_executor.layers.fused_moe.expert_cache import (
        SynchronousExpertTransferCoordinator,
    )

    class Config:
        pass

    config = Config()
    config.offload_config = SimpleNamespace(
        expert_cache_per_layer_size=0,
        expert_cache_shared_size=2,
        expert_cache_prefetch_policy="fifo",
    )
    config.model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(
            num_hidden_layers=1,
            n_routed_experts=2,
            mlp_only_layers=None,
            decoder_sparse_step=1,
        )
    )
    monkeypatch.setattr(
        cached_expert_layer,
        "CudaExpertTransferCoordinator",
        lambda device: SynchronousExpertTransferCoordinator(),
    )

    specs = {
        "w13_weight": SimpleNamespace(shape=(2, 4), dtype=torch.int32),
        "w2_weight": SimpleNamespace(shape=(3, 2), dtype=torch.int32),
        "w13_weight_scale": SimpleNamespace(shape=(2, 1), dtype=torch.uint8),
        "w2_weight_scale": SimpleNamespace(shape=(3, 1), dtype=torch.uint8),
    }
    runtime = _StreamedExpertCacheRuntime(
        config,
        torch.empty((2, 1, 1), dtype=torch.uint8),
        torch.empty((2, 1, 1), dtype=torch.uint8),
        torch.device("cpu"),
        object(),
        format_class="mxfp4-marlin-v1",
        runtime_specs=specs,
        activation_dtype=torch.bfloat16,
        hidden_size=4,
    )

    source_tensors = {
        name: torch.full(spec.shape, index + 1, dtype=spec.dtype)
        for index, (name, spec) in enumerate(specs.items())
    }
    binding = runtime.bind(
        0,
        {0: ExpertWeightBundle("mxfp4-marlin-v1", source_tensors)},
    )
    lease = binding.request(0).acquire()
    try:
        assert set(runtime.arena_tensors) == set(specs)
        assert runtime.activation_dtype == torch.bfloat16
        assert runtime.hidden_size == 4
        for name, source in source_tensors.items():
            assert torch.equal(lease.bundle.tensors[name], source)
    finally:
        lease.release()


def test_release_event_failure_poisons_cache_without_leased_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.layers.fused_moe import cached_expert_layer
    from vllm.model_executor.layers.fused_moe.cached_expert_layer import (
        CachedExpertLayer,
    )

    cache = StreamedExpertCache([_bundle(0)])
    binding = cache.bind(_binding(0, (0,), num_experts=2))
    lease = binding.acquire(0)
    layer = CachedExpertLayer.__new__(CachedExpertLayer)
    layer.runtime = SimpleNamespace(cache=cache, _stream_guard_failed=False)
    layer._prepare_stream = object()

    def fail_event_creation() -> None:
        raise RuntimeError("injected event failure")

    monkeypatch.setattr(cached_expert_layer.torch.cuda, "Event", fail_event_creation)

    with pytest.raises(ExpertCacheError, match="wave completion"):
        layer._release_wave((lease,))

    assert layer.runtime._stream_guard_failed
    assert lease.released
    assert cache.slot_snapshots()[0].state is ExpertSlotState.ABSENT
    with pytest.raises(ExpertCacheError, match="wave completion"):
        binding.request(1)


def test_streamed_host_store_preallocates_one_disjoint_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.layers.fused_moe import unquantized_fused_moe_method
    from vllm.model_executor.model_loader.sharded_state_loader import (
        ShardedStateLoader,
    )

    allocations: list[int] = []

    def allocate_storage(num_bytes: int) -> torch.UntypedStorage:
        allocations.append(num_bytes)
        return torch.UntypedStorage(num_bytes, device="cpu")

    class Owner:
        pass

    monkeypatch.setattr(
        unquantized_fused_moe_method,
        "_allocate_pinned_storage",
        allocate_storage,
    )
    store = unquantized_fused_moe_method._StreamedExpertHostStore(
        Owner(),
        (2, 5, 7),
        (3, 10, 4),
        (3, 4, 5),
        torch.bfloat16,
    )

    w13_2, w2_2 = store.claim_layer(
        2,
        (3, 10, 4),
        (3, 4, 5),
        torch.bfloat16,
    )
    w13_5, w2_5 = store.claim_layer(
        5,
        (3, 10, 4),
        (3, 4, 5),
        torch.bfloat16,
    )

    assert allocations == [1536]
    assert store.nbytes == 1536
    storage_ptr = store.storage.data_ptr()
    assert [
        tensor.untyped_storage().data_ptr() for tensor in (w13_2, w2_2, w13_5, w2_5)
    ] == [storage_ptr] * 4
    assert [
        tensor.data_ptr() - storage_ptr for tensor in (w13_2, w2_2, w13_5, w2_5)
    ] == [0, 256, 512, 768]

    module = torch.nn.Module()
    module._streamed_expert_host_store = store
    module.register_parameter("w13_2", torch.nn.Parameter(w13_2))
    module.register_parameter("w2_2", torch.nn.Parameter(w2_2))
    module.register_parameter("w13_5", torch.nn.Parameter(w13_5))
    module.register_parameter("w2_5", torch.nn.Parameter(w2_5))
    assert set(ShardedStateLoader._filter_subtensors(module.state_dict())) == {
        "w13_2",
        "w2_2",
        "w13_5",
        "w2_5",
    }

    with pytest.raises(RuntimeError, match="claimed twice"):
        store.claim_layer(2, (3, 10, 4), (3, 4, 5), torch.bfloat16)
    with pytest.raises(ValueError, match="one weight layout"):
        store.claim_layer(7, (3, 9, 4), (3, 4, 5), torch.bfloat16)
    with pytest.raises(ValueError, match="not a configured MoE layer"):
        store.claim_layer(9, (3, 10, 4), (3, 4, 5), torch.bfloat16)
    with pytest.raises(RuntimeError, match="missing configured layers"):
        store.validate_complete()

    w13_7, w2_7 = store.claim_layer(
        7,
        (3, 10, 4),
        (3, 4, 5),
        torch.bfloat16,
    )
    store.validate_complete()
    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda tensor: True)
    store.validate_layer(2, w13_2, w2_2)
    with pytest.raises(RuntimeError, match="no longer matches"):
        store.validate_layer(7, w13_7.clone(), w2_7)


def test_streamed_host_store_directly_allocates_one_precalculated_pinned_tensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.layers.fused_moe import unquantized_fused_moe_method

    empty_calls: list[dict[str, Any]] = []
    original_empty = torch.empty

    def record_empty(*args, **kwargs) -> torch.Tensor:
        empty_calls.append(dict(kwargs))
        kwargs.pop("pin_memory", None)
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", record_empty)
    monkeypatch.setattr(torch.UntypedStorage, "is_pinned", lambda storage: True)

    storage = unquantized_fused_moe_method._allocate_pinned_storage(4096)

    assert storage.nbytes() == 4096
    assert empty_calls == [
        {
            "dtype": torch.uint8,
            "device": "cpu",
            "pin_memory": True,
        }
    ]


def test_large_streamed_host_store_disables_pinned_power_of_two_rounding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.layers.fused_moe import expert_cache

    applied: list[str] = []
    monkeypatch.setattr(
        expert_cache.torch._C,
        "_accelerator_getAllocatorSettings",
        lambda: "max_split_size_mb:64",
    )
    monkeypatch.setattr(
        expert_cache.torch._C,
        "_accelerator_setAllocatorSettings",
        applied.append,
    )

    expert_cache._configure_large_pinned_allocation(33 * 1024**2)

    assert applied == [
        "max_split_size_mb:64,pinned_max_round_threshold_mb:32,"
        "pinned_max_cached_size_mb:32"
    ]


def test_small_streamed_host_store_keeps_allocator_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.layers.fused_moe import expert_cache

    applied: list[str] = []
    monkeypatch.setattr(
        expert_cache.torch._C,
        "_accelerator_setAllocatorSettings",
        applied.append,
    )

    expert_cache._configure_large_pinned_allocation(32 * 1024**2)

    assert applied == []


def test_prefetch_offloader_skips_streamed_expert_host_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.offloader import prefetch

    module = torch.nn.Module()
    module.register_parameter(
        "resident",
        torch.nn.Parameter(torch.zeros(1)),
    )
    streamed = torch.nn.Parameter(torch.zeros(1))
    streamed._vllm_streamed_expert_host = True
    module.register_parameter("streamed", streamed)

    captured: list[list[str]] = []

    class FakeModuleOffloader:
        def __init__(self, **kwargs: Any) -> None:
            captured.append(kwargs["whitelist_param_names"])

    monkeypatch.setattr(prefetch, "_ModuleOffloader", FakeModuleOffloader)
    offloader = prefetch.PrefetchOffloader.__new__(prefetch.PrefetchOffloader)
    offloader.group_size = 1
    offloader.num_in_group = 1
    offloader.offload_params = set()
    offloader.mode = "cpu"
    offloader.copy_stream = object()
    offloader.module_offloaders = []
    offloader._hook_module_forward = lambda index, layer: None

    assert offloader.wrap_modules(iter((module,))) == [module]
    assert captured == [["resident"]]


def _make_unquantized_method_for_loading_tests():
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
        UnquantizedFusedMoEMethod,
    )

    method = UnquantizedFusedMoEMethod.__new__(UnquantizedFusedMoEMethod)
    torch.nn.Module.__init__(method)
    method.moe = SimpleNamespace(is_act_and_mul=True, has_bias=False)
    method.moe_kernel = None
    return method


def test_streamed_weights_use_host_store_views_and_mark_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.layers.fused_moe import unquantized_fused_moe_method

    method = _make_unquantized_method_for_loading_tests()
    layer = torch.nn.Module()
    storage = torch.UntypedStorage(1024, device="cpu")
    w13 = torch.empty(0, dtype=torch.bfloat16).set_(storage, 0, (3, 10, 4))
    w2 = torch.empty(0, dtype=torch.bfloat16).set_(storage, 128, (3, 4, 5))
    requests: list[tuple[tuple[int, ...], tuple[int, ...], torch.dtype]] = []

    def get_weight_views(
        layer: torch.nn.Module,
        w13_shape: tuple[int, ...],
        w2_shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del layer
        requests.append((w13_shape, w2_shape, dtype))
        return w13, w2

    monkeypatch.setattr(
        unquantized_fused_moe_method,
        "_streamed_expert_cache_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        unquantized_fused_moe_method,
        "_streamed_expert_weight_views",
        get_weight_views,
    )

    method.create_weights(
        layer,
        num_experts=3,
        hidden_size=4,
        intermediate_size_per_partition=5,
        params_dtype=torch.bfloat16,
    )

    assert requests == [
        ((3, 10, 4), (3, 4, 5), torch.bfloat16),
    ]
    assert layer.w13_weight.untyped_storage().data_ptr() == storage.data_ptr()
    assert layer.w2_weight.untyped_storage().data_ptr() == storage.data_ptr()
    assert layer.w13_weight.storage_offset() == 0
    assert layer.w2_weight.storage_offset() == 128
    assert layer.w13_weight.device.type == "cpu"
    assert layer.w2_weight.device.type == "cpu"
    assert layer.w13_weight._vllm_streamed_expert_host
    assert layer.w2_weight._vllm_streamed_expert_host


def test_device_loading_context_keeps_streamed_experts_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = torch.nn.Module()
    module.register_parameter(
        "resident", torch.nn.Parameter(torch.ones(1), requires_grad=False)
    )
    streamed = torch.nn.Parameter(torch.ones(1), requires_grad=False)
    streamed._vllm_streamed_expert_host = True
    module.register_parameter("streamed", streamed)
    resident_ptr = module.resident.data_ptr()
    streamed_ptr = module.streamed.data_ptr()
    moves: list[tuple[int, torch.device]] = []

    def record_to(
        tensor: torch.Tensor, device: torch.device, *args: object, **kwargs: object
    ) -> torch.Tensor:
        del args, kwargs
        moves.append((tensor.data_ptr(), torch.device(device)))
        return tensor

    monkeypatch.setattr(torch.Tensor, "to", record_to)

    with device_loading_context(module, torch.device("cuda")):
        assert module.streamed.device.type == "cpu"

    assert moves == [
        (resident_ptr, torch.device("cuda")),
        (resident_ptr, torch.device("cpu")),
    ]
    assert all(pointer != streamed_ptr for pointer, _ in moves)


def test_streamed_method_rejects_post_load_hot_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.layers.fused_moe import unquantized_fused_moe_method

    method = _make_unquantized_method_for_loading_tests()
    method.moe_kernel = object()
    monkeypatch.setattr(
        unquantized_fused_moe_method,
        "_streamed_expert_cache_enabled",
        lambda: True,
    )

    with pytest.raises(RuntimeError, match="does not support hot weight updates"):
        method.process_weights_after_loading(torch.nn.Module())
