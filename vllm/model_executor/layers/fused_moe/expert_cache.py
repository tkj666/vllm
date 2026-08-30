# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import threading
import weakref
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum, auto
from types import MappingProxyType
from typing import Any, Protocol, TypeAlias, cast, runtime_checkable

import torch

from vllm.utils.cuda_copy_scheduler import CudaCopyScheduler
from vllm.utils.native_cuda_copy_scheduler import (
    CopyJob,
    CopySegment,
    CudaMemcpyKind,
    NativeCudaCopyScheduler,
)

ExpertCacheKey: TypeAlias = tuple[int, int, str]

_PINNED_ALLOCATOR_LIMIT_MB = 32
_PINNED_ALLOCATOR_LIMIT_BYTES = _PINNED_ALLOCATOR_LIMIT_MB * 1024**2
_PINNED_ALLOCATOR_CONFIG_LOCK = threading.Lock()


def _configure_large_pinned_allocation(num_bytes: int) -> None:
    if num_bytes <= _PINNED_ALLOCATOR_LIMIT_BYTES:
        return

    get_settings = getattr(torch._C, "_accelerator_getAllocatorSettings", None)
    set_settings = getattr(torch._C, "_accelerator_setAllocatorSettings", None)
    if get_settings is None or set_settings is None:
        raise RuntimeError(
            "streamed expert host storage requires PyTorch pinned allocator "
            "size-limit settings"
        )

    overrides = (
        f"pinned_max_round_threshold_mb:{_PINNED_ALLOCATOR_LIMIT_MB},"
        f"pinned_max_cached_size_mb:{_PINNED_ALLOCATOR_LIMIT_MB}"
    )
    with _PINNED_ALLOCATOR_CONFIG_LOCK:
        current = get_settings()
        settings = f"{current},{overrides}" if current else overrides
        try:
            set_settings(settings)
        except (RuntimeError, ValueError) as error:
            raise RuntimeError(
                "failed to configure exact-size allocation for the streamed "
                "expert host storage"
            ) from error


def allocate_streamed_expert_host_storage(num_bytes: int) -> torch.UntypedStorage:
    """Allocate one exact-size PyTorch-pinned expert host storage."""
    if num_bytes <= 0:
        raise ValueError("streamed expert host storage must be non-empty")
    _configure_large_pinned_allocation(num_bytes)
    storage = torch.empty(
        num_bytes,
        dtype=torch.uint8,
        device="cpu",
        pin_memory=True,
    ).untyped_storage()
    if storage.nbytes() != num_bytes or not storage.is_pinned():
        raise RuntimeError("failed to allocate the streamed expert host storage")
    return storage


class _StreamedExpertCacheLoadToken:
    pass


_CURRENT_STREAMED_EXPERT_CACHE_LOAD: ContextVar[
    _StreamedExpertCacheLoadToken | None
] = ContextVar("current_streamed_expert_cache_load", default=None)


@contextmanager
def streamed_expert_cache_load_context() -> Iterator[None]:
    token = _CURRENT_STREAMED_EXPERT_CACHE_LOAD.set(_StreamedExpertCacheLoadToken())
    try:
        yield
    finally:
        _CURRENT_STREAMED_EXPERT_CACHE_LOAD.reset(token)


def get_streamed_expert_cache_load_token() -> object:
    token = _CURRENT_STREAMED_EXPERT_CACHE_LOAD.get()
    if token is None:
        raise RuntimeError(
            "streamed expert weights must be created inside a model-load context"
        )
    return token


class ExpertCacheError(RuntimeError):
    """Base error raised by the streamed expert cache."""


class ExpertCacheBindingError(ExpertCacheError):
    """Raised when a layer cannot be bound to the arena."""


class ExpertCacheBusyError(ExpertCacheError):
    """Raised when an expert already has an active compute lease."""


class ExpertCacheLoadError(ExpertCacheError):
    """Raised when an expert bundle cannot be transferred into the arena."""


class ExpertCacheStaleHandleError(ExpertCacheError):
    """Raised when a load handle refers to an evicted arena generation."""


class ExpertSlotState(Enum):
    """Lifecycle state of one fixed-address arena slot."""

    ABSENT = auto()
    LOADING = auto()
    RESIDENT = auto()
    LEASED = auto()
    PREFETCH_RESERVED = auto()


@dataclass(frozen=True)
class ExpertWeightBundle:
    """Tensors transferred and retained as one expert-cache unit.

    ``format_class`` identifies the runtime representation. The tensor mapping
    is immutable, while its tensor storage remains writable so an arena bundle
    can be filled without replacing graph-visible addresses.
    """

    format_class: str
    tensors: Mapping[str, torch.Tensor]

    def __post_init__(self) -> None:
        if not self.format_class:
            raise ValueError("format_class must be non-empty")
        tensors = dict(self.tensors)
        if not tensors:
            raise ValueError("an expert weight bundle must contain tensors")
        for name, tensor in tensors.items():
            if not name:
                raise ValueError("expert weight tensor names must be non-empty")
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"expert weight {name!r} is not a tensor")
        object.__setattr__(self, "tensors", MappingProxyType(tensors))

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.tensors.values())

    @property
    def data_ptrs(self) -> tuple[int, ...]:
        return tuple(t.data_ptr() for t in self.tensors.values())

    @property
    def is_cpu(self) -> bool:
        return all(t.device.type == "cpu" for t in self.tensors.values())

    @property
    def is_pinned(self) -> bool:
        return self.is_cpu and all(t.is_pinned() for t in self.tensors.values())

    def is_compatible_with(self, other: ExpertWeightBundle) -> bool:
        if self.format_class != other.format_class:
            return False
        if self.tensors.keys() != other.tensors.keys():
            return False
        return all(
            self.tensors[name].shape == other.tensors[name].shape
            and self.tensors[name].dtype == other.tensors[name].dtype
            for name in self.tensors
        )

    def copy_from_(
        self,
        source: ExpertWeightBundle,
        *,
        non_blocking: bool,
    ) -> None:
        if not self.is_compatible_with(source):
            raise ValueError(
                "source and destination expert bundles have incompatible layouts"
            )
        for name, destination in self.tensors.items():
            destination.copy_(source.tensors[name], non_blocking=non_blocking)


@dataclass(frozen=True)
class LayerBinding:
    """Host experts and reserved arena slots belonging to one MoE layer."""

    layer_id: int
    format_class: str
    host_bundles: Mapping[int, ExpertWeightBundle]
    reserved_slot_indices: Sequence[int]

    def __post_init__(self) -> None:
        if self.layer_id < 0:
            raise ValueError("layer_id must be non-negative")
        if not self.format_class:
            raise ValueError("format_class must be non-empty")

        bundles = dict(self.host_bundles)
        if not bundles:
            raise ValueError("a layer binding must contain host expert bundles")
        for expert_id, bundle in bundles.items():
            if expert_id < 0:
                raise ValueError("physical expert IDs must be non-negative")
            if bundle.format_class != self.format_class:
                raise ValueError(
                    f"expert {expert_id} has format {bundle.format_class!r}, "
                    f"expected {self.format_class!r}"
                )

        reserved = tuple(self.reserved_slot_indices)
        if len(set(reserved)) != len(reserved):
            raise ValueError("reserved_slot_indices contains duplicates")
        object.__setattr__(self, "host_bundles", MappingProxyType(bundles))
        object.__setattr__(self, "reserved_slot_indices", reserved)


@runtime_checkable
class ExpertCacheEvent(Protocol):
    """Completion event for a transfer or compute-stream use."""

    def query(self) -> bool: ...

    def synchronize(self) -> None: ...


@runtime_checkable
class ExpertTransferCoordinator(Protocol):
    """Coordinates copies and stream dependencies for arena slots."""

    def submit_copy(
        self,
        source: ExpertWeightBundle,
        destination: ExpertWeightBundle,
        *,
        wait_for: ExpertCacheEvent | None,
        label: str,
    ) -> ExpertCacheEvent: ...

    def wait_ready(
        self,
        event: ExpertCacheEvent,
        compute_stream: object | None,
    ) -> None: ...

    def record_last_use(
        self,
        compute_stream: object | None,
    ) -> ExpertCacheEvent: ...


class ImmediateExpertCacheEvent:
    """Already-complete event used by the synchronous coordinator."""

    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        return None


class SynchronousExpertTransferCoordinator:
    """CPU coordinator used for eager execution and policy tests."""

    def submit_copy(
        self,
        source: ExpertWeightBundle,
        destination: ExpertWeightBundle,
        *,
        wait_for: ExpertCacheEvent | None,
        label: str,
    ) -> ExpertCacheEvent:
        del label
        if wait_for is not None:
            wait_for.synchronize()
        destination.copy_from_(source, non_blocking=False)
        return ImmediateExpertCacheEvent()

    def wait_ready(
        self,
        event: ExpertCacheEvent,
        compute_stream: object | None,
    ) -> None:
        del compute_stream
        event.synchronize()

    def record_last_use(
        self,
        compute_stream: object | None,
    ) -> ExpertCacheEvent:
        del compute_stream
        return ImmediateExpertCacheEvent()


class _QueuedCudaExpertCacheEvent:
    """CUDA event whose native copy-stream record is queued asynchronously."""

    def __init__(
        self,
        event: torch.cuda.Event,
        scheduler: NativeCudaCopyScheduler,
        cookie: int,
    ) -> None:
        self._event = event
        self._scheduler = scheduler
        self._cookie = cookie
        self._lock = threading.Lock()
        self._issued = False
        self._issue_error: Exception | None = None

    @property
    def cuda_event(self) -> int:
        return self._event.cuda_event

    def query(self) -> bool:
        with self._lock:
            if not self._issued:
                self._raise_issue_error_locked()
                try:
                    self._issued = self._scheduler.query_urgent_issued(self._cookie)
                except Exception as exc:
                    self._issue_error = exc
                    raise
                if not self._issued:
                    return False
        return self._event.query()

    def synchronize(self) -> None:
        self.wait_until_issued()
        self._event.synchronize()

    def wait_on_stream(self, stream: torch.cuda.Stream) -> None:
        self.wait_until_issued()
        stream.wait_event(self._event)

    def unwrap(self) -> torch.cuda.Event:
        """Return the CUDA event after its native record has been submitted."""
        self.wait_until_issued()
        return self._event

    def wait_until_issued(self) -> None:
        with self._lock:
            if self._issued:
                return
            self._raise_issue_error_locked()
            try:
                self._scheduler.wait_urgent_issued(self._cookie)
            except Exception as exc:
                self._issue_error = exc
                raise
            self._issued = True

    def _raise_issue_error_locked(self) -> None:
        if self._issue_error is not None:
            raise self._issue_error


class CudaExpertTransferCoordinator:
    """CUDA event protocol implementation with a dedicated copy stream."""

    def __init__(
        self,
        device: torch.device | str,
        *,
        copy_scheduler: CudaCopyScheduler | None = None,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("CudaExpertTransferCoordinator requires a CUDA device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.accelerator.current_device_index())
        self.copy_stream = torch.cuda.Stream(device=self.device)
        self._copy_scheduler = copy_scheduler
        self._native_copy_scheduler: NativeCudaCopyScheduler | None = None
        self._native_submission_pause_depth = 0
        self._submission_lock = threading.Lock()
        self._next_native_cookie = 0
        self._native_ready_events: dict[tuple[int, ...], torch.cuda.Event] = {}

    def register_native_destinations(
        self,
        destinations: Sequence[ExpertWeightBundle],
    ) -> None:
        """Materialize reusable completion events for fixed arena slots."""
        with (
            self._submission_lock,
            torch.accelerator.device_index(self.device.index),
            torch.cuda.stream(self.copy_stream),
        ):
            if self._native_copy_scheduler is not None:
                raise RuntimeError(
                    "native destinations must be registered before the scheduler"
                )
            for destination in destinations:
                key = destination.data_ptrs
                if key in self._native_ready_events:
                    raise ValueError("native expert-copy destination is duplicated")
                event = torch.cuda.Event()
                event.record(self.copy_stream)
                self._native_ready_events[key] = event

    def attach_native_scheduler(
        self,
        scheduler: NativeCudaCopyScheduler,
    ) -> None:
        if self._copy_scheduler is not None:
            raise RuntimeError("a Python copy scheduler is already attached")
        if self._native_copy_scheduler is not None:
            raise RuntimeError("a native copy scheduler is already attached")
        self._native_copy_scheduler = scheduler

    def submit_copy(
        self,
        source: ExpertWeightBundle,
        destination: ExpertWeightBundle,
        *,
        wait_for: ExpertCacheEvent | None,
        label: str,
    ) -> ExpertCacheEvent:
        if not source.is_pinned:
            raise ValueError("CUDA expert source bundles must use pinned CPU memory")
        if any(t.device != self.device for t in destination.tensors.values()):
            raise ValueError("all arena tensors must be on the coordinator device")
        if isinstance(wait_for, _QueuedCudaExpertCacheEvent):
            wait_for = cast(ExpertCacheEvent, wait_for.unwrap())

        with self._submission_lock:
            if (
                self._native_copy_scheduler is not None
                and self._native_submission_pause_depth == 0
            ):
                return self._submit_native_copy(
                    source,
                    destination,
                    wait_for=wait_for,
                    label=label,
                )

        def submit() -> ExpertCacheEvent:
            return self._submit_copy(
                source,
                destination,
                wait_for=wait_for,
                label=label,
            )

        if self._copy_scheduler is None:
            return submit()
        return self._copy_scheduler.submit_urgent(submit)

    def pause_native_submissions(self) -> None:
        with self._submission_lock:
            self._native_submission_pause_depth += 1

    def resume_native_submissions(self) -> None:
        with self._submission_lock:
            if self._native_submission_pause_depth == 0:
                raise RuntimeError("native expert-copy submission is not paused")
            self._native_submission_pause_depth -= 1

    def _submit_copy(
        self,
        source: ExpertWeightBundle,
        destination: ExpertWeightBundle,
        *,
        wait_for: ExpertCacheEvent | None,
        label: str,
    ) -> ExpertCacheEvent:
        with (
            self._submission_lock,
            torch.accelerator.device_index(self.device.index),
            torch.cuda.stream(self.copy_stream),
            torch.cuda.nvtx.range(label),
        ):
            if wait_for is not None:
                self.copy_stream.wait_event(cast(Any, wait_for))
            destination.copy_from_(source, non_blocking=True)
            ready = torch.cuda.Event()
            ready.record(self.copy_stream)
        return cast(ExpertCacheEvent, ready)

    def _submit_native_copy(
        self,
        source: ExpertWeightBundle,
        destination: ExpertWeightBundle,
        *,
        wait_for: ExpertCacheEvent | None,
        label: str,
    ) -> ExpertCacheEvent:
        segments = []
        for name, source_tensor in source.tensors.items():
            destination_tensor = destination.tensors[name]
            if (
                not source_tensor.is_contiguous()
                or not destination_tensor.is_contiguous()
            ):
                raise ValueError("native expert copies require contiguous tensors")
            segments.append(
                CopySegment(
                    src=source_tensor.data_ptr(),
                    dst=destination_tensor.data_ptr(),
                    nbytes=source_tensor.nbytes,
                    kind=CudaMemcpyKind.HOST_TO_DEVICE,
                )
            )
        wait_event = 0 if wait_for is None else int(getattr(wait_for, "cuda_event", 0))
        if wait_for is not None and not wait_event:
            raise RuntimeError("native expert-copy dependency event is uninitialized")

        try:
            ready = self._native_ready_events[destination.data_ptrs]
        except KeyError as exc:
            raise RuntimeError(
                "native expert-copy destination was not registered"
            ) from exc
        done_event = ready.cuda_event
        if not done_event:
            raise RuntimeError("native expert-copy completion event is uninitialized")
        cookie = self._next_native_cookie
        self._next_native_cookie += 1
        scheduler = self._native_copy_scheduler
        assert scheduler is not None
        job = CopyJob(
            cookie=cookie,
            segments=tuple(segments),
            wait_event=wait_event,
            done_event=done_event,
            label=label,
        )
        queued_event = _QueuedCudaExpertCacheEvent(ready, scheduler, cookie)
        scheduler.enqueue_urgent(job)
        return queued_event

    def wait_ready(
        self,
        event: ExpertCacheEvent,
        compute_stream: object | None,
    ) -> None:
        stream = (
            torch.cuda.current_stream(self.device)
            if compute_stream is None
            else cast(torch.cuda.Stream, compute_stream)
        )
        if isinstance(event, _QueuedCudaExpertCacheEvent):
            event.wait_on_stream(stream)
        else:
            stream.wait_event(cast(Any, event))

    def record_last_use(
        self,
        compute_stream: object | None,
    ) -> ExpertCacheEvent:
        stream = (
            torch.cuda.current_stream(self.device)
            if compute_stream is None
            else cast(torch.cuda.Stream, compute_stream)
        )
        event = torch.cuda.Event()
        event.record(stream)
        return cast(ExpertCacheEvent, event)


@dataclass(frozen=True)
class ExpertSlotSnapshot:
    """Read-only policy and diagnostics view of an arena slot."""

    index: int
    state: ExpertSlotState
    owner: ExpertCacheKey | None
    is_shared: bool
    reserved_layer_id: int | None
    fifo_age: int | None
    ready_event: ExpertCacheEvent | None
    last_use_event: ExpertCacheEvent | None


class ExpertCachePolicy(Protocol):
    """Extension point for choosing among otherwise evictable slots."""

    def select_slot(
        self,
        candidates: Sequence[ExpertSlotSnapshot],
        *,
        key: ExpertCacheKey,
    ) -> int | None: ...


class FIFOExpertCachePolicy:
    """Evicts the expert loaded earliest; cache hits do not refresh its age."""

    def select_slot(
        self,
        candidates: Sequence[ExpertSlotSnapshot],
        *,
        key: ExpertCacheKey,
    ) -> int | None:
        del key
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda slot: (
                slot.fifo_age if slot.fifo_age is not None else -1,
                slot.index,
            ),
        ).index

    def _select_slot_from_arena(
        self,
        candidates: Iterable[_ArenaSlot],
        *,
        key: ExpertCacheKey,
    ) -> int | None:
        del key
        selected = min(
            candidates,
            key=lambda slot: (
                slot.fifo_age if slot.fifo_age is not None else -1,
                slot.index,
            ),
            default=None,
        )
        return None if selected is None else selected.index


@dataclass(frozen=True)
class ExpertCacheStats:
    requests: int = 0
    hits: int = 0
    misses: int = 0
    loads: int = 0
    prefetch_requests: int = 0
    prefetch_started: int = 0
    prefetch_dropped: int = 0
    prefetch_dropped_for_hard_demand: int = 0
    prefetch_promotions: int = 0
    prefetch_windows: int = 0
    prefetch_candidates_discarded: int = 0
    evictions: int = 0
    transfer_bytes: int = 0
    pinned_bytes: int = 0
    arena_bytes: int = 0
    waves: int = 0
    load_failures: int = 0

    @property
    def h2d_bytes(self) -> int:
        return self.transfer_bytes


@dataclass
class _MutableExpertCacheStats:
    requests: int = 0
    hits: int = 0
    misses: int = 0
    loads: int = 0
    prefetch_requests: int = 0
    prefetch_started: int = 0
    prefetch_dropped: int = 0
    prefetch_dropped_for_hard_demand: int = 0
    prefetch_promotions: int = 0
    prefetch_windows: int = 0
    prefetch_candidates_discarded: int = 0
    evictions: int = 0
    transfer_bytes: int = 0
    pinned_bytes: int = 0
    arena_bytes: int = 0
    waves: int = 0
    load_failures: int = 0

    def snapshot(self) -> ExpertCacheStats:
        return ExpertCacheStats(**vars(self))


@dataclass
class _LoadRecord:
    key: ExpertCacheKey
    slot_index: int
    generation: int
    prefetch: bool
    event: ExpertCacheEvent | None = None
    error: ExpertCacheError | None = None
    evicted: bool = False
    handle_ref: weakref.ReferenceType[ExpertLoadHandle] | None = None
    previous_last_use_event: ExpertCacheEvent | None = None


@dataclass
class _ArenaSlot:
    index: int
    bundle: ExpertWeightBundle
    is_shared: bool = False
    reserved_layer_id: int | None = None
    state: ExpertSlotState = ExpertSlotState.ABSENT
    owner: ExpertCacheKey | None = None
    fifo_age: int | None = None
    ready_event: ExpertCacheEvent | None = None
    last_use_event: ExpertCacheEvent | None = None
    load: _LoadRecord | None = None
    claim: ExpertClaim | None = None
    prefetch_reservation: ExpertPrefetchReservation | None = None


@dataclass(frozen=True)
class _BoundLayer:
    binding: LayerBinding
    compatible_shared_slots: tuple[int, ...]


@dataclass(frozen=True)
class ExpertPrefetchCopy:
    """One precomputed expert bundle copy in a speculative window."""

    cookie: int
    expert_id: int
    slot_index: int
    source: ExpertWeightBundle
    destination: ExpertWeightBundle
    wait_for: ExpertCacheEvent | None
    ready_event: ExpertCacheEvent


@dataclass(frozen=True)
class _ArenaSlotBackup:
    state: ExpertSlotState
    owner: ExpertCacheKey | None
    fifo_age: int | None
    ready_event: ExpertCacheEvent | None
    last_use_event: ExpertCacheEvent | None
    load: _LoadRecord | None


class ExpertPrefetchReservation:
    """Reserved slots and immutable copy descriptors for one native window."""

    def __init__(
        self,
        cache: StreamedExpertCache,
        layer_id: int,
        copies: Sequence[ExpertPrefetchCopy],
        backups: Mapping[int, _ArenaSlotBackup],
    ) -> None:
        self._cache = cache
        self.layer_id = layer_id
        self.copies = tuple(copies)
        self._backups = dict(backups)
        self._active = True

    @property
    def active(self) -> bool:
        return self._active

    def reconcile(self, issued_cookies: Sequence[int]) -> None:
        """Publish committed copies and restore slots that were untouched."""
        self._cache._reconcile_prefetch_reservation(self, issued_cookies)

    def abort(self) -> None:
        """Restore the reservation before any native copy was submitted."""
        self.reconcile(())

    def fail_closed(self, error: ExpertCacheError) -> None:
        self._cache._fail_prefetch_reservation(self, error)


class ExpertLoadHandle:
    """One coalesced load generation for an expert bundle."""

    def __init__(self, cache: StreamedExpertCache, record: _LoadRecord) -> None:
        self._cache = cache
        self._record = record

    @property
    def key(self) -> ExpertCacheKey:
        return self._record.key

    @property
    def slot_index(self) -> int:
        return self._record.slot_index

    @property
    def ready_event(self) -> ExpertCacheEvent | None:
        return self._record.event

    def wait(self) -> None:
        """Wait on this load's event, never on the whole device."""

        self._cache._wait_for_load(self)

    def acquire(self, compute_stream: object | None = None) -> ExpertLease:
        return self._cache.acquire(self, compute_stream=compute_stream)


class ExpertClaim:
    """Exclusive eviction claim that does not wait on the compute stream."""

    def __init__(self, cache: StreamedExpertCache, record: _LoadRecord) -> None:
        self._cache = cache
        self._record = record
        self._released = False

    @property
    def key(self) -> ExpertCacheKey:
        return self._record.key

    @property
    def slot_index(self) -> int:
        return self._record.slot_index

    @property
    def released(self) -> bool:
        return self._released

    def acquire(self, compute_stream: object | None = None) -> ExpertLease:
        """Convert this claim into a compute lease and wait for readiness."""
        return self._cache.acquire_claim(self, compute_stream=compute_stream)

    def is_ready(self) -> bool:
        """Return whether the claimed expert copy has completed."""
        return self._cache.is_claim_ready(self)

    def release(self) -> None:
        """Drop an unconsumed claim while leaving any asynchronous load cached."""
        self._cache._release_claim(self)


class ExpertLease:
    """Exclusive compute lease protecting one resident arena slot."""

    def __init__(
        self,
        cache: StreamedExpertCache,
        record: _LoadRecord,
        bundle: ExpertWeightBundle,
    ) -> None:
        self._cache = cache
        self._record = record
        self._bundle = bundle
        self._released = False

    @property
    def key(self) -> ExpertCacheKey:
        return self._record.key

    @property
    def slot_index(self) -> int:
        return self._record.slot_index

    @property
    def bundle(self) -> ExpertWeightBundle:
        if self._released:
            raise ExpertCacheStaleHandleError("cannot use a released expert lease")
        return self._bundle

    @property
    def released(self) -> bool:
        return self._released

    def release(
        self,
        *,
        compute_stream: object | None = None,
        last_use_event: ExpertCacheEvent | None = None,
    ) -> None:
        self._cache._release_lease(
            self,
            compute_stream=compute_stream,
            last_use_event=last_use_event,
        )

    def __enter__(self) -> ExpertLease:
        return self

    def __exit__(self, *args: object) -> None:
        self.release()


class BoundExpertCacheLayer:
    """Layer-scoped facade returned from :meth:`StreamedExpertCache.bind`."""

    def __init__(self, cache: StreamedExpertCache, layer_id: int) -> None:
        self._cache = cache
        self.layer_id = layer_id

    def request(self, physical_expert_id: int) -> ExpertLoadHandle:
        handle = self._cache.request(self.layer_id, physical_expert_id)
        assert handle is not None
        return handle

    def claim(self, physical_expert_id: int) -> ExpertClaim:
        """Protect an expert from eviction without waiting on its ready event."""
        return self._cache.claim(self.layer_id, physical_expert_id)

    def try_claim(self, physical_expert_id: int) -> ExpertClaim | None:
        """Claim an expert only when doing so cannot block the caller."""
        return self._cache.try_claim(self.layer_id, physical_expert_id)

    def claim_resident(
        self,
        physical_expert_ids: Iterable[int],
        *,
        max_pending_claims: int | None = None,
    ) -> tuple[tuple[ExpertClaim, ...], tuple[int, ...], tuple[ExpertClaim, ...]]:
        """Protect resident experts and fill the available hard-load runway."""
        return self._cache.claim_resident(
            self.layer_id,
            physical_expert_ids,
            max_pending_claims=max_pending_claims,
        )

    def prefetch(
        self,
        physical_expert_ids: Iterable[int],
    ) -> tuple[ExpertLoadHandle, ...]:
        handles: list[ExpertLoadHandle] = []
        seen: set[int] = set()
        for expert_id in physical_expert_ids:
            if expert_id in seen:
                continue
            seen.add(expert_id)
            handle = self._cache.request(
                self.layer_id,
                expert_id,
                prefetch=True,
            )
            if handle is not None:
                handles.append(handle)
        return tuple(handles)

    def acquire(
        self,
        physical_expert_id: int,
        *,
        compute_stream: object | None = None,
    ) -> ExpertLease:
        return self.request(physical_expert_id).acquire(compute_stream)

    def slot_for(self, physical_expert_id: int) -> int | None:
        return self._cache.slot_for(self.layer_id, physical_expert_id)

    @property
    def reserved_slot_indices(self) -> tuple[int, ...]:
        return self._cache.reserved_slot_indices(self.layer_id)

    @property
    def prefetch_slot_indices(self) -> tuple[int, ...]:
        """Return layer-reserved and format-compatible shared slots."""
        return self._cache.prefetch_slot_indices(self.layer_id)

    def reserve_prefetch(
        self,
        ready_events: Mapping[int, ExpertCacheEvent],
    ) -> ExpertPrefetchReservation | None:
        """Freeze eligible slots and precompute an ascending copy window."""
        return self._cache.reserve_prefetch(self.layer_id, ready_events)

    def prefetch_copy_layouts(
        self,
    ) -> tuple[
        tuple[int, int, ExpertWeightBundle, ExpertWeightBundle],
        ...,
    ]:
        """Return immutable source/destination pairs for native job templates."""
        return self._cache.prefetch_copy_layouts(self.layer_id)

    def is_immediately_claimable(self, physical_expert_id: int) -> bool:
        """Return whether an expert can run without a transfer wait."""
        return self._cache.is_immediately_claimable(
            self.layer_id,
            physical_expert_id,
        )


class StreamedExpertCache:
    """Fixed-address, layer-reserved plus globally shared expert arena.

    The caller allocates every arena bundle before binding layers, making all
    tensor addresses stable before CUDA graph capture. A resident slot may be
    selected for replacement before its last use completes: the transfer
    coordinator orders the copy stream behind that slot's last-use event.
    Leased and loading slots are never eviction candidates.
    """

    def __init__(
        self,
        arena: Sequence[ExpertWeightBundle],
        *,
        shared_slot_indices: Sequence[int] = (),
        coordinator: ExpertTransferCoordinator | None = None,
        policy: ExpertCachePolicy | None = None,
    ) -> None:
        if not arena:
            raise ValueError("the expert cache arena must contain at least one slot")
        shared = tuple(shared_slot_indices)
        if len(set(shared)) != len(shared):
            raise ValueError("shared_slot_indices contains duplicates")
        if any(index < 0 or index >= len(arena) for index in shared):
            raise ValueError("shared_slot_indices contains an out-of-range index")

        shared_set = set(shared)
        self._slots = [
            _ArenaSlot(index=i, bundle=bundle, is_shared=i in shared_set)
            for i, bundle in enumerate(arena)
        ]
        self._shared_slot_indices = shared
        self._coordinator = coordinator or SynchronousExpertTransferCoordinator()
        self._policy = policy or FIFOExpertCachePolicy()
        self._layers: dict[int, _BoundLayer] = {}
        self._owners: dict[ExpertCacheKey, int] = {}
        self._clock = 0
        self._generation = 0
        self._hard_waiters = 0
        self._fatal_error: ExpertCacheError | None = None
        self._stats = _MutableExpertCacheStats()
        self._stats.arena_bytes = sum(bundle.nbytes for bundle in arena)
        self._condition = threading.Condition(threading.RLock())

    @property
    def arena_size(self) -> int:
        return len(self._slots)

    @property
    def shared_slot_indices(self) -> tuple[int, ...]:
        return self._shared_slot_indices

    @property
    def hard_waiter_count(self) -> int:
        with self._condition:
            return self._hard_waiters

    @property
    def stats(self) -> ExpertCacheStats:
        with self._condition:
            return self._stats.snapshot()

    def record_wave(self) -> None:
        with self._condition:
            self._stats.waves += 1

    def record_prefetch_window(self) -> None:
        with self._condition:
            self._stats.prefetch_windows += 1

    def record_prefetch_candidates_discarded(self, count: int) -> None:
        if count < 0:
            raise ValueError("discarded prefetch candidate count must be non-negative")
        if not count:
            return
        with self._condition:
            self._stats.prefetch_candidates_discarded += count

    def reserved_slot_indices(self, layer_id: int) -> tuple[int, ...]:
        with self._condition:
            bound = self._layers.get(layer_id)
            if bound is None:
                raise KeyError(f"layer {layer_id} is not bound")
            return tuple(bound.binding.reserved_slot_indices)

    def prefetch_slot_indices(self, layer_id: int) -> tuple[int, ...]:
        with self._condition:
            bound = self._layers.get(layer_id)
            if bound is None:
                raise KeyError(f"layer {layer_id} is not bound")
            return (
                *bound.binding.reserved_slot_indices,
                *bound.compatible_shared_slots,
            )

    def prefetch_copy_layouts(
        self,
        layer_id: int,
    ) -> tuple[
        tuple[int, int, ExpertWeightBundle, ExpertWeightBundle],
        ...,
    ]:
        with self._condition:
            self._raise_if_poisoned_locked()
            bound = self._layers.get(layer_id)
            if bound is None:
                raise KeyError(f"layer {layer_id} is not bound")
            return tuple(
                (
                    expert_id,
                    slot_index,
                    source,
                    self._slots[slot_index].bundle,
                )
                for expert_id, source in bound.binding.host_bundles.items()
                for slot_index in (
                    *bound.binding.reserved_slot_indices,
                    *bound.compatible_shared_slots,
                )
            )

    def reserve_prefetch(
        self,
        layer_id: int,
        ready_events: Mapping[int, ExpertCacheEvent],
    ) -> ExpertPrefetchReservation | None:
        """Reserve requested compatible slots for speculative copies."""
        with self._condition:
            self._raise_if_poisoned_locked()
            bound = self._layers.get(layer_id)
            if bound is None:
                raise KeyError(f"layer {layer_id} is not bound")
            compatible = (
                *bound.binding.reserved_slot_indices,
                *bound.compatible_shared_slots,
            )
            requested = set(ready_events)
            if not requested:
                return None
            if not requested <= set(compatible):
                raise ValueError(
                    "ready_events contains a slot incompatible with the layer"
                )

            self._refresh_bound_ready_locked(bound)
            slots = tuple(
                self._slots[index]
                for index in compatible
                if index in requested
                and self._slots[index].prefetch_reservation is None
                and self._slots[index].claim is None
                and self._slots[index].state
                in (ExpertSlotState.ABSENT, ExpertSlotState.RESIDENT)
            )
            if not slots:
                return None

            resident_keys = set(self._owners)
            candidates = tuple(
                expert_id
                for expert_id in sorted(bound.binding.host_bundles)
                if (layer_id, expert_id, bound.binding.format_class)
                not in resident_keys
            )
            if not candidates:
                return None

            ordered_slots = tuple(
                sorted(
                    slots,
                    key=lambda slot: (
                        (
                            0
                            if slot.state is ExpertSlotState.ABSENT
                            and not slot.is_shared
                            else 1
                            if slot.state is ExpertSlotState.ABSENT
                            else 2
                            if slot.is_shared
                            else 3
                        ),
                        slot.fifo_age if slot.fifo_age is not None else -1,
                        slot.index,
                    ),
                )
            )
            backups = {
                slot.index: _ArenaSlotBackup(
                    state=slot.state,
                    owner=slot.owner,
                    fifo_age=slot.fifo_age,
                    ready_event=slot.ready_event,
                    last_use_event=slot.last_use_event,
                    load=slot.load,
                )
                for slot in slots
            }
            copies: list[ExpertPrefetchCopy] = []
            for cookie, (expert_id, slot) in enumerate(
                zip(candidates, ordered_slots, strict=False)
            ):
                copies.append(
                    ExpertPrefetchCopy(
                        cookie=cookie,
                        expert_id=expert_id,
                        slot_index=slot.index,
                        source=bound.binding.host_bundles[expert_id],
                        destination=slot.bundle,
                        wait_for=backups[slot.index].last_use_event,
                        ready_event=ready_events[slot.index],
                    )
                )
            if not copies:
                return None

            selected_indices = {copy.slot_index for copy in copies}
            backups = {
                slot_index: backup
                for slot_index, backup in backups.items()
                if slot_index in selected_indices
            }

            reservation = ExpertPrefetchReservation(
                self,
                layer_id,
                copies,
                backups,
            )
            for slot in ordered_slots:
                if slot.index not in selected_indices:
                    continue
                if slot.owner is not None:
                    self._owners.pop(slot.owner, None)
                slot.state = ExpertSlotState.PREFETCH_RESERVED
                slot.owner = None
                slot.fifo_age = None
                slot.ready_event = None
                slot.last_use_event = None
                slot.load = None
                slot.prefetch_reservation = reservation
            self._condition.notify_all()
            return reservation

    def bind(self, binding: LayerBinding) -> BoundExpertCacheLayer:
        """Bind a layer's host store to its exclusive reserved slot range."""

        with self._condition:
            self._raise_if_poisoned_locked()
            if binding.layer_id in self._layers:
                raise ExpertCacheBindingError(
                    f"layer {binding.layer_id} is already bound"
                )
            prototype = next(iter(binding.host_bundles.values()))
            for expert_id, bundle in binding.host_bundles.items():
                if not bundle.is_cpu:
                    raise ExpertCacheBindingError(
                        f"host expert {expert_id} for layer {binding.layer_id} "
                        "is not stored on CPU"
                    )
                if not prototype.is_compatible_with(bundle):
                    raise ExpertCacheBindingError(
                        f"host expert {expert_id} has an incompatible bundle layout"
                    )

            for index in binding.reserved_slot_indices:
                self._validate_slot_index(index)
                slot = self._slots[index]
                if slot.is_shared:
                    raise ExpertCacheBindingError(
                        f"slot {index} is shared and cannot be layer-reserved"
                    )
                if slot.reserved_layer_id is not None:
                    raise ExpertCacheBindingError(
                        f"slot {index} is already reserved by layer "
                        f"{slot.reserved_layer_id}"
                    )
                if not slot.bundle.is_compatible_with(prototype):
                    raise ExpertCacheBindingError(
                        f"reserved slot {index} has an incompatible bundle layout"
                    )

            compatible_shared = tuple(
                index
                for index in self._shared_slot_indices
                if self._slots[index].bundle.is_compatible_with(prototype)
            )
            if not binding.reserved_slot_indices and not compatible_shared:
                raise ExpertCacheBindingError(
                    f"layer {binding.layer_id} has no compatible arena slots"
                )

            for index in binding.reserved_slot_indices:
                self._slots[index].reserved_layer_id = binding.layer_id
            self._layers[binding.layer_id] = _BoundLayer(
                binding=binding,
                compatible_shared_slots=compatible_shared,
            )
            self._stats.pinned_bytes += sum(
                bundle.nbytes for bundle in binding.host_bundles.values()
            )
        return BoundExpertCacheLayer(self, binding.layer_id)

    def request(
        self,
        layer_id: int,
        physical_expert_id: int,
        *,
        prefetch: bool = False,
    ) -> ExpertLoadHandle | None:
        result = self._request(
            layer_id,
            physical_expert_id,
            prefetch=prefetch,
            make_claim=False,
        )
        assert result is None or isinstance(result, ExpertLoadHandle)
        return result

    def claim(self, layer_id: int, physical_expert_id: int) -> ExpertClaim:
        """Get or start a load and protect its slot without a stream wait."""
        result = self._request(
            layer_id,
            physical_expert_id,
            prefetch=False,
            make_claim=True,
        )
        assert isinstance(result, ExpertClaim)
        return result

    def try_claim(
        self,
        layer_id: int,
        physical_expert_id: int,
    ) -> ExpertClaim | None:
        """Claim a hit or start a load without waiting for an arena slot."""
        with self._condition:
            self._raise_if_poisoned_locked()
            bound, source, key = self._resolve_request_locked(
                layer_id,
                physical_expert_id,
            )
            self._refresh_bound_ready_locked(bound)
            return self._try_claim_resolved_locked(bound, source, key)

    def claim_resident(
        self,
        layer_id: int,
        physical_expert_ids: Iterable[int],
        *,
        max_pending_claims: int | None = None,
    ) -> tuple[tuple[ExpertClaim, ...], tuple[int, ...], tuple[ExpertClaim, ...]]:
        """Protect resident routes and fill the hard-load runway atomically."""
        if max_pending_claims is not None and max_pending_claims <= 0:
            raise ValueError("max_pending_claims must be positive")
        expert_ids = tuple(dict.fromkeys(physical_expert_ids))
        with self._condition:
            self._raise_if_poisoned_locked()
            resolved = tuple(
                (
                    expert_id,
                    *self._resolve_request_locked(layer_id, expert_id),
                )
                for expert_id in expert_ids
            )
            if resolved:
                self._refresh_bound_ready_locked(resolved[0][1])
            claims: list[ExpertClaim] = []
            pending: list[
                tuple[int, _BoundLayer, ExpertWeightBundle, ExpertCacheKey]
            ] = []
            pending_claims: list[ExpertClaim] = []
            try:
                for expert_id, bound, source, key in resolved:
                    slot_index = self._owners.get(key)
                    if slot_index is None:
                        pending.append((expert_id, bound, source, key))
                        continue
                    slot = self._slots[slot_index]
                    record = slot.load
                    assert record is not None
                    if (
                        slot.state is not ExpertSlotState.RESIDENT
                        or slot.claim is not None
                    ):
                        pending.append((expert_id, bound, source, key))
                        continue
                    self._stats.requests += 1
                    self._stats.hits += 1
                    if record.prefetch:
                        record.prefetch = False
                        self._stats.prefetch_promotions += 1
                    claims.append(self._claim_record_locked(record))
                for _, bound, source, key in pending:
                    if (
                        max_pending_claims is not None
                        and len(pending_claims) == max_pending_claims
                    ):
                        break
                    claim = self._try_claim_resolved_locked(bound, source, key)
                    if claim is not None:
                        pending_claims.append(claim)
            except Exception:
                for claim in pending_claims:
                    self._release_claim_locked(claim)
                for claim in claims:
                    self._release_claim_locked(claim)
                raise
            return (
                tuple(claims),
                tuple(expert_id for expert_id, _, _, _ in pending),
                tuple(pending_claims),
            )

    def _try_claim_resolved_locked(
        self,
        bound: _BoundLayer,
        source: ExpertWeightBundle,
        key: ExpertCacheKey,
    ) -> ExpertClaim | None:
        slot_index = self._owners.get(key)
        if slot_index is not None:
            slot = self._slots[slot_index]
            record = slot.load
            assert record is not None
            if slot.claim is not None or slot.state is ExpertSlotState.LEASED:
                return None
            self._stats.requests += 1
            self._stats.hits += 1
            if record.prefetch:
                record.prefetch = False
                self._stats.prefetch_promotions += 1
            return self._claim_record_locked(record)

        selected = self._select_slot_locked(bound, key)
        if selected is None:
            return None
        self._stats.requests += 1
        self._stats.misses += 1
        handle = self._start_load_locked(
            selected,
            key,
            source,
            prefetch=False,
        )
        return self._claim_record_locked(handle._record)

    def _request(
        self,
        layer_id: int,
        physical_expert_id: int,
        *,
        prefetch: bool,
        make_claim: bool,
    ) -> ExpertLoadHandle | ExpertClaim | None:
        """Get or start one expert load.

        Hard requests wait on one reusable event or on a lease release when all
        compatible slots are busy. Prefetch requests never wait and may return
        ``None``. Claims additionally exclude their slot from eviction without
        introducing a compute-stream dependency.
        """

        with self._condition:
            self._raise_if_poisoned_locked()
            bound, source, key = self._resolve_request_locked(
                layer_id,
                physical_expert_id,
            )
            self._stats.requests += 1
            if prefetch:
                self._stats.prefetch_requests += 1
            else:
                self._hard_waiters += 1

        try:
            counted_miss = False
            while True:
                wait_record: _LoadRecord | None = None
                wait_event: ExpertCacheEvent | None = None
                with self._condition:
                    self._raise_if_poisoned_locked()
                    self._refresh_bound_ready_locked(bound)
                    slot_index = self._owners.get(key)
                    if slot_index is not None:
                        slot = self._slots[slot_index]
                        record = slot.load
                        assert record is not None
                        if make_claim and (
                            slot.claim is not None
                            or slot.state is ExpertSlotState.LEASED
                        ):
                            self._condition.wait()
                            continue
                        self._stats.hits += 1
                        if not prefetch and record.prefetch:
                            record.prefetch = False
                            self._stats.prefetch_promotions += 1
                        if make_claim:
                            return self._claim_record_locked(record)
                        return self._load_handle_locked(record)

                    if not counted_miss:
                        self._stats.misses += 1
                        counted_miss = True

                    if prefetch and self._hard_waiters:
                        self._stats.prefetch_dropped += 1
                        self._stats.prefetch_dropped_for_hard_demand += 1
                        return None

                    selected = self._select_slot_locked(bound, key)
                    if selected is not None:
                        handle = self._start_load_locked(
                            selected,
                            key,
                            source,
                            prefetch=prefetch,
                        )
                        if make_claim:
                            return self._claim_record_locked(handle._record)
                        return handle

                    if prefetch:
                        self._stats.prefetch_dropped += 1
                        return None

                    wait_record, wait_event = self._oldest_loading_event_locked(bound)
                    if wait_event is None:
                        self._condition.wait()
                        continue

                assert wait_record is not None
                self._synchronize_load_event(wait_record, wait_event)
        finally:
            if not prefetch:
                with self._condition:
                    self._hard_waiters -= 1
                    self._condition.notify_all()

    def acquire(
        self,
        handle: ExpertLoadHandle,
        *,
        compute_stream: object | None = None,
    ) -> ExpertLease:
        """Wait on readiness from the compute stream and protect the slot."""

        if handle._cache is not self:
            raise ValueError("expert load handle belongs to another cache")
        record = handle._record
        with self._condition:
            self._raise_if_poisoned_locked()
            self._refresh_slot_ready_locked(self._slots[record.slot_index])
            slot = self._validate_record_locked(record)
            if slot.claim is not None:
                raise ExpertCacheBusyError(
                    f"expert {record.key} has an active eviction claim"
                )
            if slot.state is ExpertSlotState.LEASED:
                raise ExpertCacheBusyError(
                    f"expert {record.key} already has an active lease"
                )
            if slot.state not in (
                ExpertSlotState.LOADING,
                ExpertSlotState.RESIDENT,
            ):
                raise ExpertCacheStaleHandleError(
                    f"expert {record.key} is not loadable from state {slot.state.name}"
                )
            event = slot.ready_event
            assert event is not None
            slot.state = ExpertSlotState.LEASED

        try:
            self._coordinator.wait_ready(event, compute_stream)
        except Exception as exc:
            error = ExpertCacheLoadError(
                f"failed to make expert {record.key} ready on the compute stream"
            )
            with self._condition:
                self._fail_load_locked(record, error)
            raise error from exc

        return ExpertLease(self, record, slot.bundle)

    def acquire_claim(
        self,
        claim: ExpertClaim,
        *,
        compute_stream: object | None = None,
    ) -> ExpertLease:
        """Convert a protected claim into a lease and enqueue its ready wait."""
        if claim._cache is not self:
            raise ValueError("expert claim belongs to another cache")
        if claim._released:
            raise ExpertCacheStaleHandleError("expert claim was already released")

        record = claim._record
        with self._condition:
            self._raise_if_poisoned_locked()
            self._refresh_slot_ready_locked(self._slots[record.slot_index])
            slot = self._validate_record_locked(record)
            if slot.claim is not claim:
                raise ExpertCacheStaleHandleError(
                    f"expert {record.key} no longer owns its eviction claim"
                )
            if slot.state not in (
                ExpertSlotState.LOADING,
                ExpertSlotState.RESIDENT,
            ):
                raise ExpertCacheStaleHandleError(
                    f"expert {record.key} is not loadable from state {slot.state.name}"
                )
            event = slot.ready_event
            assert event is not None
            slot.claim = None
            slot.state = ExpertSlotState.LEASED
            claim._released = True

        try:
            self._coordinator.wait_ready(event, compute_stream)
        except Exception as exc:
            error = ExpertCacheLoadError(
                f"failed to make expert {record.key} ready on the compute stream"
            )
            with self._condition:
                self._fail_load_locked(record, error)
            raise error from exc

        return ExpertLease(self, record, slot.bundle)

    def is_claim_ready(self, claim: ExpertClaim) -> bool:
        """Query readiness without releasing the claim or waiting."""
        if claim._cache is not self:
            raise ValueError("expert claim belongs to another cache")
        if claim._released:
            raise ExpertCacheStaleHandleError("expert claim was already released")

        record = claim._record
        with self._condition:
            self._raise_if_poisoned_locked()
            self._refresh_slot_ready_locked(self._slots[record.slot_index])
            slot = self._validate_record_locked(record)
            if slot.claim is not claim:
                raise ExpertCacheStaleHandleError(
                    f"expert {record.key} no longer owns its eviction claim"
                )
            if slot.state not in (
                ExpertSlotState.LOADING,
                ExpertSlotState.RESIDENT,
            ):
                raise ExpertCacheStaleHandleError(
                    f"expert {record.key} is not loadable from state {slot.state.name}"
                )
            return slot.state is ExpertSlotState.RESIDENT

    def slot_for(self, layer_id: int, physical_expert_id: int) -> int | None:
        with self._condition:
            bound = self._layers.get(layer_id)
            if bound is None:
                raise KeyError(f"layer {layer_id} is not bound")
            key = (layer_id, physical_expert_id, bound.binding.format_class)
            return self._owners.get(key)

    def is_immediately_claimable(
        self,
        layer_id: int,
        physical_expert_id: int,
    ) -> bool:
        with self._condition:
            self._raise_if_poisoned_locked()
            _, _, key = self._resolve_request_locked(
                layer_id,
                physical_expert_id,
            )
            slot_index = self._owners.get(key)
            if slot_index is None:
                return False
            slot = self._slots[slot_index]
            self._refresh_slot_ready_locked(slot)
            return (
                slot.owner == key
                and slot.state is ExpertSlotState.RESIDENT
                and slot.claim is None
            )

    def slot_snapshots(self) -> tuple[ExpertSlotSnapshot, ...]:
        with self._condition:
            self._refresh_ready_locked()
            return tuple(self._snapshot_locked(slot) for slot in self._slots)

    def _reconcile_prefetch_reservation(
        self,
        reservation: ExpertPrefetchReservation,
        issued_cookies: Sequence[int],
    ) -> None:
        if reservation._cache is not self:
            raise ValueError("expert prefetch reservation belongs to another cache")
        issued = tuple(issued_cookies)
        if issued != tuple(range(len(issued))):
            raise ExpertCacheError(
                "native expert prefetch must commit an ascending copy prefix"
            )
        if len(issued) > len(reservation.copies):
            raise ExpertCacheError("native expert prefetch committed unknown copies")

        with self._condition:
            self._raise_if_poisoned_locked()
            if not reservation._active:
                raise ExpertCacheError("expert prefetch reservation is inactive")
            for slot_index in reservation._backups:
                slot = self._slots[slot_index]
                if slot.prefetch_reservation is not reservation:
                    raise ExpertCacheError(
                        f"arena slot {slot_index} lost its prefetch reservation"
                    )

            committed = reservation.copies[: len(issued)]
            final_by_slot: dict[int, tuple[ExpertPrefetchCopy, int, int]] = {}
            copies_per_slot: dict[int, int] = {}
            for copy in committed:
                self._clock += 1
                self._generation += 1
                final_by_slot[copy.slot_index] = (
                    copy,
                    self._clock,
                    self._generation,
                )
                copies_per_slot[copy.slot_index] = (
                    copies_per_slot.get(copy.slot_index, 0) + 1
                )
                self._stats.requests += 1
                self._stats.misses += 1
                self._stats.loads += 1
                self._stats.prefetch_requests += 1
                self._stats.prefetch_started += 1
                self._stats.transfer_bytes += copy.source.nbytes

            for slot_index, backup in reservation._backups.items():
                slot = self._slots[slot_index]
                final = final_by_slot.get(slot_index)
                if final is None:
                    slot.state = backup.state
                    slot.owner = backup.owner
                    slot.fifo_age = backup.fifo_age
                    slot.ready_event = backup.ready_event
                    slot.last_use_event = backup.last_use_event
                    slot.load = backup.load
                    slot.prefetch_reservation = None
                    if backup.owner is not None:
                        self._owners[backup.owner] = slot_index
                    continue

                if backup.load is not None:
                    backup.load.evicted = True
                overwrite_count = copies_per_slot[slot_index]
                if backup.owner is None:
                    overwrite_count -= 1
                self._stats.evictions += overwrite_count

                copy, fifo_age, generation = final
                bound = self._layers[reservation.layer_id]
                key = (
                    reservation.layer_id,
                    copy.expert_id,
                    bound.binding.format_class,
                )
                record = _LoadRecord(
                    key=key,
                    slot_index=slot_index,
                    generation=generation,
                    prefetch=True,
                    event=copy.ready_event,
                    previous_last_use_event=backup.last_use_event,
                )
                slot.state = ExpertSlotState.LOADING
                slot.owner = key
                slot.fifo_age = fifo_age
                slot.ready_event = copy.ready_event
                slot.last_use_event = None
                slot.load = record
                slot.prefetch_reservation = None
                self._owners[key] = slot_index

            self._stats.prefetch_candidates_discarded += len(reservation.copies) - len(
                committed
            )
            reservation._active = False
            self._condition.notify_all()

    def _fail_prefetch_reservation(
        self,
        reservation: ExpertPrefetchReservation,
        error: ExpertCacheError,
    ) -> None:
        if reservation._cache is not self:
            raise ValueError("expert prefetch reservation belongs to another cache")
        with self._condition:
            if not reservation._active:
                return
            reservation._active = False
        self.fail_closed(error)

    def _wait_for_load(self, handle: ExpertLoadHandle) -> None:
        if handle._cache is not self:
            raise ValueError("expert load handle belongs to another cache")
        record = handle._record
        with self._condition:
            self._raise_if_poisoned_locked()
            slot = self._validate_record_locked(record)
            event = slot.ready_event
            assert event is not None
        self._synchronize_load_event(record, event)

    def fail_closed(
        self,
        error: ExpertCacheError,
        *,
        leases: Sequence[ExpertLease] = (),
    ) -> None:
        """Poison the cache when stream ordering can no longer be guaranteed."""
        with self._condition:
            if self._fatal_error is None:
                self._fatal_error = error
            fatal_error = self._fatal_error

            for lease in leases:
                if lease._cache is not self:
                    raise ValueError("expert lease belongs to another cache")
                lease._released = True

            self._owners.clear()
            for slot in self._slots:
                if slot.prefetch_reservation is not None:
                    slot.prefetch_reservation._active = False
                record = slot.load
                if record is not None and record.error is None:
                    record.error = fatal_error
                if slot.claim is not None:
                    slot.claim._released = True
                slot.state = ExpertSlotState.ABSENT
                slot.owner = None
                slot.fifo_age = None
                slot.ready_event = None
                slot.last_use_event = None
                slot.load = None
                slot.claim = None
                slot.prefetch_reservation = None
            self._condition.notify_all()

    def _release_lease(
        self,
        lease: ExpertLease,
        *,
        compute_stream: object | None,
        last_use_event: ExpertCacheEvent | None,
    ) -> None:
        if lease._cache is not self:
            raise ValueError("expert lease belongs to another cache")
        if lease._released:
            raise ExpertCacheStaleHandleError("expert lease was already released")
        if last_use_event is not None and compute_stream is not None:
            raise ValueError(
                "provide either compute_stream or last_use_event, not both"
            )

        completion = last_use_event
        if completion is None:
            completion = self._coordinator.record_last_use(compute_stream)

        with self._condition:
            slot = self._validate_record_locked(lease._record)
            if slot.state is not ExpertSlotState.LEASED:
                raise ExpertCacheStaleHandleError(
                    f"expert {lease.key} no longer owns a leased slot"
                )
            slot.last_use_event = completion
            slot.state = ExpertSlotState.RESIDENT
            lease._released = True
            self._condition.notify_all()

    def _claim_record_locked(self, record: _LoadRecord) -> ExpertClaim:
        slot = self._validate_record_locked(record)
        if slot.claim is not None or slot.state is ExpertSlotState.LEASED:
            raise ExpertCacheBusyError(f"expert {record.key} cannot be claimed")
        claim = ExpertClaim(self, record)
        slot.claim = claim
        return claim

    def _release_claim(self, claim: ExpertClaim) -> None:
        if claim._cache is not self:
            raise ValueError("expert claim belongs to another cache")
        with self._condition:
            self._release_claim_locked(claim)

    def _release_claim_locked(self, claim: ExpertClaim) -> None:
        if claim._released:
            return
        record = claim._record
        slot = self._slots[record.slot_index]
        if slot.load is record and slot.claim is claim:
            slot.claim = None
        elif record.error is None and not record.evicted:
            raise ExpertCacheStaleHandleError(
                f"expert {record.key} no longer owns its eviction claim"
            )
        claim._released = True
        self._condition.notify_all()

    def _resolve_request_locked(
        self,
        layer_id: int,
        physical_expert_id: int,
    ) -> tuple[_BoundLayer, ExpertWeightBundle, ExpertCacheKey]:
        bound = self._layers.get(layer_id)
        if bound is None:
            raise KeyError(f"layer {layer_id} is not bound")
        source = bound.binding.host_bundles.get(physical_expert_id)
        if source is None:
            raise KeyError(
                f"layer {layer_id} has no physical expert {physical_expert_id}"
            )
        key = (layer_id, physical_expert_id, bound.binding.format_class)
        return bound, source, key

    def _select_slot_locked(
        self,
        bound: _BoundLayer,
        key: ExpertCacheKey,
    ) -> int | None:
        reserved = bound.binding.reserved_slot_indices
        shared = bound.compatible_shared_slots

        free_reserved = self._first_absent_slot(reserved)
        if free_reserved is not None:
            return free_reserved
        free_shared = self._first_absent_slot(shared)
        if free_shared is not None:
            return free_shared

        shared_victim = self._policy_victim_locked(shared, key)
        if shared_victim is not None:
            return shared_victim
        return self._policy_victim_locked(reserved, key)

    def _first_absent_slot(self, indices: Sequence[int]) -> int | None:
        return next(
            (
                index
                for index in indices
                if self._slots[index].state is ExpertSlotState.ABSENT
                and self._slots[index].claim is None
            ),
            None,
        )

    def _policy_victim_locked(
        self,
        indices: Sequence[int],
        key: ExpertCacheKey,
    ) -> int | None:
        if type(self._policy) is FIFOExpertCachePolicy:
            policy = cast(FIFOExpertCachePolicy, self._policy)
            return policy._select_slot_from_arena(
                (
                    self._slots[index]
                    for index in indices
                    if self._slots[index].state is ExpertSlotState.RESIDENT
                    and self._slots[index].claim is None
                ),
                key=key,
            )
        candidates = tuple(
            self._snapshot_locked(self._slots[index])
            for index in indices
            if self._slots[index].state is ExpertSlotState.RESIDENT
            and self._slots[index].claim is None
        )
        selected = self._policy.select_slot(candidates, key=key)
        if selected is not None and selected not in {slot.index for slot in candidates}:
            raise ExpertCacheError(
                f"cache policy selected non-candidate arena slot {selected}"
            )
        return selected

    def _start_load_locked(
        self,
        slot_index: int,
        key: ExpertCacheKey,
        source: ExpertWeightBundle,
        *,
        prefetch: bool,
    ) -> ExpertLoadHandle:
        slot = self._slots[slot_index]
        if slot.state not in (ExpertSlotState.ABSENT, ExpertSlotState.RESIDENT):
            raise ExpertCacheBusyError(
                f"arena slot {slot_index} cannot be loaded from {slot.state.name}"
            )
        if slot.claim is not None:
            raise ExpertCacheBusyError(f"arena slot {slot_index} has an active claim")

        wait_for = slot.last_use_event
        if slot.owner is not None:
            old_record = slot.load
            assert old_record is not None
            old_record.evicted = True
            self._owners.pop(slot.owner, None)
            self._stats.evictions += 1

        self._clock += 1
        self._generation += 1
        record = _LoadRecord(
            key=key,
            slot_index=slot_index,
            generation=self._generation,
            prefetch=prefetch,
            previous_last_use_event=wait_for,
        )
        handle = self._load_handle_locked(record)
        slot.state = ExpertSlotState.LOADING
        slot.owner = key
        slot.fifo_age = self._clock
        slot.ready_event = None
        slot.last_use_event = None
        slot.load = record
        self._owners[key] = slot_index

        try:
            event = self._coordinator.submit_copy(
                source,
                slot.bundle,
                wait_for=wait_for,
                label=(
                    f"expert_cache:{'prefetch' if prefetch else 'hard'}:"
                    f"{key[0]}:{key[1]}"
                ),
            )
            if not isinstance(event, ExpertCacheEvent):
                raise TypeError("transfer coordinator returned an invalid event")
            record.event = event
            slot.ready_event = event
            if event.query():
                slot.state = ExpertSlotState.RESIDENT
        except Exception as exc:
            error = ExpertCacheLoadError(
                f"failed to load expert {key} into arena slot {slot_index}"
            )
            self._fail_load_locked(record, error)
            raise error from exc

        self._stats.loads += 1
        self._stats.transfer_bytes += source.nbytes
        if prefetch:
            self._stats.prefetch_started += 1
        self._condition.notify_all()
        return handle

    def _load_handle_locked(self, record: _LoadRecord) -> ExpertLoadHandle:
        handle = None if record.handle_ref is None else record.handle_ref()
        if handle is None:
            handle = ExpertLoadHandle(self, record)
            record.handle_ref = weakref.ref(handle)
        return handle

    def _oldest_loading_event_locked(
        self,
        bound: _BoundLayer,
    ) -> tuple[_LoadRecord | None, ExpertCacheEvent | None]:
        indices = (
            *bound.binding.reserved_slot_indices,
            *bound.compatible_shared_slots,
        )
        loading = [
            self._slots[index]
            for index in indices
            if self._slots[index].state is ExpertSlotState.LOADING
            and self._slots[index].ready_event is not None
        ]
        if not loading:
            return None, None
        slot = min(
            loading,
            key=lambda candidate: (
                candidate.fifo_age if candidate.fifo_age is not None else -1,
                candidate.index,
            ),
        )
        assert slot.load is not None
        return slot.load, slot.ready_event

    def _synchronize_load_event(
        self,
        record: _LoadRecord,
        event: ExpertCacheEvent,
    ) -> None:
        try:
            event.synchronize()
        except Exception as exc:
            error = ExpertCacheLoadError(f"expert load event failed for {record.key}")
            with self._condition:
                self._fail_load_locked(record, error)
            raise error from exc

        with self._condition:
            if record.error is not None:
                raise record.error
            if record.evicted:
                raise ExpertCacheStaleHandleError(
                    f"expert {record.key} was evicted while waiting"
                )
            slot = self._validate_record_locked(record)
            if slot.state is ExpertSlotState.LOADING:
                slot.state = ExpertSlotState.RESIDENT
            self._condition.notify_all()

    def _refresh_ready_locked(self) -> None:
        for slot in self._slots:
            self._refresh_slot_ready_locked(slot)

    def _refresh_bound_ready_locked(self, bound: _BoundLayer) -> None:
        for index in bound.binding.reserved_slot_indices:
            self._refresh_slot_ready_locked(self._slots[index])
        for index in bound.compatible_shared_slots:
            self._refresh_slot_ready_locked(self._slots[index])

    def _refresh_slot_ready_locked(self, slot: _ArenaSlot) -> None:
        if slot.state is not ExpertSlotState.LOADING:
            return
        event = slot.ready_event
        record = slot.load
        assert event is not None and record is not None
        try:
            ready = event.query()
        except Exception as exc:
            error = ExpertCacheLoadError(f"expert load event failed for {record.key}")
            self._fail_load_locked(record, error)
            error.__cause__ = exc
            return
        if ready:
            slot.state = ExpertSlotState.RESIDENT
            self._condition.notify_all()

    def _fail_load_locked(
        self,
        record: _LoadRecord,
        error: ExpertCacheLoadError,
    ) -> None:
        if record.error is None:
            record.error = error
            self._stats.load_failures += 1
        slot = self._slots[record.slot_index]
        if slot.load is not record:
            return
        if slot.owner is not None:
            self._owners.pop(slot.owner, None)
        slot.state = ExpertSlotState.ABSENT
        slot.owner = None
        slot.fifo_age = None
        slot.ready_event = None
        slot.last_use_event = record.previous_last_use_event
        slot.load = None
        if slot.claim is not None:
            slot.claim._released = True
        slot.claim = None
        self._condition.notify_all()

    def _validate_record_locked(self, record: _LoadRecord) -> _ArenaSlot:
        if record.error is not None:
            raise record.error
        if record.evicted:
            raise ExpertCacheStaleHandleError(
                f"expert {record.key} was evicted from arena slot {record.slot_index}"
            )
        slot = self._slots[record.slot_index]
        if slot.load is not record or slot.owner != record.key:
            raise ExpertCacheStaleHandleError(
                f"expert {record.key} no longer owns arena slot {record.slot_index}"
            )
        return slot

    def _raise_if_poisoned_locked(self) -> None:
        if self._fatal_error is not None:
            raise self._fatal_error

    def _snapshot_locked(self, slot: _ArenaSlot) -> ExpertSlotSnapshot:
        return ExpertSlotSnapshot(
            index=slot.index,
            state=slot.state,
            owner=slot.owner,
            is_shared=slot.is_shared,
            reserved_layer_id=slot.reserved_layer_id,
            fifo_age=slot.fifo_age,
            ready_event=slot.ready_event,
            last_use_event=slot.last_use_event,
        )

    def _validate_slot_index(self, index: int) -> None:
        if index < 0 or index >= len(self._slots):
            raise ExpertCacheBindingError(f"arena slot index {index} is out of range")
