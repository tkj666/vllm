# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import threading
import weakref
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from queue import SimpleQueue
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import torch

from vllm import _custom_ops as ops
from vllm.compilation.cuda_graph import CUDAGraphOptions, CUDAGraphWrapper
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.expert_cache import (
    BoundExpertCacheLayer,
    CudaExpertTransferCoordinator,
    ExpertCacheError,
    ExpertCacheLoadError,
    ExpertCacheStats,
    ExpertClaim,
    ExpertLease,
    ExpertLoadHandle,
    ExpertPrefetchCopy,
    ExpertPrefetchReservation,
    ExpertWeightBundle,
    FIFOExpertCachePolicy,
    LayerBinding,
    StreamedExpertCache,
    get_streamed_expert_cache_load_token,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEKernel,
    PreparedStreamedMoEBatch,
    StreamedMoEBuffers,
)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.model_executor.utils import replace_parameter
from vllm.utils.cuda_copy_scheduler import (
    CopyIssueStatus,
    CopyWindow,
    CudaCopyScheduler,
)
from vllm.utils.native_cuda_copy_scheduler import (
    CopyJob,
    CopySegment,
    CopyWindowSnapshot,
    CudaMemcpyKind,
    NativeCudaCopyScheduler,
)
from vllm.utils.torch_utils import current_stream as get_vllm_current_stream

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
        SharedExperts,
    )


_FORMAT_CLASS = "triton-bf16-w13-w2-v1"
_DUMMY_PREFETCH_MAX_INFLIGHT = 1
_HARD_LOAD_RUNWAY = 2
_RUNTIMES: weakref.WeakValueDictionary[int, _StreamedExpertCacheRuntime] = (
    weakref.WeakValueDictionary()
)
logger = init_logger(__name__)


@dataclass(frozen=True)
class _PreparedExpertDemand:
    expert_ids: tuple[int, ...]
    resident_claims: tuple[ExpertClaim, ...]
    pending_ids: tuple[int, ...]
    pending_claims: tuple[ExpertClaim, ...]


@dataclass(frozen=True)
class PreparedBatch:
    """One prepared MoE batch plus its admitted expert demand."""

    kernel_batch: PreparedStreamedMoEBatch
    routing_ids: torch.Tensor
    routing_ready: torch.cuda.Event
    demand: _PreparedExpertDemand | None = None
    next_prefetch_request: _NativeDummyPrefetchRequest | None = None


@dataclass(frozen=True)
class _DummyPrefetchWindow:
    layer_id: int
    copy_window: CopyWindow


@dataclass
class _NativeDummyPrefetchWindow:
    layer_id: int
    scheduler_handle: int
    reservation: ExpertPrefetchReservation
    released: bool = False
    cancel_lock: threading.Lock = field(
        default_factory=threading.Lock,
        repr=False,
    )


@dataclass
class _NativeDummyPrefetchRequest:
    layer_id: int
    done: threading.Event = field(default_factory=threading.Event)
    canceled: bool = False
    activate_requested: bool = False
    start_event: int = 0
    stop_event: int = 0
    window: _NativeDummyPrefetchWindow | None = None


@dataclass(frozen=True)
class _NativeDummyPrefetchBinding:
    binding: BoundExpertCacheLayer
    ready_events: dict[int, torch.cuda.Event]
    copy_job_templates: Mapping[tuple[int, int], CopyJob]


class _DummyExpertPrefetcher:
    """Submit ascending speculative experts between adjacent MoE layers."""

    def __init__(
        self,
        layer_ids: Sequence[int],
        cache: StreamedExpertCache,
        scheduler: CudaCopyScheduler | None = None,
    ) -> None:
        self._next_layer = dict(zip(layer_ids, layer_ids[1:]))
        self._cache = cache
        self._bindings: dict[int, tuple[BoundExpertCacheLayer, int]] = {}
        self._scheduler = scheduler or CudaCopyScheduler(
            max_inflight=min(2, cache.arena_size),
            thread_name="vllm-expert-copy-scheduler",
        )
        self._lock = threading.RLock()
        self._window: _DummyPrefetchWindow | None = None
        self._pause_depth = 0
        self._stopped = False

    @property
    def active_layer_id(self) -> int | None:
        with self._lock:
            return None if self._window is None else self._window.layer_id

    @property
    def pending_count(self) -> int:
        with self._lock:
            window = self._window
        if window is None:
            return 0
        return self._scheduler.pending_count(window.copy_window)

    def register(
        self,
        layer_id: int,
        binding: BoundExpertCacheLayer,
        num_experts: int,
    ) -> None:
        if num_experts <= 0:
            raise ValueError("dummy expert prefetch requires at least one expert")
        with self._lock:
            if layer_id in self._bindings:
                raise ValueError(f"dummy prefetch layer {layer_id} is already bound")
            self._bindings[layer_id] = (binding, num_experts)

    def start_after(self, completed_layer_id: int) -> _DummyPrefetchWindow | None:
        target_layer_id = self._next_layer.get(completed_layer_id)
        if target_layer_id is None:
            return None

        with self._lock:
            if self._stopped or self._pause_depth:
                return None
            registration = self._bindings.get(target_layer_id)
            if registration is None:
                logger.warning(
                    "Skipping dummy expert prefetch for unbound layer %d",
                    target_layer_id,
                )
                return None
            previous_window = self._window
            self._window = None

        if previous_window is not None:
            self._scheduler.cancel_pending(previous_window.copy_window)

        binding, num_experts = registration

        def issue(expert_id: int):
            if binding.slot_for(expert_id) is not None:
                return CopyIssueStatus.SKIPPED
            handles = binding.prefetch((expert_id,))
            if not handles:
                return None
            ready_event = handles[0].ready_event
            if ready_event is None:
                raise RuntimeError("dummy prefetch load has no completion event")
            return ready_event

        def on_error(error: Exception) -> None:
            logger.warning(
                "Dummy expert prefetch failed for layer %d: %s",
                target_layer_id,
                error,
            )

        with self._lock:
            if self._stopped or self._pause_depth:
                return None
            copy_window = self._scheduler.submit_window(
                num_experts,
                issue,
                activate=False,
                on_discard=self._cache.record_prefetch_candidates_discarded,
                on_error=on_error,
            )
            window = _DummyPrefetchWindow(target_layer_id, copy_window)
            self._window = window
        self._cache.record_prefetch_window()
        self._scheduler.activate_window(copy_window, prime_runway=True)
        return window

    def cancel_for_demand(self, layer_id: int) -> None:
        mismatched_layer_id: int | None = None
        with self._lock:
            window = self._window
            if window is not None:
                if window.layer_id != layer_id:
                    mismatched_layer_id = window.layer_id
                self._window = None
        if window is not None:
            self._scheduler.cancel_pending(
                window.copy_window,
                wait_for_issue=False,
                defer_discard=True,
            )
        if mismatched_layer_id is not None:
            logger.warning(
                "Canceled dummy expert prefetch for layer %d at layer %d demand",
                mismatched_layer_id,
                layer_id,
            )

    def cancel_window(self, window: _DummyPrefetchWindow) -> None:
        with self._lock:
            if self._window is window:
                self._window = None
        self._scheduler.cancel_pending(window.copy_window)

    def set_stop_event(
        self,
        layer_id: int,
        event: torch.cuda.Event,
    ) -> None:
        with self._lock:
            window = self._window
            if window is None or window.layer_id != layer_id:
                return
        self._scheduler.set_stop_event(window.copy_window, event)

    def pause_and_drain(self) -> None:
        with self._lock:
            if self._stopped:
                return
            self._pause_depth += 1
            self._window = None
        self._scheduler.pause_and_drain()

    def resume(self) -> None:
        with self._lock:
            if self._pause_depth == 0:
                if self._stopped:
                    return
                raise RuntimeError("dummy expert prefetch is not paused")
            self._pause_depth -= 1
            self._scheduler.resume()

    def close(self, *, wait: bool = True) -> None:
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            self._window = None
        self._scheduler.close(wait=wait)


class _NativeDummyExpertPrefetcher:
    """Submit a precomputed expert window without per-copy Python callbacks."""

    def __init__(
        self,
        layer_ids: Sequence[int],
        cache: StreamedExpertCache,
        scheduler: NativeCudaCopyScheduler,
        copy_stream: torch.cuda.Stream,
    ) -> None:
        self._next_layer = dict(zip(layer_ids, layer_ids[1:]))
        self._cache = cache
        self._scheduler = scheduler
        self._copy_stream = copy_stream
        self._bindings: dict[int, _NativeDummyPrefetchBinding] = {}
        self._lock = threading.RLock()
        self._pause_lock = threading.Lock()
        self._window: _NativeDummyPrefetchWindow | None = None
        self._request: _NativeDummyPrefetchRequest | None = None
        self._requests: SimpleQueue[_NativeDummyPrefetchRequest | None] = SimpleQueue()
        self._planner_error: ExpertCacheError | None = None
        self._pause_depth = 0
        self._stopped = False
        self._close_started = False
        self._close_done = threading.Event()
        self._planner = threading.Thread(
            target=self._planner_loop,
            name="vllm-expert-prefetch-planner",
            daemon=True,
        )
        self._planner.start()

    @property
    def active_layer_id(self) -> int | None:
        with self._lock:
            if self._window is not None:
                return self._window.layer_id
            return None if self._request is None else self._request.layer_id

    @property
    def pending_count(self) -> int:
        with self._lock:
            window = self._window
            if window is None:
                return 0
            with window.cancel_lock:
                if window.released:
                    return 0
                return self._scheduler.pending_count(window.scheduler_handle)

    def register(
        self,
        layer_id: int,
        binding: BoundExpertCacheLayer,
        num_experts: int,
    ) -> None:
        if num_experts <= 0:
            raise ValueError("dummy expert prefetch requires at least one expert")
        slot_indices = binding.prefetch_slot_indices
        ready_events = {slot_index: torch.cuda.Event() for slot_index in slot_indices}
        for event in ready_events.values():
            event.record(self._copy_stream)
        copy_job_templates: dict[tuple[int, int], CopyJob] = {}
        expected_keys = {
            (expert_id, slot_index)
            for expert_id in range(num_experts)
            for slot_index in ready_events
        }
        for (
            expert_id,
            slot_index,
            source,
            destination,
        ) in binding.prefetch_copy_layouts():
            if slot_index not in ready_events:
                continue
            key = (expert_id, slot_index)
            if key not in expected_keys:
                raise ValueError(
                    "native prefetch copy layout contains an unexpected "
                    f"expert-slot pair {key}"
                )
            if key in copy_job_templates:
                raise ValueError(
                    "native prefetch copy layout contains a duplicate "
                    f"expert-slot pair {key}"
                )
            copy_job_templates[key] = self._copy_job_template(
                layer_id,
                expert_id,
                source,
                destination,
                ready_events[slot_index],
            )
        if copy_job_templates.keys() != expected_keys:
            missing = sorted(expected_keys - copy_job_templates.keys())
            raise ValueError(
                "native prefetch copy layout does not cover every registered "
                f"expert-slot pair; missing {missing}"
            )
        with self._lock:
            if layer_id in self._bindings:
                raise ValueError(f"dummy prefetch layer {layer_id} is already bound")
            self._bindings[layer_id] = _NativeDummyPrefetchBinding(
                binding,
                ready_events,
                MappingProxyType(copy_job_templates),
            )

    def start_after(
        self,
        completed_layer_id: int,
    ) -> _NativeDummyPrefetchRequest | None:
        request = self.prepare_after(completed_layer_id)
        if request is not None:
            self.activate_prepared(request)
        return request

    def prepare_after(
        self,
        completed_layer_id: int,
    ) -> _NativeDummyPrefetchRequest | None:
        target_layer_id = self._next_layer.get(completed_layer_id)
        if target_layer_id is None:
            return None

        with self._lock:
            self._raise_planner_error_locked()
            if self._stopped or self._pause_depth:
                return None
            registration = self._bindings.get(target_layer_id)
        if registration is None:
            logger.warning(
                "Skipping dummy expert prefetch for unbound layer %d",
                target_layer_id,
            )
            return None
        with self._lock:
            self._raise_planner_error_locked()
            if self._stopped or self._pause_depth:
                return None
            if self._request is not None or self._window is not None:
                raise RuntimeError(
                    "cannot schedule a new dummy prefetch while one is active"
                )
            request = _NativeDummyPrefetchRequest(target_layer_id)
            self._request = request
            self._requests.put(request)
            return request

    def activate_prepared(
        self,
        request: _NativeDummyPrefetchRequest,
        start_event: torch.cuda.Event | None = None,
    ) -> None:
        event_handle = 0 if start_event is None else int(start_event.cuda_event)
        if start_event is not None and not event_handle:
            raise RuntimeError("prefetch start event has not been recorded")
        with self._lock:
            self._raise_planner_error_locked()
            if request.canceled or self._stopped or self._pause_depth:
                return
            if self._request is not request:
                return
            if request.start_event and request.start_event != event_handle:
                raise RuntimeError("prefetch start event was already set")
            request.start_event = event_handle
            request.activate_requested = True
        try:
            self._activate_if_ready(request)
        except Exception as exc:
            error = self._record_planner_failure(request, exc)
            logger.exception(
                "Could not activate dummy expert prefetch for layer %d",
                request.layer_id,
            )
            raise error from exc

    def cancel_for_demand(self, layer_id: int) -> None:
        mismatched_layer_id: int | None = None
        with self._lock:
            request = self._request
            if request is not None:
                if request.layer_id != layer_id:
                    mismatched_layer_id = request.layer_id
                request.canceled = True
                self._request = None
            window = self._window
            if window is not None:
                if window.layer_id != layer_id:
                    mismatched_layer_id = window.layer_id
                self._window = None
        self._cancel_request_and_window(request, window)
        self._raise_planner_error()
        if mismatched_layer_id is not None:
            logger.warning(
                "Canceled dummy expert prefetch for layer %d at layer %d demand",
                mismatched_layer_id,
                layer_id,
            )

    def cancel_window(
        self,
        item: _NativeDummyPrefetchRequest | _NativeDummyPrefetchWindow,
    ) -> None:
        if isinstance(item, _NativeDummyPrefetchRequest):
            with self._lock:
                if self._request is item:
                    item.canceled = True
                    self._request = None
            item.done.wait()
            window = item.window
            if window is None:
                self._raise_planner_error()
                return
        else:
            window = item
        with self._lock:
            if self._window is window:
                self._window = None
        self._cancel_and_reconcile(window)
        self._raise_planner_error()

    def set_stop_event(
        self,
        layer_id: int,
        event: torch.cuda.Event,
    ) -> None:
        event_handle = event.cuda_event
        if not event_handle:
            raise RuntimeError("routing stop event has not been recorded")
        with self._lock:
            self._raise_planner_error_locked()
            request = self._request
            if request is not None and request.layer_id == layer_id:
                if request.stop_event:
                    raise RuntimeError("routing stop event was already set")
                request.stop_event = event_handle
                return
            window = self._window
            if window is None or window.layer_id != layer_id:
                return
            self._scheduler.set_stop_event(window.scheduler_handle, event_handle)

    def pause_and_drain(self) -> None:
        with self._pause_lock:
            with self._lock:
                if self._stopped:
                    return
                self._pause_depth += 1
                request = self._request
                if request is not None:
                    request.canceled = True
                    self._request = None
                window = self._window
                self._window = None

            scheduler_pause_started = False
            try:
                self._cancel_request_and_window(request, window)
                self._raise_planner_error()
                scheduler_pause_started = True
                self._scheduler.pause_and_drain()
            except Exception as error:
                rollback_error: Exception | None = None
                if scheduler_pause_started:
                    try:
                        self._scheduler.resume()
                    except Exception as exc:
                        rollback_error = exc
                with self._lock:
                    self._pause_depth -= 1
                    if rollback_error is not None:
                        self._stopped = True
                if rollback_error is None:
                    raise
                fatal_error = ExpertCacheLoadError(
                    "native expert prefetch pause failed and could not be "
                    f"rolled back: {error}"
                )
                fatal_error.add_note(f"Scheduler resume also failed: {rollback_error}")
                self._cache.fail_closed(fatal_error)
                raise fatal_error from error

    def resume(self) -> None:
        with self._pause_lock:
            with self._lock:
                if self._pause_depth == 0:
                    if self._stopped:
                        return
                    raise RuntimeError("dummy expert prefetch is not paused")
            self._scheduler.resume()
            with self._lock:
                self._pause_depth -= 1

    def close(self, *, wait: bool = True) -> None:
        cleanup: (
            tuple[
                _NativeDummyPrefetchRequest | None,
                _NativeDummyPrefetchWindow | None,
            ]
            | None
        ) = None
        with self._lock:
            if not self._close_started:
                self._close_started = True
                self._stopped = True
                request = self._request
                if request is not None:
                    request.canceled = True
                    self._request = None
                window = self._window
                self._window = None
                cleanup = (request, window)
        if cleanup is None:
            if wait:
                self._close_done.wait()
            return
        if wait:
            self._finish_close(*cleanup)
            return
        threading.Thread(
            target=self._finish_close_in_background,
            args=cleanup,
            name="vllm-expert-prefetch-close",
            daemon=True,
        ).start()

    def _finish_close_in_background(
        self,
        request: _NativeDummyPrefetchRequest | None,
        window: _NativeDummyPrefetchWindow | None,
    ) -> None:
        try:
            self._finish_close(request, window)
        except Exception:
            logger.exception("Could not close the native expert copy scheduler")

    def _finish_close(
        self,
        request: _NativeDummyPrefetchRequest | None,
        window: _NativeDummyPrefetchWindow | None,
    ) -> None:
        try:
            self._cancel_request_and_window(request, window)
        finally:
            try:
                self._requests.put(None)
                self._planner.join()
            finally:
                try:
                    self._scheduler.close()
                finally:
                    self._close_done.set()

    def _planner_loop(self) -> None:
        while True:
            request = self._requests.get()
            if request is None:
                return
            try:
                self._prepare_request(request)
            except Exception as exc:
                self._record_planner_failure(request, exc)
                logger.exception(
                    "Dummy expert prefetch planner failed for layer %d",
                    request.layer_id,
                )
            finally:
                request.done.set()

    def _prepare_request(
        self,
        request: _NativeDummyPrefetchRequest,
    ) -> None:
        with self._lock:
            if (
                request.canceled
                or self._request is not request
                or self._stopped
                or self._pause_depth
            ):
                return
            registration = self._bindings[request.layer_id]

        with torch.cuda.device(self._copy_stream.device):
            with torch.cuda.nvtx.range("expert_cache:prefetch_prepare:reserve"):
                reservation = registration.binding.reserve_prefetch(
                    registration.ready_events
                )
            if reservation is None:
                with self._lock:
                    if self._request is request:
                        self._request = None
                return
            if not self._request_is_active(request):
                reservation.abort()
                return
            try:
                with torch.cuda.nvtx.range("expert_cache:prefetch_prepare:materialize"):
                    jobs = tuple(
                        self._copy_job(
                            copy,
                            registration.copy_job_templates[
                                (copy.expert_id, copy.slot_index)
                            ],
                        )
                        for copy in reservation.copies
                    )
                if not self._request_is_active(request):
                    reservation.abort()
                    return
                with torch.cuda.nvtx.range("expert_cache:prefetch_prepare:native"):
                    scheduler_handle = self._scheduler.prepare_window(jobs)
            except Exception as error:
                reservation.abort()
                with self._lock:
                    if self._request is request:
                        self._request = None
                logger.warning(
                    "Could not prepare dummy expert prefetch for layer %d: %s",
                    request.layer_id,
                    error,
                )
                return

        window = _NativeDummyPrefetchWindow(
            request.layer_id,
            scheduler_handle,
            reservation,
        )
        with self._lock:
            request.window = window
            should_keep = (
                not request.canceled
                and self._request is request
                and not self._stopped
                and not self._pause_depth
            )
        if not should_keep:
            self._cancel_and_reconcile(window)
            return
        self._activate_if_ready(request)

    def _activate_if_ready(
        self,
        request: _NativeDummyPrefetchRequest,
    ) -> bool:
        activation_error: Exception | None = None
        window: _NativeDummyPrefetchWindow | None = None
        with self._lock:
            if (
                not request.activate_requested
                or request.canceled
                or self._request is not request
                or self._stopped
                or self._pause_depth
                or request.window is None
            ):
                return False
            window = request.window
            try:
                with torch.cuda.nvtx.range("expert_cache:prefetch_prepare:activate"):
                    self._scheduler.activate_window(
                        window.scheduler_handle,
                        start_event=request.start_event,
                        stop_event=request.stop_event,
                    )
            except Exception as exc:
                activation_error = exc
                request.canceled = True
                self._request = None
            else:
                self._window = window
                self._request = None
        assert window is not None
        if activation_error is not None:
            try:
                self._cancel_and_reconcile(window)
            except Exception as cleanup_error:
                activation_error.add_note(
                    f"Window cleanup also failed: {cleanup_error}"
                )
            raise activation_error
        try:
            self._cache.record_prefetch_window()
        except Exception as error:
            with self._lock:
                if self._window is window:
                    self._window = None
            try:
                self._cancel_and_reconcile(window)
            except Exception as cleanup_error:
                error.add_note(f"Window cleanup also failed: {cleanup_error}")
            raise
        return True

    def _record_planner_failure(
        self,
        request: _NativeDummyPrefetchRequest,
        error: Exception,
    ) -> ExpertCacheError:
        failure = (
            error
            if isinstance(error, ExpertCacheError)
            else ExpertCacheLoadError("dummy expert prefetch planner failed")
        )
        if failure is not error:
            failure.__cause__ = error
        with self._lock:
            if self._planner_error is None:
                self._planner_error = failure
            if self._request is request:
                request.canceled = True
                self._request = None
        self._cache.fail_closed(failure)
        return failure

    def _cancel_request_and_window(
        self,
        request: _NativeDummyPrefetchRequest | None,
        window: _NativeDummyPrefetchWindow | None,
    ) -> None:
        if request is not None:
            request.done.wait()
        request_window = None if request is None else request.window
        if request_window is not None:
            self._cancel_and_reconcile(request_window)
        if window is not None and window is not request_window:
            self._cancel_and_reconcile(window)

    def _request_is_active(self, request: _NativeDummyPrefetchRequest) -> bool:
        with self._lock:
            return (
                not request.canceled
                and self._request is request
                and not self._stopped
                and not self._pause_depth
            )

    def _raise_planner_error(self) -> None:
        with self._lock:
            self._raise_planner_error_locked()

    def _raise_planner_error_locked(self) -> None:
        if self._planner_error is not None:
            raise self._planner_error

    def _cancel_and_reconcile(self, window: _NativeDummyPrefetchWindow) -> None:
        with window.cancel_lock:
            if window.released:
                return
            failure: ExpertCacheLoadError | None = None
            try:
                with torch.cuda.nvtx.range("expert_cache:prefetch_cancel:snapshot"):
                    snapshot = self._scheduler.cancel_and_snapshot(
                        window.scheduler_handle
                    )
            except Exception as error:
                failure = self._prefetch_window_failure(
                    window,
                    "snapshot",
                    error,
                )
            else:
                try:
                    if window.reservation.active:
                        with torch.cuda.nvtx.range(
                            "expert_cache:prefetch_cancel:reconcile"
                        ):
                            self._reconcile_snapshot(window, snapshot)
                except Exception as error:
                    failure = self._prefetch_window_failure(
                        window,
                        "reconcile",
                        error,
                    )
            try:
                with torch.cuda.nvtx.range("expert_cache:prefetch_cancel:release"):
                    self._scheduler.release_window(window.scheduler_handle)
            except Exception as error:
                release_failure = self._prefetch_window_failure(
                    window,
                    "release",
                    error,
                )
                if failure is None:
                    failure = release_failure
                else:
                    failure.add_note(str(release_failure))
            finally:
                window.released = True
            if failure is not None:
                self._cache.fail_closed(failure)
                raise failure

    @staticmethod
    def _prefetch_window_failure(
        window: _NativeDummyPrefetchWindow,
        operation: str,
        error: Exception,
    ) -> ExpertCacheLoadError:
        failure = ExpertCacheLoadError(
            f"native expert prefetch {operation} failed for layer "
            f"{window.layer_id}: {error}"
        )
        failure.__cause__ = error
        return failure

    @staticmethod
    def _reconcile_snapshot(
        window: _NativeDummyPrefetchWindow,
        snapshot: CopyWindowSnapshot,
    ) -> None:
        if snapshot.error is not None or snapshot.failed:
            details = snapshot.error or f"failed cookies: {snapshot.failed}"
            error = ExpertCacheLoadError(
                f"native expert prefetch failed for layer {window.layer_id}: {details}"
            )
            window.reservation.fail_closed(error)
            raise error
        window.reservation.reconcile(snapshot.issued)

    @staticmethod
    def _copy_job_template(
        layer_id: int,
        expert_id: int,
        source: ExpertWeightBundle,
        destination: ExpertWeightBundle,
        ready_event: torch.cuda.Event,
    ) -> CopyJob:
        if not source.is_pinned:
            raise ValueError("native expert prefetch source must be pinned")
        segments = []
        for name, source_tensor in source.tensors.items():
            destination_tensor = destination.tensors[name]
            if (
                not source_tensor.is_contiguous()
                or not destination_tensor.is_contiguous()
            ):
                raise ValueError("native expert prefetch requires contiguous tensors")
            segments.append(
                CopySegment(
                    src=source_tensor.data_ptr(),
                    dst=destination_tensor.data_ptr(),
                    nbytes=source_tensor.nbytes,
                    kind=CudaMemcpyKind.HOST_TO_DEVICE,
                )
            )
        done_event = getattr(ready_event, "cuda_event", 0)
        if not done_event:
            raise RuntimeError("native expert prefetch ready event is uninitialized")
        return CopyJob(
            cookie=0,
            segments=tuple(segments),
            done_event=int(done_event),
            label=f"expert_cache:prefetch:{layer_id}:{expert_id}",
        )

    @staticmethod
    def _copy_job(
        copy: ExpertPrefetchCopy,
        template: CopyJob,
    ) -> CopyJob:
        wait_event = (
            0 if copy.wait_for is None else int(getattr(copy.wait_for, "cuda_event", 0))
        )
        if copy.wait_for is not None and not wait_event:
            raise RuntimeError("native expert prefetch wait event is uninitialized")
        return CopyJob(
            cookie=copy.cookie,
            segments=template.segments,
            wait_event=wait_event,
            done_event=template.done_event,
            label=template.label,
        )


class _StreamedExpertCacheRuntime:
    def __init__(
        self,
        vllm_config: VllmConfig,
        prototype_w13: torch.Tensor,
        prototype_w2: torch.Tensor,
        device: torch.device,
        load_token: object,
        *,
        format_class: str = _FORMAT_CLASS,
        runtime_specs: Mapping[str, Any] | None = None,
        activation_dtype: torch.dtype | None = None,
        hidden_size: int | None = None,
    ) -> None:
        self.vllm_config = vllm_config
        offload_config = vllm_config.offload_config
        self.per_layer_size = offload_config.expert_cache_per_layer_size
        self.shared_size = offload_config.expert_cache_shared_size
        self.layer_ids = _moe_layer_ids(vllm_config)
        self.layer_ordinals = {
            layer_id: ordinal for ordinal, layer_id in enumerate(self.layer_ids)
        }
        self.device = device
        self.format_class = format_class
        self.activation_dtype = activation_dtype or prototype_w13.dtype
        self.hidden_size = hidden_size or prototype_w13.shape[-1]
        self._config_ref = weakref.ref(vllm_config)
        self._load_token = load_token
        self._forward_lock = threading.Lock()
        self._last_compute_stream_id: tuple[torch.device, int] | None = None
        self._last_compute_done: torch.cuda.Event | None = None
        self._stream_guard_failed = False
        self._wave_capture_stream: torch.cuda.Stream | None = None
        self._wave_graph_pool: Any | None = None
        self._shared_staging: (
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ]
            | None
        ) = None
        self._shared_execution_buffers: StreamedMoEBuffers | None = None
        self._shared_host_staging: torch.Tensor | None = None
        self._shared_host_num_experts: int | None = None

        arena_size = len(self.layer_ids) * self.per_layer_size + self.shared_size
        if arena_size <= 0:
            raise ValueError("streamed expert caching requires at least one arena slot")
        padded_arena_size = ((arena_size + 31) // 32) * 32
        if padded_arena_size >= 1024:
            raise ValueError(
                "streamed expert-cache arena is too large for MoE alignment: "
                f"{arena_size} slots (maximum 992)"
            )
        if runtime_specs is None:
            runtime_layout = {
                "w13_weight": (prototype_w13.shape[1:], prototype_w13.dtype),
                "w2_weight": (prototype_w2.shape[1:], prototype_w2.dtype),
            }
        else:
            runtime_layout = {
                name: _runtime_tensor_spec(spec) for name, spec in runtime_specs.items()
            }
        if not {"w13_weight", "w2_weight"} <= set(runtime_layout):
            raise ValueError(
                "streamed expert runtime requires w13_weight and w2_weight"
            )
        self.arena_tensors = {
            name: torch.empty(
                (arena_size, *shape),
                dtype=dtype,
                device=device,
            )
            for name, (shape, dtype) in runtime_layout.items()
        }
        self.w13 = self.arena_tensors["w13_weight"]
        self.w2 = self.arena_tensors["w2_weight"]
        arena = tuple(
            ExpertWeightBundle(
                format_class,
                {name: tensor[slot] for name, tensor in self.arena_tensors.items()},
            )
            for slot in range(arena_size)
        )
        shared_start = len(self.layer_ids) * self.per_layer_size
        coordinator = CudaExpertTransferCoordinator(device)
        self._coordinator = coordinator
        native_copy_scheduler: NativeCudaCopyScheduler | None = None
        if offload_config.expert_cache_prefetch_policy == "dummy":
            coordinator.register_native_destinations(arena)
            native_copy_scheduler = NativeCudaCopyScheduler(
                coordinator.copy_stream.cuda_stream,
                device=coordinator.device.index,
                max_inflight=min(
                    _DUMMY_PREFETCH_MAX_INFLIGHT,
                    arena_size,
                ),
                poll_interval_us=10,
            )
            coordinator.attach_native_scheduler(native_copy_scheduler)
        try:
            self.cache = StreamedExpertCache(
                arena,
                shared_slot_indices=range(shared_start, arena_size),
                coordinator=coordinator,
                policy=FIFOExpertCachePolicy(),
            )
        except Exception:
            if native_copy_scheduler is not None:
                native_copy_scheduler.close()
            raise
        self._dummy_prefetcher = (
            _NativeDummyExpertPrefetcher(
                self.layer_ids,
                self.cache,
                native_copy_scheduler,
                coordinator.copy_stream,
            )
            if native_copy_scheduler is not None
            else None
        )
        self._register_dummy_prefetch_finalizer()

    def _register_dummy_prefetch_finalizer(self) -> None:
        prefetcher = self._dummy_prefetcher
        self._dummy_prefetch_finalizer = (
            weakref.finalize(self, prefetcher.close, wait=True)
            if prefetcher is not None
            else None
        )

    @property
    def capacity_per_wave(self) -> int:
        return self.per_layer_size + self.shared_size

    def tensor(self, name: str) -> torch.Tensor:
        try:
            return self.arena_tensors[name]
        except KeyError as exc:
            raise KeyError(f"unknown streamed expert tensor {name!r}") from exc

    @property
    def stats(self) -> ExpertCacheStats:
        return self.cache.stats

    def belongs_to(self, config: VllmConfig, load_token: object) -> bool:
        return self._config_ref() is config and self._load_token is load_token

    def bind(
        self,
        layer_id: int,
        host_bundles: dict[int, ExpertWeightBundle],
    ) -> BoundExpertCacheLayer:
        try:
            ordinal = self.layer_ordinals[layer_id]
        except KeyError as exc:
            raise ValueError(f"layer {layer_id} is not a configured MoE layer") from exc
        reserved_start = ordinal * self.per_layer_size
        binding = self.cache.bind(
            LayerBinding(
                layer_id=layer_id,
                format_class=self.format_class,
                host_bundles=host_bundles,
                reserved_slot_indices=range(
                    reserved_start,
                    reserved_start + self.per_layer_size,
                ),
            )
        )
        if self._dummy_prefetcher is not None:
            self._dummy_prefetcher.register(
                layer_id,
                binding,
                len(host_bundles),
            )
        return binding

    def start_dummy_prefetch_after(
        self,
        layer_id: int,
    ) -> (
        _DummyPrefetchWindow
        | _NativeDummyPrefetchRequest
        | _NativeDummyPrefetchWindow
        | None
    ):
        if self._dummy_prefetcher is not None:
            return self._dummy_prefetcher.start_after(layer_id)
        return None

    def prepare_dummy_prefetch_after(
        self,
        layer_id: int,
    ) -> _NativeDummyPrefetchRequest | None:
        if isinstance(self._dummy_prefetcher, _NativeDummyExpertPrefetcher):
            return self._dummy_prefetcher.prepare_after(layer_id)
        return None

    def activate_dummy_prefetch(
        self,
        request: _NativeDummyPrefetchRequest,
        start_event: torch.cuda.Event | None,
    ) -> None:
        if not isinstance(self._dummy_prefetcher, _NativeDummyExpertPrefetcher):
            raise RuntimeError("only native dummy prefetch supports early preparation")
        self._dummy_prefetcher.activate_prepared(request, start_event)

    def cancel_dummy_prefetch_for(self, layer_id: int) -> None:
        if self._dummy_prefetcher is not None:
            self._dummy_prefetcher.cancel_for_demand(layer_id)

    def cancel_dummy_prefetch_window(
        self,
        window: (
            _DummyPrefetchWindow
            | _NativeDummyPrefetchRequest
            | _NativeDummyPrefetchWindow
        ),
    ) -> None:
        if self._dummy_prefetcher is not None:
            self._dummy_prefetcher.cancel_window(window)

    def set_dummy_prefetch_stop_event(
        self,
        layer_id: int,
        event: torch.cuda.Event,
    ) -> None:
        if self._dummy_prefetcher is not None:
            self._dummy_prefetcher.set_stop_event(layer_id, event)

    def pause_dummy_prefetch_and_drain(self) -> None:
        if self._dummy_prefetcher is not None:
            self._coordinator.pause_native_submissions()
            try:
                self._dummy_prefetcher.pause_and_drain()
            except Exception:
                self._coordinator.resume_native_submissions()
                raise

    def resume_dummy_prefetch(self) -> None:
        if self._dummy_prefetcher is not None:
            self._dummy_prefetcher.resume()
            self._coordinator.resume_native_submissions()

    def get_staging(
        self,
        *,
        max_num_tokens: int,
        hidden_size: int,
        top_k: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return model-wide staging shared by serialized MoE layers."""
        if self._shared_staging is None:
            self._shared_staging = (
                torch.empty(
                    (max_num_tokens, hidden_size),
                    dtype=dtype,
                    device=self.device,
                ),
                torch.empty(
                    (max_num_tokens, hidden_size),
                    dtype=dtype,
                    device=self.device,
                ),
                torch.empty(
                    (max_num_tokens, top_k),
                    dtype=torch.float32,
                    device=self.device,
                ),
                torch.empty(
                    (max_num_tokens, top_k),
                    dtype=torch.int32,
                    device=self.device,
                ),
            )
        hidden, shared_output, topk_weights, topk_ids = self._shared_staging
        if (
            hidden.shape != (max_num_tokens, hidden_size)
            or hidden.dtype != dtype
            or shared_output.shape != (max_num_tokens, hidden_size)
            or shared_output.dtype != dtype
            or topk_weights.shape != (max_num_tokens, top_k)
            or topk_ids.shape != (max_num_tokens, top_k)
        ):
            raise ValueError(
                "streamed expert-cache layers must share token, hidden, shared "
                "output, and top-k staging geometry"
            )
        return self._shared_staging

    def get_execution_buffers(
        self,
        kernel: FusedMoEKernel,
        *,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
    ) -> StreamedMoEBuffers:
        """Return model-wide fixed workspaces shared by serialized layers."""
        if self._shared_execution_buffers is None:
            self._shared_execution_buffers = kernel.create_streamed_buffers(
                hidden_states=hidden_states,
                w1=self.w13,
                w2=self.w2,
                topk_ids=topk_ids,
                activation=activation,
                global_num_experts=global_num_experts,
            )
        return self._shared_execution_buffers

    def get_host_staging(
        self,
        *,
        max_num_tokens: int,
        top_k: int,
        num_experts: int,
    ) -> torch.Tensor:
        """Return model-wide pinned routing staging."""
        if self._shared_host_staging is None:
            self._shared_host_staging = torch.empty(
                (max_num_tokens, top_k),
                dtype=torch.int32,
                device="cpu",
            ).pin_memory()
            self._shared_host_num_experts = num_experts
        routing = self._shared_host_staging
        if (
            routing.shape != (max_num_tokens, top_k)
            or self._shared_host_num_experts != num_experts
        ):
            raise ValueError(
                "streamed expert-cache layers must share routing staging geometry"
            )
        return routing

    def begin_forward(self, stream: torch.cuda.Stream) -> None:
        """Reject fixed-buffer reuse from an unordered CUDA stream."""
        if self._stream_guard_failed:
            raise RuntimeError("streamed expert-cache stream guard is unavailable")
        if self._last_compute_done is None:
            return
        stream_id = _stream_identity(stream)
        if stream_id == self._last_compute_stream_id:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "streamed expert caching does not support overlapping CUDA streams"
            )
        try:
            complete = self._last_compute_done.query()
        except Exception as exc:
            self._stream_guard_failed = True
            raise RuntimeError(
                "failed to query streamed expert-cache forward completion"
            ) from exc
        if not complete:
            raise RuntimeError(
                "streamed expert caching does not support overlapping CUDA streams"
            )
        if complete:
            self._last_compute_done = None
            self._last_compute_stream_id = None

    def finish_forward(self, stream: torch.cuda.Stream) -> None:
        """Record when graph-visible buffers become reusable cross-stream."""
        try:
            done = torch.cuda.Event()
            done.record(stream)
        except Exception as exc:
            self._stream_guard_failed = True
            raise RuntimeError(
                "failed to record streamed expert-cache forward completion"
            ) from exc
        self._last_compute_stream_id = _stream_identity(stream)
        self._last_compute_done = done


class CachedExpertLayer:
    """Host-scheduled streamed execution for one routed-expert layer."""

    def __init__(
        self,
        runtime: _StreamedExpertCacheRuntime,
        binding: BoundExpertCacheLayer,
        *,
        kernel: FusedMoEKernel,
        activation: MoEActivation,
        num_experts: int,
        max_num_tokens: int,
        top_k: int,
    ) -> None:
        self.runtime = runtime
        self.binding = binding
        self.num_experts = num_experts
        self.max_num_tokens = max_num_tokens
        self.top_k = top_k
        self._kernel: FusedMoEKernel | None = kernel
        self._active_batch: PreparedStreamedMoEBatch | None = None
        self._prepare_stream: torch.cuda.Stream | None = None

        (
            self._hidden_staging,
            self._shared_output_staging,
            self._topk_weights_staging,
            self._topk_ids_staging,
        ) = runtime.get_staging(
            max_num_tokens=max_num_tokens,
            hidden_size=runtime.hidden_size,
            top_k=top_k,
            dtype=runtime.activation_dtype,
        )
        self._execution_buffers = runtime.get_execution_buffers(
            kernel,
            hidden_states=self._hidden_staging,
            topk_ids=self._topk_ids_staging,
            activation=activation,
            global_num_experts=num_experts,
        )
        self.wave_expert_map = torch.full(
            (num_experts,), -1, dtype=torch.int32, device=runtime.device
        )
        self._routing_staging = runtime.get_host_staging(
            max_num_tokens=max_num_tokens,
            top_k=top_k,
            num_experts=num_experts,
        )
        self._routing_available = torch.cuda.Event()
        self._routing_ready = torch.cuda.Event()
        self._prefetch_start = torch.cuda.Event()

        max_routes = max_num_tokens * top_k
        alignment_namespace = max(num_experts, runtime.cache.arena_size)
        max_padded = max_routes + alignment_namespace * 255
        if max_routes < alignment_namespace:
            max_padded = min(max_routes * 256, max_padded)
        self._alignment_outputs = (
            torch.empty(max_padded, dtype=torch.int32, device=runtime.device),
            torch.empty(max_padded, dtype=torch.int32, device=runtime.device),
            torch.empty(1, dtype=torch.int32, device=runtime.device),
            torch.empty(
                alignment_namespace + 1,
                dtype=torch.int32,
                device=runtime.device,
            ),
        )
        cudagraph_mode = runtime.vllm_config.compilation_config.cudagraph_mode
        self._wave_runner: Callable[..., Any]
        if cudagraph_mode.has_piecewise_cudagraphs():
            if runtime._wave_capture_stream is None:
                runtime._wave_capture_stream = torch.cuda.Stream(device=runtime.device)
            if runtime._wave_graph_pool is None:
                runtime._wave_graph_pool = torch.cuda.graph_pool_handle()
            self._wave_runner = CUDAGraphWrapper(
                self._run_active_wave,
                runtime.vllm_config,
                CUDAGraphMode.PIECEWISE,
                CUDAGraphOptions(
                    debug_log_enable=False,
                    gc_disable=True,
                    weak_ref_output=False,
                    capture_stream=runtime._wave_capture_stream,
                    graph_pool=runtime._wave_graph_pool,
                    use_global_graph_pool=False,
                    isolate_graph_pool=True,
                    use_direct_capture=True,
                    warmup_before_capture=True,
                    capture_error_mode="thread_local",
                ),
            )
        else:
            self._wave_runner = self._run_active_wave

    @property
    def stats(self) -> ExpertCacheStats:
        return self.runtime.stats

    @property
    def graph_data_ptrs(self) -> tuple[int, ...]:
        return tuple(tensor.data_ptr() for tensor in self._graph_closure_tensors()) + (
            self._hidden_staging.data_ptr(),
            self._shared_output_staging.data_ptr(),
            self._topk_weights_staging.data_ptr(),
            self._topk_ids_staging.data_ptr(),
            self._execution_buffers.workspace13.data_ptr(),
            self._execution_buffers.workspace2.data_ptr(),
            self._execution_buffers.route_output.data_ptr(),
            self._execution_buffers.output.data_ptr(),
            self.wave_expert_map.data_ptr(),
            *(tensor.data_ptr() for tensor in self._alignment_outputs),
        )

    def _graph_closure_tensors(self) -> tuple[torch.Tensor, ...]:
        kernel = self._kernel
        graph_tensors = ()
        if kernel is not None:
            get_graph_tensors = getattr(
                kernel.fused_experts,
                "streamed_graph_tensors",
                None,
            )
            if get_graph_tensors is not None:
                graph_tensors = tuple(get_graph_tensors())
        return (
            *self.runtime.arena_tensors.values(),
            *graph_tensors,
        )

    def set_kernel(self, kernel: FusedMoEKernel) -> None:
        self._kernel = kernel

    def stage_shared_output(self, output: torch.Tensor) -> torch.Tensor:
        """Copy shared-expert output into fixed graph-visible storage."""
        expected_hidden_size = self._shared_output_staging.shape[1]
        if output.ndim != 2 or output.shape[1] != expected_hidden_size:
            raise ValueError(
                "expected shared-expert output with shape "
                f"[tokens, {expected_hidden_size}], got {tuple(output.shape)}"
            )
        if output.shape[0] > self.max_num_tokens:
            raise ValueError(
                f"shared-expert output has {output.shape[0]} tokens, but the fixed "
                f"staging capacity is {self.max_num_tokens}"
            )
        if output.dtype != self._shared_output_staging.dtype:
            raise ValueError(
                "shared-expert output dtype must match streamed expert staging"
            )
        if output.device != self.runtime.device:
            raise ValueError(
                "shared-expert output must be on the streamed expert-cache device"
            )

        staged_output = self._shared_output_staging[: output.shape[0]]
        staged_output.copy_(output, non_blocking=True)
        return staged_output

    def prefetch(self, expert_ids: Iterable[int]) -> tuple[ExpertLoadHandle, ...]:
        """Start best-effort loads without displacing hard demand."""
        return self.binding.prefetch(expert_ids)

    def prepare(
        self,
        kernel: FusedMoEKernel,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        activation: MoEActivation,
        global_num_experts: int,
        apply_router_weight_on_input: bool,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> PreparedBatch:
        """Begin fixed-buffer routing readback and prepare the kernel once."""
        if not self.runtime._forward_lock.acquire(blocking=False):
            raise RuntimeError(
                "streamed expert caching does not support overlapping forwards"
            )
        stream = torch.cuda.current_stream(self.runtime.device)
        try:
            if _stream_identity(get_vllm_current_stream()) != _stream_identity(stream):
                torch.cuda.set_stream(stream)
            self.runtime.begin_forward(stream)
            self._prepare_stream = stream
            return self._prepare_locked(
                kernel,
                x,
                topk_weights,
                topk_ids,
                activation=activation,
                global_num_experts=global_num_experts,
                apply_router_weight_on_input=apply_router_weight_on_input,
                shared_experts=shared_experts,
                shared_experts_input=shared_experts_input,
            )
        except Exception:
            try:
                self.runtime.cancel_dummy_prefetch_for(self.binding.layer_id)
            finally:
                if self._prepare_stream is not None:
                    self.runtime.finish_forward(self._prepare_stream)
                self._prepare_stream = None
                self.runtime._forward_lock.release()
            raise

    def _prepare_locked(
        self,
        kernel: FusedMoEKernel,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        activation: MoEActivation,
        global_num_experts: int,
        apply_router_weight_on_input: bool,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> PreparedBatch:
        if topk_ids.dtype != torch.int32:
            raise ValueError("streamed expert caching requires int32 routing IDs")
        if topk_weights.dtype != torch.float32:
            raise ValueError("streamed expert caching requires float32 routing weights")
        if topk_ids.ndim != 2 or topk_ids.shape[1] != self.top_k:
            raise ValueError(
                f"expected topk_ids with shape [tokens, {self.top_k}], got "
                f"{tuple(topk_ids.shape)}"
            )
        if topk_weights.shape != topk_ids.shape:
            raise ValueError(
                "topk_weights and topk_ids must have the same two-dimensional shape"
            )
        expected_hidden_size = self._hidden_staging.shape[1]
        if x.ndim != 2 or x.shape != (topk_ids.shape[0], expected_hidden_size):
            raise ValueError(
                "expected hidden states with shape "
                f"[{topk_ids.shape[0]}, {expected_hidden_size}], got {tuple(x.shape)}"
            )
        if x.dtype != self._hidden_staging.dtype:
            raise ValueError("streamed expert caching requires BF16 routed activations")
        if (
            x.device != self.runtime.device
            or topk_weights.device != self.runtime.device
            or topk_ids.device != self.runtime.device
        ):
            raise ValueError("streamed expert inputs must be on the cache CUDA device")
        if topk_ids.shape[0] > self.max_num_tokens:
            raise ValueError(
                f"routing batch has {topk_ids.shape[0]} tokens, but the fixed "
                f"staging capacity is {self.max_num_tokens}"
            )

        self.set_kernel(kernel)
        num_tokens = topk_ids.shape[0]
        hidden_states = self._hidden_staging[:num_tokens]
        staged_topk_weights = self._topk_weights_staging[:num_tokens]
        staged_topk_ids = self._topk_ids_staging[:num_tokens]
        routing_ids = self._routing_staging[:num_tokens]
        current_stream = torch.cuda.current_stream(self.runtime.device)
        self._routing_available.record(current_stream)
        self.runtime.set_dummy_prefetch_stop_event(
            self.binding.layer_id,
            self._routing_available,
        )
        routing_ids.copy_(topk_ids, non_blocking=True)
        self._routing_ready.record(current_stream)
        demand = self._prepare_expert_demand(routing_ids, self._routing_ready)
        next_prefetch_request: _NativeDummyPrefetchRequest | None = None
        try:
            next_prefetch_request = self.runtime.prepare_dummy_prefetch_after(
                self.binding.layer_id
            )
            hidden_states.copy_(x, non_blocking=True)
            staged_topk_weights.copy_(topk_weights, non_blocking=True)
            staged_topk_ids.copy_(topk_ids, non_blocking=True)
            kernel_batch = kernel.prepare_streamed(
                hidden_states=hidden_states,
                w1=self.runtime.w13,
                w2=self.runtime.w2,
                topk_weights=staged_topk_weights,
                topk_ids=staged_topk_ids,
                activation=activation,
                global_num_experts=global_num_experts,
                apply_router_weight_on_input=apply_router_weight_on_input,
                shared_experts=shared_experts,
                shared_experts_input=shared_experts_input,
                buffers=self._execution_buffers,
            )
            if (
                kernel_batch.a1q.data_ptr() != hidden_states.data_ptr()
                or kernel_batch.a1q_scale is not None
                or kernel_batch.topk_weights.data_ptr()
                != staged_topk_weights.data_ptr()
                or kernel_batch.topk_ids.data_ptr() != staged_topk_ids.data_ptr()
            ):
                raise RuntimeError(
                    "streamed expert caching requires undispatched BF16 modular "
                    "MoE inputs"
                )
        except Exception as error:
            self._release_prepared_demand(demand)
            if next_prefetch_request is not None:
                try:
                    self.runtime.cancel_dummy_prefetch_window(next_prefetch_request)
                except Exception as cleanup_error:
                    error.add_note(
                        f"Prepared prefetch cleanup also failed: {cleanup_error}"
                    )
            raise
        return PreparedBatch(
            kernel_batch,
            routing_ids,
            self._routing_ready,
            demand,
            next_prefetch_request,
        )

    def _prepare_expert_demand(
        self,
        routing_ids: torch.Tensor,
        routing_ready: torch.cuda.Event,
    ) -> _PreparedExpertDemand:
        with torch.cuda.nvtx.range("expert_cache:hard_prepare:routing_wait"):
            routing_ready.synchronize()
        with torch.cuda.nvtx.range("expert_cache:hard_prepare:cancel_prefetch"):
            self.runtime.cancel_dummy_prefetch_for(self.binding.layer_id)
        with torch.cuda.nvtx.range("expert_cache:hard_prepare:routing_scan"):
            routed_expert_ids = _ordered_unique(routing_ids)
            if any(
                expert_id < -1 or expert_id >= self.num_experts
                for expert_id in routed_expert_ids
            ):
                raise ValueError(
                    "routing IDs must be within the configured physical expert range"
                )
            expert_ids = tuple(
                expert_id for expert_id in routed_expert_ids if expert_id >= 0
            )
        if not expert_ids:
            return _PreparedExpertDemand((), (), (), ())

        with torch.cuda.nvtx.range("expert_cache:hard_prepare:claim_resident"):
            resident_claims, pending_ids, pending_claims = self.binding.claim_resident(
                expert_ids,
                max_pending_claims=_HARD_LOAD_RUNWAY,
            )
        pending_id_set = set(pending_ids)
        if (
            len({claim.key[1] for claim in pending_claims}) != len(pending_claims)
            or not {claim.key[1] for claim in pending_claims} <= pending_id_set
        ):
            self._release_claims(resident_claims)
            self._release_claims(pending_claims)
            raise RuntimeError("streamed expert hard-runway claims do not match demand")
        return _PreparedExpertDemand(
            expert_ids,
            resident_claims,
            pending_ids,
            pending_claims,
        )

    def _release_prepared_demand(self, demand: _PreparedExpertDemand) -> None:
        self._release_claims(demand.resident_claims)
        self._release_claims(demand.pending_claims)

    def execute(self, prepared_batch: PreparedBatch) -> torch.Tensor:
        """Consume each ready expert and reduce all route outputs once."""
        prepare_stream = self._prepare_stream
        if prepare_stream is None:
            raise RuntimeError("CachedExpertLayer has no active prepared stream")
        execution_succeeded = False
        demand = getattr(prepared_batch, "demand", None)
        next_prefetch_window = getattr(
            prepared_batch,
            "next_prefetch_request",
            None,
        )
        next_prefetch_prepare_attempted = isinstance(
            next_prefetch_window,
            _NativeDummyPrefetchRequest,
        )
        next_prefetch_activate_attempted = False
        next_prefetch_window: (
            _DummyPrefetchWindow
            | _NativeDummyPrefetchRequest
            | _NativeDummyPrefetchWindow
            | None
        )

        def prepare_next_prefetch() -> None:
            nonlocal next_prefetch_prepare_attempted, next_prefetch_window
            if next_prefetch_prepare_attempted:
                return
            next_prefetch_prepare_attempted = True
            prepare = getattr(
                self.runtime,
                "prepare_dummy_prefetch_after",
                None,
            )
            if prepare is not None:
                next_prefetch_window = prepare(self.binding.layer_id)

        def activate_next_prefetch(
            *,
            wait_for_compute: bool = True,
            start_event_recorded: bool = False,
        ) -> None:
            nonlocal next_prefetch_activate_attempted, next_prefetch_window
            if next_prefetch_activate_attempted:
                return
            next_prefetch_activate_attempted = True
            prepare_next_prefetch()
            if isinstance(next_prefetch_window, _NativeDummyPrefetchRequest):
                start_event = None
                if wait_for_compute:
                    if not start_event_recorded:
                        self._prefetch_start.record(prepare_stream)
                    start_event = self._prefetch_start
                self.runtime.activate_dummy_prefetch(
                    next_prefetch_window,
                    start_event,
                )
            else:
                next_prefetch_window = self.runtime.start_dummy_prefetch_after(
                    self.binding.layer_id
                )

        try:
            kernel = self._kernel
            if kernel is None:
                raise RuntimeError(
                    "CachedExpertLayer must be prepared before execution"
                )
            if _stream_identity(torch.cuda.current_stream(self.runtime.device)) != (
                _stream_identity(prepare_stream)
            ):
                raise RuntimeError(
                    "streamed expert prepare and execute must use the same CUDA stream"
                )
            if demand is None:
                demand = self._prepare_expert_demand(
                    prepared_batch.routing_ids,
                    prepared_batch.routing_ready,
                )
            expert_ids = demand.expert_ids
            resident_claims = demand.resident_claims
            pending_ids = demand.pending_ids
            initial_pending_claims = demand.pending_claims
            prepare_next_prefetch()
            if not expert_ids:
                if isinstance(next_prefetch_window, _NativeDummyPrefetchRequest):
                    activate_next_prefetch(wait_for_compute=False)
                result = kernel.finalize_streamed(prepared_batch.kernel_batch)
                activate_next_prefetch()
                execution_succeeded = True
                return result

            pending_id_set = set(pending_ids)
            pending_claims = {claim.key[1]: claim for claim in initial_pending_claims}
            if (
                len(pending_claims) != len(initial_pending_claims)
                or not pending_claims.keys() <= pending_id_set
            ):
                self._release_claims(initial_pending_claims)
                raise RuntimeError(
                    "streamed expert hard-runway claims do not match demand"
                )
            hard_submitted = set(pending_claims)
            remaining_pending = list(pending_ids)

            def activate_native_prefetch_if_hard_submitted() -> None:
                if hard_submitted == pending_id_set and isinstance(
                    next_prefetch_window,
                    _NativeDummyPrefetchRequest,
                ):
                    activate_next_prefetch(wait_for_compute=False)

            def fill_hard_runway() -> None:
                if len(pending_claims) < _HARD_LOAD_RUNWAY:
                    for expert_id in remaining_pending:
                        if expert_id in hard_submitted:
                            continue
                        claim = self.binding.try_claim(expert_id)
                        if claim is None:
                            continue
                        if claim.key[1] != expert_id:
                            claim.release()
                            raise RuntimeError(
                                "streamed expert hard-runway claim does not match "
                                "demand"
                            )
                        pending_claims[expert_id] = claim
                        hard_submitted.add(expert_id)
                        if len(pending_claims) == _HARD_LOAD_RUNWAY:
                            break
                activate_native_prefetch_if_hard_submitted()

            activate_native_prefetch_if_hard_submitted()

            unconsumed_resident = {claim.key[1]: claim for claim in resident_claims}
            try:
                if unconsumed_resident:
                    wave_expert_ids = tuple(unconsumed_resident)
                    wave_claims = tuple(
                        unconsumed_resident.pop(expert_id)
                        for expert_id in wave_expert_ids
                    )
                    leases = self._acquire_wave(wave_claims)
                    try:
                        self._execute_wave(
                            kernel,
                            prepared_batch.kernel_batch,
                            wave_expert_ids,
                            leases,
                        )
                    finally:
                        self._release_wave(leases)
                    self.runtime.cache.record_wave()
                    fill_hard_runway()

                while remaining_pending:
                    fill_hard_runway()
                    claimed_ids = tuple(
                        expert_id
                        for expert_id in remaining_pending
                        if expert_id in pending_claims
                    )
                    if len(claimed_ids) > 1:
                        ready_ids = tuple(
                            expert_id
                            for expert_id in claimed_ids
                            if pending_claims[expert_id].is_ready()
                        )
                    else:
                        ready_ids = claimed_ids
                    wave_expert_ids = ready_ids or claimed_ids[:1]
                    if not wave_expert_ids:
                        expert_id = remaining_pending[0]
                        claim = self.binding.claim(expert_id)
                        if claim.key[1] != expert_id:
                            claim.release()
                            raise RuntimeError(
                                "streamed expert blocking claim does not match demand"
                            )
                        hard_submitted.add(expert_id)
                        pending_claims[expert_id] = claim
                        wave_expert_ids = (expert_id,)
                        activate_native_prefetch_if_hard_submitted()
                    wave_claims = tuple(
                        pending_claims.pop(expert_id) for expert_id in wave_expert_ids
                    )
                    leases = self._acquire_wave(wave_claims)
                    try:
                        self._execute_wave(
                            kernel,
                            prepared_batch.kernel_batch,
                            wave_expert_ids,
                            leases,
                        )
                    finally:
                        self._release_wave(leases)
                    for expert_id in wave_expert_ids:
                        remaining_pending.remove(expert_id)
                    self.runtime.cache.record_wave()
                    fill_hard_runway()
                if len(hard_submitted) != len(pending_ids):
                    raise RuntimeError(
                        "streamed expert hard runway ended before all demand "
                        "was submitted"
                    )
            finally:
                self._release_claims(tuple(unconsumed_resident.values()))
                self._release_claims(tuple(pending_claims.values()))

            result = kernel.finalize_streamed(prepared_batch.kernel_batch)
            activate_next_prefetch()
            execution_succeeded = True
            return result
        finally:
            try:
                if demand is not None:
                    self._release_prepared_demand(demand)
                if not execution_succeeded and next_prefetch_window is not None:
                    self.runtime.cancel_dummy_prefetch_window(next_prefetch_window)
            finally:
                try:
                    try:
                        self.runtime.finish_forward(prepare_stream)
                    except Exception as error:
                        if execution_succeeded and next_prefetch_window is not None:
                            try:
                                self.runtime.cancel_dummy_prefetch_window(
                                    next_prefetch_window
                                )
                            except Exception as cleanup_error:
                                error.add_note(
                                    "Activated prefetch cleanup also failed: "
                                    f"{cleanup_error}"
                                )
                        raise
                finally:
                    self._prepare_stream = None
                    self.runtime._forward_lock.release()

    def _acquire_wave(
        self,
        claims: Sequence[ExpertClaim],
    ) -> tuple[ExpertLease, ...]:
        leases: list[ExpertLease] = []
        try:
            for claim in claims:
                leases.append(claim.acquire(self._prepare_stream))
        except Exception:
            try:
                self._release_wave(leases)
            finally:
                self._release_claims(claims)
            raise
        return tuple(leases)

    @staticmethod
    def _release_claims(claims: Sequence[ExpertClaim]) -> None:
        for claim in claims:
            claim.release()

    def _release_wave(self, leases: Sequence[ExpertLease]) -> None:
        if not leases:
            return
        stream = self._prepare_stream
        if stream is None:
            raise RuntimeError("cannot release expert leases without a compute stream")
        try:
            last_use = torch.cuda.Event()
            last_use.record(stream)
        except Exception as exc:
            self.runtime._stream_guard_failed = True
            error = ExpertCacheError(
                "failed to record streamed expert-cache wave completion"
            )
            self.runtime.cache.fail_closed(error, leases=leases)
            raise error from exc
        for lease in leases:
            lease.release(last_use_event=last_use)

    def _execute_wave(
        self,
        kernel: FusedMoEKernel,
        batch: PreparedStreamedMoEBatch,
        expert_ids: Sequence[int],
        leases: Sequence[ExpertLease],
    ) -> None:
        self._stage_wave_map(
            expert_ids,
            tuple(lease.slot_index for lease in leases),
        )
        self._invoke_wave(kernel, batch)

    def _stage_wave_map(
        self,
        expert_ids: Sequence[int],
        slot_indices: Sequence[int],
    ) -> None:
        ops.moe_update_expert_map(
            self.wave_expert_map,
            list(expert_ids),
            list(slot_indices),
        )

    def _invoke_wave(
        self,
        kernel: FusedMoEKernel,
        batch: PreparedStreamedMoEBatch,
    ) -> None:
        self._kernel = kernel
        self._active_batch = batch
        try:
            _invoke_cudagraph_with_initial_replay(
                self._wave_runner,
                batch.output,
                batch.route_output,
                batch.a1q,
                batch.topk_weights,
                batch.topk_ids,
                batch.workspace13,
                batch.workspace2,
                self.wave_expert_map,
                *self._alignment_outputs,
                *self._graph_closure_tensors(),
            )
        finally:
            self._active_batch = None

    def _run_active_wave(self, *graph_tensors: torch.Tensor) -> torch.Tensor:
        del graph_tensors
        kernel = self._kernel
        batch = self._active_batch
        if kernel is None or batch is None:
            raise RuntimeError("no streamed expert wave is active")
        kernel.execute_streamed_wave(
            batch=batch,
            w1=self.runtime.w13,
            w2=self.runtime.w2,
            expert_map=self.wave_expert_map,
            alignment_outputs=self._alignment_outputs,
        )
        return batch.route_output


@contextmanager
def suspend_streamed_expert_cache_prefetch() -> Generator[None, None, None]:
    """Drain and pause speculative copies while CUDA graphs are captured."""
    paused_runtimes: list[_StreamedExpertCacheRuntime] = []
    try:
        for runtime in tuple(_RUNTIMES.values()):
            runtime.pause_dummy_prefetch_and_drain()
            paused_runtimes.append(runtime)
        yield
    finally:
        for runtime in reversed(paused_runtimes):
            runtime.resume_dummy_prefetch()


def bind_streamed_expert_layer(
    layer: RoutedExperts,
    kernel: FusedMoEKernel,
    *,
    streamed_weights: Any | None = None,
) -> CachedExpertLayer:
    """Move a loaded routed-expert layer into the model's unified arena."""
    vllm_config = get_current_vllm_config()
    if layer.moe_config.has_bias:
        raise ValueError("streamed expert caching does not support expert bias")
    if streamed_weights is None:
        host_w13 = layer.w13_weight.data
        host_w2 = layer.w2_weight.data
        format_class = _FORMAT_CLASS
        runtime_specs: Mapping[str, Any] | None = None
        activation_dtype = host_w13.dtype
        hidden_size = host_w13.shape[-1]
        host_bundles = {
            expert_id: ExpertWeightBundle(
                format_class,
                {
                    "w13_weight": host_w13[expert_id],
                    "w2_weight": host_w2[expert_id],
                },
            )
            for expert_id in range(layer.local_num_experts)
        }
    else:
        format_class = streamed_weights.format_class
        runtime_specs = streamed_weights.runtime_specs
        activation_dtype = streamed_weights.activation_dtype
        hidden_size = streamed_weights.hidden_size
        host_bundles = dict(streamed_weights.host_bundles)
        if set(host_bundles) != set(range(layer.local_num_experts)):
            raise ValueError(
                "streamed expert bundles must cover every local physical expert"
            )
        first_bundle = host_bundles[0]
        host_w13 = first_bundle.tensors["w13_weight"].unsqueeze(0)
        host_w2 = first_bundle.tensors["w2_weight"].unsqueeze(0)

    if not all(bundle.is_pinned for bundle in host_bundles.values()):
        raise ValueError(
            "streamed expert checkpoint weights must be loaded into pinned CPU memory"
        )
    if any(bundle.format_class != format_class for bundle in host_bundles.values()):
        raise ValueError("streamed expert bundles have inconsistent formats")

    device = torch.device(layer.moe_config.device)
    if device.index is None:
        device = torch.device("cuda", torch.accelerator.current_device_index())
    load_token = get_streamed_expert_cache_load_token()
    runtime_key = id(load_token)
    runtime = _RUNTIMES.get(runtime_key)
    if runtime is None:
        runtime = _StreamedExpertCacheRuntime(
            vllm_config,
            host_w13,
            host_w2,
            device,
            load_token,
            format_class=format_class,
            runtime_specs=runtime_specs,
            activation_dtype=activation_dtype,
            hidden_size=hidden_size,
        )
        _RUNTIMES[runtime_key] = runtime
    elif not runtime.belongs_to(vllm_config, load_token):
        raise RuntimeError("stale streamed expert-cache runtime configuration")
    elif runtime.format_class != format_class:
        raise RuntimeError(
            "streamed expert-cache layers cannot mix runtime weight formats"
        )

    layer_id = extract_layer_index(layer.layer_name)
    binding = runtime.bind(layer_id, host_bundles)
    for name, tensor in runtime.arena_tensors.items():
        if not hasattr(layer, name):
            raise RuntimeError(
                f"streamed expert layer is missing runtime tensor {name!r}"
            )
        replace_parameter(layer, name, tensor)
        getattr(layer, name).weight_loader = _reject_weight_reload

    cached_layer = CachedExpertLayer(
        runtime,
        binding,
        kernel=kernel,
        activation=layer.activation,
        num_experts=layer.global_num_experts,
        max_num_tokens=layer.moe_config.max_num_tokens,
        top_k=layer.top_k,
    )
    return cached_layer


def _moe_layer_ids(vllm_config: VllmConfig) -> tuple[int, ...]:
    config = vllm_config.model_config.hf_text_config
    mlp_only_layers = set(getattr(config, "mlp_only_layers", None) or ())
    sparse_step = getattr(config, "decoder_sparse_step", 1)
    num_experts = getattr(
        config,
        "num_experts",
        getattr(config, "n_routed_experts", 0),
    )
    return tuple(
        layer_id
        for layer_id in range(config.num_hidden_layers)
        if layer_id not in mlp_only_layers
        and num_experts > 0
        and (layer_id + 1) % sparse_step == 0
    )


def _runtime_tensor_spec(spec: Any) -> tuple[torch.Size, torch.dtype]:
    if isinstance(spec, tuple) and len(spec) == 2:
        shape, dtype = spec
    elif isinstance(spec, Mapping):
        shape, dtype = spec["shape"], spec["dtype"]
    else:
        shape, dtype = spec.shape, spec.dtype
    if not isinstance(dtype, torch.dtype):
        raise TypeError("streamed expert runtime tensor dtype must be torch.dtype")
    return torch.Size(shape), dtype


def _ordered_unique(expert_ids: torch.Tensor) -> tuple[int, ...]:
    return tuple(dict.fromkeys(expert_ids.view(-1).tolist()))


def _stream_identity(stream: torch.cuda.Stream) -> tuple[torch.device, int]:
    return torch.device(stream.device), stream.cuda_stream


def _invoke_cudagraph_with_initial_replay(
    runner: Callable[..., Any],
    *args: torch.Tensor,
) -> Any:
    """Run a newly captured side-effect graph once before returning."""
    if not isinstance(runner, CUDAGraphWrapper):
        return runner(*args)
    captured_before = sum(
        entry.cudagraph is not None
        for entry in runner.concrete_cudagraph_entries.values()
    )
    output = runner(*args)
    captured_after = sum(
        entry.cudagraph is not None
        for entry in runner.concrete_cudagraph_entries.values()
    )
    if (
        captured_after > captured_before
        and not runner.cudagraph_options.use_direct_capture
    ):
        output = runner(*args)
    return output


def _reject_weight_reload(*args: object, **kwargs: object) -> None:
    del args, kwargs
    raise RuntimeError("streamed expert caching does not support hot weight updates")
