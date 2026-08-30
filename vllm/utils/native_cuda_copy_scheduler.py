# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

try:
    from vllm import cuda_copy_scheduler_C as _native
except ImportError as exc:
    _native = None
    _native_import_error = exc
else:
    _native_import_error = None


class CudaMemcpyKind(IntEnum):
    """Values accepted by ``cudaMemcpyAsync``."""

    HOST_TO_HOST = 0
    HOST_TO_DEVICE = 1
    DEVICE_TO_HOST = 2
    DEVICE_TO_DEVICE = 3
    DEFAULT = 4


class CopyJobStatus(IntEnum):
    """Host-side admission state of a copy job."""

    QUEUED = 0
    ISSUING = 1
    ISSUED = 2
    COMPLETED = 3
    CANCELED = 4
    FAILED = 5


@dataclass(frozen=True)
class CopySegment:
    """One contiguous asynchronous memory copy."""

    src: int
    dst: int
    nbytes: int
    kind: CudaMemcpyKind = CudaMemcpyKind.DEFAULT


@dataclass(frozen=True)
class CopyJob:
    """A group of segments committed atomically by the copy scheduler."""

    cookie: int
    segments: tuple[CopySegment, ...]
    wait_event: int = 0
    done_event: int = 0
    label: str = ""


@dataclass(frozen=True)
class CopyWindowSnapshot:
    """Host-side state captured for one speculative window."""

    handle: int
    queued: tuple[int, ...]
    issuing: tuple[int, ...]
    issued: tuple[int, ...]
    completed: tuple[int, ...]
    canceled: tuple[int, ...]
    failed: tuple[int, ...]
    error: str | None

    def status(self, cookie: int) -> CopyJobStatus:
        """Return the status of a cookie present in this snapshot."""
        if cookie in self.queued:
            return CopyJobStatus.QUEUED
        if cookie in self.issuing:
            return CopyJobStatus.ISSUING
        if cookie in self.completed:
            return CopyJobStatus.COMPLETED
        if cookie in self.canceled:
            return CopyJobStatus.CANCELED
        if cookie in self.failed:
            return CopyJobStatus.FAILED
        if cookie in self.issued:
            return CopyJobStatus.ISSUED
        raise KeyError(cookie)

    @classmethod
    def _from_native(cls, value: dict[str, Any]) -> CopyWindowSnapshot:
        return cls(
            handle=value["handle"],
            queued=tuple(value["queued"]),
            issuing=tuple(value["issuing"]),
            issued=tuple(value["issued"]),
            completed=tuple(value["completed"]),
            canceled=tuple(value["canceled"]),
            failed=tuple(value["failed"]),
            error=value["error"],
        )


def native_copy_scheduler_available() -> bool:
    """Return whether the optional native extension can be imported."""
    return _native is not None


class NativeCudaCopyScheduler:
    """Issue and reap generic asynchronous copies on native worker threads.

    Streams, events, and memory are externally owned. Their lifetimes must
    extend through the corresponding job's completion event.
    ``poll_interval_us`` remains accepted for compatibility; completion is
    event-driven.
    """

    def __init__(
        self,
        stream_handle: int,
        *,
        device: int | None = None,
        max_inflight: int = 4,
        poll_interval_us: int = 50,
    ) -> None:
        if _native is None:
            raise RuntimeError(
                "native CUDA copy scheduler extension is unavailable"
            ) from _native_import_error
        if device is None:
            import torch

            device = torch.cuda.current_device()
        if stream_handle <= 0:
            raise ValueError("stream_handle must be positive")
        if max_inflight <= 0:
            raise ValueError("max_inflight must be positive")
        if poll_interval_us <= 0:
            raise ValueError("poll_interval_us must be positive")
        self._native = _native
        self._handle = _native.create(
            device,
            stream_handle,
            max_inflight,
            poll_interval_us,
        )
        self._closed = False

    def prepare_window(self, jobs: Sequence[CopyJob]) -> int:
        """Register immutable jobs without making them eligible to issue."""
        self._check_open()
        return self._native.prepare_window(
            self._handle,
            self._encode_jobs(jobs),
        )

    def activate_window(
        self,
        handle: int,
        *,
        start_event: int = 0,
        stop_event: int = 0,
    ) -> None:
        """Make a prepared window eligible for speculative admission."""
        self._check_open()
        self._native.activate_window(
            self._handle,
            handle,
            start_event,
            stop_event,
        )

    def submit_window(
        self,
        jobs: Sequence[CopyJob],
        *,
        start_event: int = 0,
        stop_event: int = 0,
    ) -> int:
        """Prepare and immediately activate a speculative copy window."""
        self._check_open()
        return self._native.submit_window(
            self._handle,
            self._encode_jobs(jobs),
            start_event,
            stop_event,
        )

    def set_stop_event(self, handle: int, event: int) -> None:
        """Stop admitting jobs after an external event becomes ready."""
        self._check_open()
        self._native.set_stop_event(self._handle, handle, event)

    def submit_urgent(self, job: CopyJob) -> None:
        """Commit an urgent job before any further speculative admission."""
        self._check_open()
        self._native.submit_urgent(self._handle, self._encode_job(job))

    def enqueue_urgent(self, job: CopyJob) -> None:
        """Queue an urgent job and return before it is committed."""
        self._check_open()
        self._native.enqueue_urgent(self._handle, self._encode_job(job))

    def query_urgent_issued(self, cookie: int) -> bool:
        """Consume a completed issue ticket without blocking."""
        self._check_open()
        return self._native.query_urgent_issued(self._handle, cookie)

    def wait_urgent_issued(self, cookie: int) -> None:
        """Consume an issue ticket after its CUDA event has been recorded."""
        self._check_open()
        self._native.wait_urgent_issued(self._handle, cookie)

    def cancel_and_snapshot(self, handle: int) -> CopyWindowSnapshot:
        """Cancel unissued work and return an admission-fenced snapshot."""
        self._check_open()
        value = self._native.cancel_and_snapshot(self._handle, handle)
        return CopyWindowSnapshot._from_native(value)

    def snapshot(self, handle: int) -> CopyWindowSnapshot:
        """Read window state without canceling it or waiting for DMA."""
        self._check_open()
        return CopyWindowSnapshot._from_native(
            self._native.snapshot(self._handle, handle)
        )

    def pending_count(self, handle: int) -> int:
        """Return the number of jobs not yet admitted to the CUDA stream."""
        self._check_open()
        return self._native.pending_count(self._handle, handle)

    def release_window(self, handle: int) -> None:
        """Release bookkeeping after a window has stopped admission."""
        self._check_open()
        self._native.release_window(self._handle, handle)

    def pause_and_drain(self) -> tuple[CopyWindowSnapshot, ...]:
        """Pause all admission and drain previously committed jobs."""
        self._check_open()
        return tuple(
            CopyWindowSnapshot._from_native(value)
            for value in self._native.pause_and_drain(self._handle)
        )

    def resume(self) -> None:
        """Release one pause nesting level."""
        self._check_open()
        self._native.resume(self._handle)

    def stats(self) -> dict[str, int]:
        """Return native scheduler counters."""
        self._check_open()
        return self._native.stats(self._handle)

    def wait_event_on_stream(self, event: int, stream: int) -> None:
        """Enqueue a wait for an externally owned event on a stream."""
        self._check_open()
        self._native.wait_event_on_stream(event, stream)

    def close(self) -> None:
        """Cancel queued work, drain committed jobs, and stop the worker."""
        if self._closed:
            return
        self._native.close(self._handle)
        self._closed = True

    def __enter__(self) -> NativeCudaCopyScheduler:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("native CUDA copy scheduler is closed")

    @staticmethod
    def _encode_jobs(jobs: Sequence[CopyJob]) -> tuple[tuple[Any, ...], ...]:
        return tuple(NativeCudaCopyScheduler._encode_job(job) for job in jobs)

    @staticmethod
    def _encode_job(job: CopyJob) -> tuple[Any, ...]:
        if job.done_event <= 0:
            raise ValueError("copy job done_event must be positive")
        if not job.segments:
            raise ValueError("copy job must contain at least one segment")
        segments = tuple(
            (segment.src, segment.dst, segment.nbytes, int(segment.kind))
            for segment in job.segments
        )
        return (
            job.cookie,
            segments,
            job.wait_event,
            job.done_event,
            job.label,
        )
