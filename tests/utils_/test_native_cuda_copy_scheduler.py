# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import threading
import time
from typing import Any

import pytest
import torch

import vllm.utils.native_cuda_copy_scheduler as scheduler_module
from vllm.utils.native_cuda_copy_scheduler import (
    CopyJob,
    CopyJobStatus,
    CopySegment,
    CudaMemcpyKind,
    NativeCudaCopyScheduler,
)


class _FakeNativeScheduler:
    def __init__(self) -> None:
        self.handle = object()
        self.prepared_jobs: tuple[Any, ...] | None = None
        self.activated: tuple[int, int, int] | None = None
        self.enqueued_job: tuple[Any, ...] | None = None
        self.queried_cookie: int | None = None
        self.waited_cookie: int | None = None
        self.urgent_issued = False
        self.released: int | None = None
        self.closed = False

    def create(
        self,
        device: int,
        stream: int,
        max_inflight: int,
        poll_interval_us: int,
    ) -> object:
        assert (device, stream, max_inflight, poll_interval_us) == (2, 7, 3, 11)
        return self.handle

    def prepare_window(
        self,
        handle: object,
        jobs: tuple[Any, ...],
    ) -> int:
        assert handle is self.handle
        self.prepared_jobs = jobs
        return 13

    def activate_window(
        self,
        handle: object,
        window: int,
        start_event: int,
        stop_event: int,
    ) -> None:
        assert handle is self.handle
        self.activated = (window, start_event, stop_event)

    def snapshot(self, handle: object, window: int) -> dict[str, Any]:
        assert handle is self.handle
        return {
            "handle": window,
            "queued": [101],
            "issuing": [],
            "issued": [100],
            "completed": [],
            "canceled": [],
            "failed": [],
            "error": None,
        }

    def enqueue_urgent(self, handle: object, job: tuple[Any, ...]) -> None:
        assert handle is self.handle
        self.enqueued_job = job

    def query_urgent_issued(self, handle: object, cookie: int) -> bool:
        assert handle is self.handle
        self.queried_cookie = cookie
        return self.urgent_issued

    def wait_urgent_issued(self, handle: object, cookie: int) -> None:
        assert handle is self.handle
        self.waited_cookie = cookie

    def release_window(self, handle: object, window: int) -> None:
        assert handle is self.handle
        self.released = window

    def close(self, handle: object) -> None:
        assert handle is self.handle
        self.closed = True


def test_prepared_window_encodes_generic_copy_descriptors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = _FakeNativeScheduler()
    monkeypatch.setattr(scheduler_module, "_native", native)
    scheduler = NativeCudaCopyScheduler(
        7,
        device=2,
        max_inflight=3,
        poll_interval_us=11,
    )
    job = CopyJob(
        cookie=100,
        segments=(
            CopySegment(10, 20, 30),
            CopySegment(40, 50, 60, CudaMemcpyKind.HOST_TO_DEVICE),
        ),
        wait_event=70,
        done_event=80,
        label="expert_cache:prefetch:1:2",
    )

    handle = scheduler.prepare_window((job,))
    scheduler.activate_window(handle, start_event=85, stop_event=90)

    assert native.prepared_jobs == (
        (
            100,
            ((10, 20, 30, 4), (40, 50, 60, 1)),
            70,
            80,
            "expert_cache:prefetch:1:2",
        ),
    )
    assert native.activated == (13, 85, 90)
    snapshot = scheduler.snapshot(handle)
    assert snapshot.status(100) is CopyJobStatus.ISSUED
    assert snapshot.status(101) is CopyJobStatus.QUEUED
    scheduler.release_window(handle)
    assert native.released == 13
    scheduler.close()
    assert native.closed


def test_copy_job_requires_completion_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = _FakeNativeScheduler()
    monkeypatch.setattr(scheduler_module, "_native", native)
    scheduler = NativeCudaCopyScheduler(
        7,
        device=2,
        max_inflight=3,
        poll_interval_us=11,
    )
    job = CopyJob(
        cookie=1,
        segments=(CopySegment(10, 20, 30),),
    )

    with pytest.raises(ValueError, match="done_event must be positive"):
        scheduler.prepare_window((job,))


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not scheduler_module.native_copy_scheduler_available(),
    reason="requires the native CUDA copy scheduler",
)
def test_prepared_window_is_inert_until_activation() -> None:
    device = torch.cuda.current_device()
    copy_stream = torch.cuda.Stream(device=device)
    source = torch.arange(256, dtype=torch.int32).pin_memory()
    destination = torch.full_like(source, -1, device=f"cuda:{device}")
    ready = torch.cuda.Event()
    ready.record(copy_stream)
    copy_stream.synchronize()
    scheduler = NativeCudaCopyScheduler(
        copy_stream.cuda_stream,
        device=device,
        max_inflight=1,
        poll_interval_us=10,
    )
    handle = scheduler.prepare_window(
        (
            CopyJob(
                cookie=0,
                segments=(
                    CopySegment(
                        source.data_ptr(),
                        destination.data_ptr(),
                        source.nbytes,
                        CudaMemcpyKind.HOST_TO_DEVICE,
                    ),
                ),
                done_event=ready.cuda_event,
            ),
        )
    )

    try:
        time.sleep(0.02)
        copy_stream.synchronize()
        snapshot = scheduler.snapshot(handle)
        assert snapshot.queued == (0,)
        assert not snapshot.issued
        assert torch.equal(destination.cpu(), torch.full_like(source, -1))

        scheduler.activate_window(handle)
        with pytest.raises(RuntimeError, match="already active"):
            scheduler.activate_window(handle)
        ready.synchronize()
        assert torch.equal(destination.cpu(), source)
        scheduler.cancel_and_snapshot(handle)
        scheduler.release_window(handle)
    finally:
        scheduler.close()


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not scheduler_module.native_copy_scheduler_available(),
    reason="requires the native CUDA copy scheduler",
)
def test_window_start_event_delays_first_copy() -> None:
    device = torch.cuda.current_device()
    copy_stream = torch.cuda.Stream(device=device)
    gate_stream = torch.cuda.Stream(device=device)
    source = torch.arange(256, dtype=torch.int32).pin_memory()
    destination = torch.full_like(source, -1, device=f"cuda:{device}")
    start = torch.cuda.Event()
    with torch.cuda.stream(gate_stream):
        torch.cuda._sleep(300_000_000)
        start.record(gate_stream)
    ready = torch.cuda.Event()
    ready.record(copy_stream)
    copy_stream.synchronize()
    scheduler = NativeCudaCopyScheduler(
        copy_stream.cuda_stream,
        device=device,
        max_inflight=1,
        poll_interval_us=10,
    )
    handle = scheduler.prepare_window(
        (
            CopyJob(
                cookie=0,
                segments=(
                    CopySegment(
                        source.data_ptr(),
                        destination.data_ptr(),
                        source.nbytes,
                        CudaMemcpyKind.HOST_TO_DEVICE,
                    ),
                ),
                done_event=ready.cuda_event,
            ),
        )
    )

    try:
        scheduler.activate_window(handle, start_event=start.cuda_event)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            snapshot = scheduler.snapshot(handle)
            if snapshot.issued == (0,):
                break
            time.sleep(0.001)
        else:
            pytest.fail(f"start-gated copy was not issued: {snapshot}")

        assert not start.query()
        assert not ready.query()
        ready.synchronize()
        assert torch.equal(destination.cpu(), source)
        scheduler.cancel_and_snapshot(handle)
        scheduler.release_window(handle)
    finally:
        scheduler.close()


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not scheduler_module.native_copy_scheduler_available(),
    reason="requires the native CUDA copy scheduler",
)
def test_completion_refills_single_speculative_runway() -> None:
    device = torch.cuda.current_device()
    copy_stream = torch.cuda.Stream(device=device)
    delay_stream = torch.cuda.Stream(device=device)
    source_storage = torch.arange(512, dtype=torch.int32).reshape(2, 256)
    source_storage = source_storage.pin_memory()
    destination_storage = torch.full_like(
        source_storage,
        -1,
        device=f"cuda:{device}",
    )
    gate = torch.cuda.Event()
    with torch.cuda.stream(delay_stream):
        torch.cuda._sleep(300_000_000)
        gate.record(delay_stream)
    ready_events = [torch.cuda.Event() for _ in range(2)]
    with torch.cuda.stream(copy_stream):
        for ready in ready_events:
            ready.record(copy_stream)
    copy_stream.synchronize()

    scheduler = NativeCudaCopyScheduler(
        copy_stream.cuda_stream,
        device=device,
        max_inflight=1,
        poll_interval_us=10,
    )
    jobs = tuple(
        CopyJob(
            cookie=index,
            segments=(
                CopySegment(
                    source.data_ptr(),
                    destination.data_ptr(),
                    source.nbytes,
                    CudaMemcpyKind.HOST_TO_DEVICE,
                ),
            ),
            wait_event=gate.cuda_event if index == 0 else 0,
            done_event=ready.cuda_event,
        )
        for index, (source, destination, ready) in enumerate(
            zip(source_storage, destination_storage, ready_events)
        )
    )
    handle = scheduler.submit_window(jobs)

    try:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            snapshot = scheduler.snapshot(handle)
            if snapshot.issued == (0,):
                break
            time.sleep(0.001)
        else:
            pytest.fail(f"first speculative copy was not issued: {snapshot}")
        assert snapshot.queued == (1,)
        assert not gate.query()

        ready_events[1].synchronize()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            snapshot = scheduler.snapshot(handle)
            if snapshot.completed == (0, 1):
                break
            time.sleep(0.001)
        else:
            pytest.fail(f"speculative copies were not reaped: {snapshot}")
        snapshot = scheduler.cancel_and_snapshot(handle)
        assert snapshot.issued == (0, 1)
        assert snapshot.completed == (0, 1)
        assert torch.equal(destination_storage.cpu(), source_storage)
        scheduler.release_window(handle)
    finally:
        scheduler.close()


def test_urgent_issue_ticket_is_separate_from_enqueue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = _FakeNativeScheduler()
    monkeypatch.setattr(scheduler_module, "_native", native)
    scheduler = NativeCudaCopyScheduler(
        7,
        device=2,
        max_inflight=3,
        poll_interval_us=11,
    )
    job = CopyJob(
        cookie=17,
        segments=(CopySegment(10, 20, 30),),
        wait_event=40,
        done_event=50,
        label="urgent:17",
    )

    scheduler.enqueue_urgent(job)

    assert native.enqueued_job == (
        17,
        ((10, 20, 30, 4),),
        40,
        50,
        "urgent:17",
    )
    assert not scheduler.query_urgent_issued(17)
    assert native.queried_cookie == 17
    native.urgent_issued = True
    assert scheduler.query_urgent_issued(17)
    scheduler.wait_urgent_issued(23)
    assert native.waited_cookie == 23
    scheduler.close()


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not scheduler_module.native_copy_scheduler_available(),
    reason="requires the native CUDA copy scheduler",
)
def test_pause_drains_admitted_urgent_and_rejects_new_urgent() -> None:
    device = torch.cuda.current_device()
    copy_stream = torch.cuda.Stream(device=device)
    delay_stream = torch.cuda.Stream(device=device)
    source = torch.arange(256, dtype=torch.uint8, pin_memory=True)
    destination = torch.zeros_like(source, device=f"cuda:{device}")
    gate = torch.cuda.Event()
    with torch.cuda.stream(delay_stream):
        torch.cuda._sleep(300_000_000)
        gate.record(delay_stream)
    ready = torch.cuda.Event()
    ready.record(copy_stream)
    copy_stream.synchronize()

    scheduler = NativeCudaCopyScheduler(
        copy_stream.cuda_stream,
        device=device,
        max_inflight=1,
        poll_interval_us=10,
    )
    try:
        job = CopyJob(
            cookie=1,
            segments=(
                CopySegment(
                    source.data_ptr(),
                    destination.data_ptr(),
                    source.nbytes,
                    CudaMemcpyKind.HOST_TO_DEVICE,
                ),
            ),
            wait_event=gate.cuda_event,
            done_event=ready.cuda_event,
        )
        scheduler.enqueue_urgent(job)

        pause_done = threading.Event()
        pause_result: list[tuple[Any, ...]] = []
        pause_thread = threading.Thread(
            target=lambda: (
                pause_result.append(scheduler.pause_and_drain()),
                pause_done.set(),
            )
        )
        pause_thread.start()

        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                handle = scheduler.prepare_window(())
            except RuntimeError as error:
                assert "paused" in str(error)
                break
            else:
                scheduler.release_window(handle)
        else:
            pytest.fail("scheduler did not enter the paused state")

        with pytest.raises(RuntimeError, match="paused"):
            scheduler.enqueue_urgent(
                CopyJob(
                    cookie=2,
                    segments=job.segments,
                    done_event=ready.cuda_event,
                )
            )
        assert not pause_done.is_set()

        assert pause_done.wait(timeout=5)
        pause_thread.join(timeout=2)
        assert not pause_thread.is_alive()
        scheduler.wait_urgent_issued(1)

        assert pause_result == [()]
        assert torch.equal(destination.cpu(), source)
        assert scheduler.stats()["issued_jobs"] == 1
        scheduler.resume()

        scheduler.submit_urgent(
            CopyJob(
                cookie=2,
                segments=job.segments,
                done_event=ready.cuda_event,
            )
        )
    finally:
        scheduler.close()


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not scheduler_module.native_copy_scheduler_available(),
    reason="requires the native CUDA copy scheduler",
)
def test_queued_urgent_copies_arm_independently() -> None:
    device = torch.cuda.current_device()
    copy_stream = torch.cuda.Stream(device=device)
    compute_stream = torch.cuda.Stream(device=device)
    source_storage = torch.empty((8, 1 << 20), dtype=torch.uint8)
    for index, source in enumerate(source_storage):
        source.fill_(index)
    source_storage = source_storage.pin_memory()
    destination_storage = torch.zeros_like(
        source_storage,
        device=f"cuda:{device}",
    )
    sources = tuple(source_storage.unbind())
    destinations = tuple(destination_storage.unbind())
    ready_events = [torch.cuda.Event() for _ in sources]
    with torch.cuda.stream(copy_stream):
        for event in ready_events:
            event.record(copy_stream)

    scheduler = NativeCudaCopyScheduler(
        copy_stream.cuda_stream,
        device=device,
        max_inflight=1,
        poll_interval_us=10,
    )
    try:
        for index, (source, destination, ready) in enumerate(
            zip(sources, destinations, ready_events)
        ):
            scheduler.enqueue_urgent(
                CopyJob(
                    cookie=index,
                    segments=(
                        CopySegment(
                            source.data_ptr(),
                            destination.data_ptr(),
                            source.nbytes,
                            CudaMemcpyKind.HOST_TO_DEVICE,
                        ),
                    ),
                    done_event=ready.cuda_event,
                )
            )

        scheduler.wait_urgent_issued(0)
        compute_stream.wait_event(ready_events[0])
        compute_stream.synchronize()

        assert torch.equal(destinations[0].cpu(), sources[0])
        for index in range(1, len(sources)):
            scheduler.wait_urgent_issued(index)
        copy_stream.synchronize()
        assert all(
            torch.equal(destination.cpu(), source)
            for source, destination in zip(sources, destinations)
        )
    finally:
        scheduler.close()


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not scheduler_module.native_copy_scheduler_available(),
    reason="requires the native CUDA copy scheduler",
)
def test_reused_external_event_does_not_merge_job_completion() -> None:
    device = torch.cuda.current_device()
    copy_stream = torch.cuda.Stream(device=device)
    delay_stream = torch.cuda.Stream(device=device)
    source_storage = torch.arange(2 * 256, dtype=torch.int32).reshape(2, 256)
    source_storage = source_storage.pin_memory()
    destination_storage = torch.full_like(
        source_storage,
        -1,
        device=f"cuda:{device}",
    )
    first_gate = torch.cuda.Event()
    second_gate = torch.cuda.Event()
    with torch.cuda.stream(delay_stream):
        torch.cuda._sleep(20_000_000)
        first_gate.record(delay_stream)
        torch.cuda._sleep(300_000_000)
        second_gate.record(delay_stream)
    shared_ready = torch.cuda.Event()
    shared_ready.record(copy_stream)
    copy_stream.synchronize()

    scheduler = NativeCudaCopyScheduler(
        copy_stream.cuda_stream,
        device=device,
        max_inflight=2,
        poll_interval_us=10,
    )
    try:
        for cookie, (source, destination, gate) in enumerate(
            zip(
                source_storage,
                destination_storage,
                (first_gate, second_gate),
            )
        ):
            scheduler.enqueue_urgent(
                CopyJob(
                    cookie=cookie,
                    segments=(
                        CopySegment(
                            source.data_ptr(),
                            destination.data_ptr(),
                            source.nbytes,
                            CudaMemcpyKind.HOST_TO_DEVICE,
                        ),
                    ),
                    wait_event=gate.cuda_event,
                    done_event=shared_ready.cuda_event,
                )
            )
        scheduler.wait_urgent_issued(0)
        scheduler.wait_urgent_issued(1)

        first_gate.synchronize()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            stats = scheduler.stats()
            if stats["completed_jobs"] != 0 or second_gate.query():
                break
            time.sleep(0.001)

        assert stats["completed_jobs"] == 1
        assert not second_gate.query()

        copy_stream.synchronize()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            stats = scheduler.stats()
            if stats["completed_jobs"] == 2:
                break
            time.sleep(0.001)
        assert stats["completed_jobs"] == 2
        assert torch.equal(destination_storage.cpu(), source_storage)
    finally:
        scheduler.close()


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not scheduler_module.native_copy_scheduler_available(),
    reason="requires the native CUDA copy scheduler",
)
def test_concurrent_close_joins_native_worker_once() -> None:
    device = torch.cuda.current_device()
    copy_stream = torch.cuda.Stream(device=device)
    delay_stream = torch.cuda.Stream(device=device)
    source = torch.arange(256, dtype=torch.int32).pin_memory()
    destination = torch.zeros_like(source, device=f"cuda:{device}")
    gate = torch.cuda.Event()
    with torch.cuda.stream(delay_stream):
        torch.cuda._sleep(100_000_000)
        gate.record(delay_stream)
    ready = torch.cuda.Event()
    ready.record(copy_stream)
    copy_stream.synchronize()

    scheduler = NativeCudaCopyScheduler(
        copy_stream.cuda_stream,
        device=device,
        max_inflight=1,
        poll_interval_us=10,
    )
    scheduler.enqueue_urgent(
        CopyJob(
            cookie=0,
            segments=(
                CopySegment(
                    source.data_ptr(),
                    destination.data_ptr(),
                    source.nbytes,
                    CudaMemcpyKind.HOST_TO_DEVICE,
                ),
            ),
            wait_event=gate.cuda_event,
            done_event=ready.cuda_event,
        )
    )
    scheduler.wait_urgent_issued(0)

    barrier = threading.Barrier(3)
    errors: list[BaseException] = []

    def close() -> None:
        barrier.wait()
        try:
            scheduler.close()
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=close) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=5)

    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    assert torch.equal(destination.cpu(), source)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not scheduler_module.native_copy_scheduler_available(),
    reason="requires the native CUDA copy scheduler",
)
def test_close_waits_for_job_that_is_still_issuing() -> None:
    device = torch.cuda.current_device()
    copy_stream = torch.cuda.Stream(device=device)
    delay_stream = torch.cuda.Stream(device=device)
    source = torch.arange(256, dtype=torch.uint8).pin_memory()
    destination = torch.zeros_like(source, device=f"cuda:{device}")
    gate = torch.cuda.Event()
    with torch.cuda.stream(delay_stream):
        torch.cuda._sleep(100_000_000)
        gate.record(delay_stream)
    ready = torch.cuda.Event()
    ready.record(copy_stream)
    copy_stream.synchronize()

    scheduler = NativeCudaCopyScheduler(
        copy_stream.cuda_stream,
        device=device,
        max_inflight=1,
        poll_interval_us=10,
    )
    segment = CopySegment(
        source.data_ptr(),
        destination.data_ptr(),
        source.nbytes,
        CudaMemcpyKind.HOST_TO_DEVICE,
    )
    scheduler.enqueue_urgent(
        CopyJob(
            cookie=0,
            segments=(segment,) * 4096,
            wait_event=gate.cuda_event,
            done_event=ready.cuda_event,
        )
    )

    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        stats = scheduler.stats()
        if stats["queued_urgent_jobs"] == 0 and stats["issued_jobs"] == 0:
            break
    else:
        scheduler.close()
        pytest.fail("did not observe the urgent job while it was issuing")

    scheduler.close()

    assert ready.query()
    assert torch.equal(destination.cpu(), source)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not scheduler_module.native_copy_scheduler_available(),
    reason="requires the native CUDA copy scheduler",
)
def test_speculative_runway_queues_two_and_cancels_the_remainder() -> None:
    device = torch.cuda.current_device()
    copy_stream = torch.cuda.Stream(device=device)
    delay_stream = torch.cuda.Stream(device=device)
    source_storage = torch.arange(3 * 256, dtype=torch.int32).reshape(3, 256)
    source_storage = source_storage.pin_memory()
    destination_storage = torch.full_like(
        source_storage,
        -1,
        device=f"cuda:{device}",
    )
    gate = torch.cuda.Event()
    with torch.cuda.stream(delay_stream):
        torch.cuda._sleep(100_000_000)
        gate.record(delay_stream)
    ready_events = [torch.cuda.Event() for _ in range(3)]
    with torch.cuda.stream(copy_stream):
        for ready in ready_events:
            ready.record(copy_stream)
    copy_stream.synchronize()
    scheduler = NativeCudaCopyScheduler(
        copy_stream.cuda_stream,
        device=device,
        max_inflight=2,
        poll_interval_us=10,
    )
    try:
        jobs = tuple(
            CopyJob(
                cookie=index,
                segments=(
                    CopySegment(
                        source.data_ptr(),
                        destination.data_ptr(),
                        source.nbytes,
                        CudaMemcpyKind.HOST_TO_DEVICE,
                    ),
                ),
                wait_event=gate.cuda_event,
                done_event=ready.cuda_event,
            )
            for index, (source, destination, ready) in enumerate(
                zip(source_storage, destination_storage, ready_events)
            )
        )
        handle = scheduler.submit_window(jobs)

        for _ in range(1000):
            snapshot = scheduler.snapshot(handle)
            if snapshot.issued == (0, 1):
                break
        else:
            pytest.fail(f"two-copy runway was not filled: {snapshot}")

        snapshot = scheduler.cancel_and_snapshot(handle)
        assert snapshot.issued == (0, 1)
        assert snapshot.canceled == (2,)

        copy_stream.synchronize()
        assert torch.equal(destination_storage[:2].cpu(), source_storage[:2])
        assert torch.equal(
            destination_storage[2].cpu(),
            torch.full_like(source_storage[2], -1),
        )
        scheduler.release_window(handle)
    finally:
        scheduler.close()
