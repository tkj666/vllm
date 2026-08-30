# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import pytest

from vllm.utils.cuda_copy_scheduler import CopyIssueStatus, CudaCopyScheduler


class _FakeEvent:
    def __init__(self, *, complete: bool = False) -> None:
        self._complete = threading.Event()
        if complete:
            self._complete.set()
        self._lock = threading.Lock()
        self.query_count = 0
        self.synchronize_count = 0
        self.synchronize_entered = threading.Event()

    def query(self) -> bool:
        with self._lock:
            self.query_count += 1
        return self._complete.is_set()

    def synchronize(self) -> None:
        with self._lock:
            self.synchronize_count += 1
        self.synchronize_entered.set()
        assert self._complete.wait(timeout=2)

    def complete(self) -> None:
        self._complete.set()


def _wait_for(predicate: Callable[[], bool], timeout: float = 2) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for scheduler state")
        time.sleep(0.001)


def test_inflight_bound_refills_in_ascending_order() -> None:
    issued: list[int] = []
    events: dict[int, _FakeEvent] = {}
    lock = threading.Lock()
    scheduler = CudaCopyScheduler(max_inflight=2)

    def issue(index: int) -> _FakeEvent:
        event = _FakeEvent()
        with lock:
            issued.append(index)
            events[index] = event
        return event

    window = scheduler.submit_window(5, issue)
    try:
        _wait_for(lambda: len(issued) == 2)
        time.sleep(0.02)
        assert issued == [0, 1]
        assert scheduler.pending_count(window) == 3

        events[0].complete()
        _wait_for(lambda: len(issued) == 3)
        assert issued == [0, 1, 2]
    finally:
        scheduler.cancel_pending(window)
        assert scheduler.pending_count(window) == 0
        for event in events.values():
            event.complete()
        scheduler.close()


def test_prime_runway_skips_items_without_consuming_copy_credit() -> None:
    issued: list[int] = []
    events: list[_FakeEvent] = []
    scheduler = CudaCopyScheduler(max_inflight=2)

    def issue(index: int) -> _FakeEvent | CopyIssueStatus:
        issued.append(index)
        if index in (0, 2):
            return CopyIssueStatus.SKIPPED
        event = _FakeEvent()
        events.append(event)
        return event

    window = scheduler.submit_window(6, issue, activate=False)
    try:
        scheduler.activate_window(window, prime_runway=True)
        assert issued == [0, 1, 2, 3]
        assert scheduler.pending_count(window) == 2
    finally:
        scheduler.cancel_pending(window)
        for event in events:
            event.complete()
        scheduler.close()


def test_stop_event_discards_window_before_next_refill() -> None:
    issued: list[int] = []
    discarded: list[int] = []
    copy_event = _FakeEvent()
    stop_event = _FakeEvent()
    scheduler = CudaCopyScheduler(max_inflight=1)

    def issue(index: int) -> _FakeEvent:
        issued.append(index)
        return copy_event

    window = scheduler.submit_window(4, issue, on_discard=discarded.append)
    try:
        _wait_for(lambda: issued == [0])
        scheduler.set_stop_event(window, stop_event)
        stop_event.complete()
        copy_event.complete()
        _wait_for(lambda: discarded == [3])
        assert issued == [0]
        assert scheduler.pending_count(window) == 0
    finally:
        scheduler.close()


def test_stop_event_error_stops_only_its_window() -> None:
    issued: list[tuple[str, int]] = []
    discarded: list[int] = []
    errors: list[Exception] = []
    copy_event = _FakeEvent()
    scheduler = CudaCopyScheduler(max_inflight=1)

    class FailingStopEvent(_FakeEvent):
        def query(self) -> bool:
            raise RuntimeError("stop query failed")

    first = scheduler.submit_window(
        4,
        lambda index: (issued.append(("first", index)), copy_event)[1],
        on_discard=discarded.append,
        on_error=errors.append,
    )
    try:
        _wait_for(lambda: issued == [("first", 0)])
        scheduler.set_stop_event(first, FailingStopEvent())
        copy_event.complete()
        _wait_for(lambda: len(errors) == 1)

        scheduler.submit_window(
            1,
            lambda index: (
                issued.append(("second", index)),
                _FakeEvent(complete=True),
            )[1],
        )
        _wait_for(lambda: issued[-1] == ("second", 0))

        assert scheduler._thread.is_alive()
        assert discarded == [3]
        assert str(errors[0]) == "stop query failed"
    finally:
        scheduler.close()


def test_cancel_waits_for_issue_and_prevents_post_return_submission() -> None:
    issue_entered = threading.Event()
    release_issue = threading.Event()
    cancel_returned = threading.Event()
    issued: list[int] = []
    discarded: list[int] = []
    event = _FakeEvent()
    scheduler = CudaCopyScheduler()

    def issue(index: int) -> _FakeEvent:
        issued.append(index)
        issue_entered.set()
        assert release_issue.wait(timeout=2)
        return event

    window = scheduler.submit_window(5, issue, on_discard=discarded.append)
    assert issue_entered.wait(timeout=2)

    cancel_thread = threading.Thread(
        target=lambda: (scheduler.cancel_pending(window), cancel_returned.set())
    )
    cancel_thread.start()
    time.sleep(0.02)
    assert not cancel_returned.is_set()

    release_issue.set()
    assert cancel_returned.wait(timeout=2)
    cancel_thread.join()
    event.complete()
    time.sleep(0.02)
    scheduler.close()

    assert issued == [0]
    assert discarded == [4]


def test_urgent_submission_overtakes_pending_window_items() -> None:
    issue_entered = threading.Event()
    release_issue = threading.Event()
    order: list[str] = []
    order_lock = threading.Lock()
    scheduler = CudaCopyScheduler(max_inflight=1)

    def issue(index: int) -> _FakeEvent:
        with order_lock:
            order.append(f"speculative-{index}")
        if index == 0:
            issue_entered.set()
            assert release_issue.wait(timeout=2)
        return _FakeEvent(complete=True)

    scheduler.submit_window(3, issue)
    assert issue_entered.wait(timeout=2)

    urgent_done = threading.Event()

    def run_urgent() -> None:
        scheduler.submit_urgent(lambda: order.append("urgent"))
        urgent_done.set()

    urgent_thread = threading.Thread(target=run_urgent)
    urgent_thread.start()
    assert urgent_done.wait(timeout=2)
    release_issue.set()
    _wait_for(lambda: len(order) == 4)
    urgent_thread.join()
    scheduler.close()

    assert order == [
        "speculative-0",
        "urgent",
        "speculative-1",
        "speculative-2",
    ]


def test_urgent_submission_does_not_wait_on_blocked_speculative_callback() -> None:
    resource = threading.Lock()
    resource.acquire()
    issue_entered = threading.Event()
    urgent_done = threading.Event()
    scheduler = CudaCopyScheduler()

    def issue(index: int) -> _FakeEvent:
        issue_entered.set()
        with resource:
            return _FakeEvent(complete=True)

    window = scheduler.submit_window(1, issue)
    assert issue_entered.wait(timeout=2)

    urgent_thread = threading.Thread(
        target=lambda: (scheduler.submit_urgent(lambda: None), urgent_done.set())
    )
    urgent_thread.start()
    try:
        assert urgent_done.wait(timeout=2)
    finally:
        resource.release()
        urgent_thread.join(timeout=2)
        scheduler.cancel_pending(window)
        scheduler.close()


def test_urgent_submission_is_reentrant_from_issue_callback() -> None:
    order: list[str] = []
    scheduler = CudaCopyScheduler()

    def issue(index: int) -> _FakeEvent:
        order.append(f"issue-{index}")
        scheduler.submit_urgent(lambda: order.append(f"nested-{index}"))
        return _FakeEvent(complete=True)

    scheduler.submit_window(2, issue)
    _wait_for(lambda: len(order) == 4)
    scheduler.close()

    assert order == ["issue-0", "nested-0", "issue-1", "nested-1"]


def test_none_stops_window_and_discards_remainder() -> None:
    issued: list[int] = []
    discarded: list[int] = []
    scheduler = CudaCopyScheduler(max_inflight=2)

    def issue(index: int) -> _FakeEvent | None:
        issued.append(index)
        if index == 1:
            return None
        return _FakeEvent(complete=True)

    scheduler.submit_window(5, issue, on_discard=discarded.append)
    _wait_for(lambda: bool(discarded))
    scheduler.close()

    assert issued == [0, 1]
    assert discarded == [3]


def test_pause_drains_events_and_rejects_windows_until_nested_resume() -> None:
    events = [_FakeEvent(), _FakeEvent()]
    issued: list[int] = []
    discarded: list[int] = []
    scheduler = CudaCopyScheduler(max_inflight=2)

    def issue(index: int) -> _FakeEvent:
        issued.append(index)
        return events[index]

    scheduler.submit_window(4, issue, on_discard=discarded.append)
    _wait_for(lambda: issued == [0, 1])

    pause_done = threading.Event()
    pause_thread = threading.Thread(
        target=lambda: (scheduler.pause_and_drain(), pause_done.set())
    )
    pause_thread.start()
    assert events[0].synchronize_entered.wait(timeout=2)
    assert not pause_done.is_set()
    events[0].complete()
    assert events[1].synchronize_entered.wait(timeout=2)
    assert not pause_done.is_set()
    events[1].complete()
    assert pause_done.wait(timeout=2)
    pause_thread.join()

    query_counts = [event.query_count for event in events]
    time.sleep(0.02)
    assert [event.query_count for event in events] == query_counts
    assert discarded == [2]

    scheduler.pause_and_drain()
    with pytest.raises(RuntimeError, match="paused"):
        scheduler.submit_window(
            2,
            lambda index: (issued.append(index + 10), _FakeEvent(complete=True))[1],
        )
    assert issued == [0, 1]

    scheduler.resume()
    with pytest.raises(RuntimeError, match="paused"):
        scheduler.submit_window(1, lambda _: _FakeEvent(complete=True))
    scheduler.resume()
    scheduler.submit_window(
        2,
        lambda index: (issued.append(index + 10), _FakeEvent(complete=True))[1],
    )
    _wait_for(lambda: issued == [0, 1, 10, 11])
    scheduler.close()


def test_pause_rejects_new_urgent_and_drains_admitted_waiters() -> None:
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()
    release_second = threading.Event()
    pause_done = threading.Event()
    errors: list[BaseException] = []
    scheduler = CudaCopyScheduler()

    def urgent(
        entered: threading.Event,
        release: threading.Event,
    ) -> None:
        entered.set()
        assert release.wait(timeout=2)

    def run_urgent(
        entered: threading.Event,
        release: threading.Event,
    ) -> None:
        try:
            scheduler.submit_urgent(lambda: urgent(entered, release))
        except BaseException as error:
            errors.append(error)

    first_thread = threading.Thread(
        target=run_urgent,
        args=(first_entered, release_first),
    )
    second_thread = threading.Thread(
        target=run_urgent,
        args=(second_entered, release_second),
    )
    first_thread.start()
    assert first_entered.wait(timeout=2)
    second_thread.start()
    _wait_for(lambda: scheduler._urgent_waiters == 1)

    pause_thread = threading.Thread(
        target=lambda: (scheduler.pause_and_drain(), pause_done.set())
    )
    pause_thread.start()
    _wait_for(lambda: scheduler._pause_depth == 1)

    with pytest.raises(RuntimeError, match="paused"):
        scheduler.submit_urgent(lambda: None)
    assert not pause_done.is_set()

    release_first.set()
    assert second_entered.wait(timeout=2)
    assert not pause_done.is_set()
    release_second.set()
    assert pause_done.wait(timeout=2)

    for thread in (first_thread, second_thread, pause_thread):
        thread.join(timeout=2)
        assert not thread.is_alive()
    assert not errors

    scheduler.resume()
    scheduler.submit_urgent(lambda: None)
    scheduler.close()


def test_issue_error_reports_and_stops_window() -> None:
    issued: list[int] = []
    discarded: list[int] = []
    errors: list[Exception] = []
    scheduler = CudaCopyScheduler(max_inflight=1)

    def issue(index: int) -> _FakeEvent:
        issued.append(index)
        if index == 1:
            raise RuntimeError("copy submission failed")
        return _FakeEvent(complete=True)

    scheduler.submit_window(
        4,
        issue,
        on_discard=discarded.append,
        on_error=errors.append,
    )
    _wait_for(lambda: bool(errors))
    scheduler.close()

    assert issued == [0, 1]
    assert discarded == [2]
    assert len(errors) == 1
    assert str(errors[0]) == "copy submission failed"


def test_close_waits_for_submitted_events_and_rejects_new_work() -> None:
    events = [_FakeEvent(), _FakeEvent()]
    issued: list[int] = []
    discarded: list[int] = []
    scheduler = CudaCopyScheduler(max_inflight=2)

    def issue(index: int) -> _FakeEvent:
        issued.append(index)
        return events[index]

    scheduler.submit_window(3, issue, on_discard=discarded.append)
    _wait_for(lambda: issued == [0, 1])

    close_done = threading.Event()
    close_thread = threading.Thread(
        target=lambda: (scheduler.close(), close_done.set())
    )
    close_thread.start()
    assert events[0].synchronize_entered.wait(timeout=2)
    assert not close_done.is_set()
    with pytest.raises(RuntimeError, match="closed"):
        scheduler.submit_window(1, lambda _: _FakeEvent(complete=True))
    with pytest.raises(RuntimeError, match="closed"):
        scheduler.submit_urgent(lambda: None)

    events[0].complete()
    assert events[1].synchronize_entered.wait(timeout=2)
    events[1].complete()
    assert close_done.wait(timeout=2)
    close_thread.join()

    assert discarded == [1]
