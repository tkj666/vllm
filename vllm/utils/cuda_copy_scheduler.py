# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum, auto
from typing import Protocol, TypeVar, runtime_checkable

_T = TypeVar("_T")
_DEFAULT_POLL_INTERVAL_S = 0.00005

logger = logging.getLogger(__name__)


@runtime_checkable
class CopyCompletionEvent(Protocol):
    """Completion event returned after enqueuing an asynchronous copy."""

    def query(self) -> bool: ...

    def synchronize(self) -> None: ...


class CopyIssueStatus(Enum):
    """Non-event result from a speculative issue callback."""

    SKIPPED = auto()


class CopyWindow:
    """Handle for one ascending speculative-copy window."""

    def __init__(
        self,
        owner: object,
        num_items: int,
        issue: Callable[[int], CopyCompletionEvent | CopyIssueStatus | None],
        on_discard: Callable[[int], None] | None,
        on_error: Callable[[Exception], None] | None,
    ) -> None:
        self._owner = owner
        self.num_items = num_items
        self._issue = issue
        self._on_discard = on_discard
        self._on_error = on_error
        self._next_index = 0
        self._accepting = num_items > 0
        self._queued = False
        self._stop_event: CopyCompletionEvent | None = None


@dataclass(frozen=True)
class _InflightCopy:
    window: CopyWindow
    event: CopyCompletionEvent


class CudaCopyScheduler:
    """Keep asynchronous copies queued from one background issue thread.

    The scheduler is CUDA-shaped but has no CUDA or torch dependency. Normal
    speculative progress polls completion events in batches while later copies
    remain queued on the communication stream. Urgent callbacks are admission
    barriers, not scheduler-owned copies; their caller retains completion-event
    ownership.
    """

    def __init__(
        self,
        max_inflight: int = 1,
        *,
        poll_interval_s: float = _DEFAULT_POLL_INTERVAL_S,
        on_discard: Callable[[int], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
        thread_name: str = "vllm-cuda-copy-scheduler",
    ) -> None:
        if max_inflight <= 0:
            raise ValueError("max_inflight must be positive")
        if poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be positive")

        self._max_inflight = max_inflight
        self._poll_interval_s = poll_interval_s
        self._default_on_discard = on_discard
        self._default_on_error = on_error
        self._owner = object()
        self._condition = threading.Condition()
        self._windows: deque[CopyWindow] = deque()
        self._deferred_discards: deque[tuple[CopyWindow, int]] = deque()
        self._deferred_errors: deque[tuple[CopyWindow, Exception]] = deque()
        self._inflight: list[_InflightCopy] = []
        self._issuing: tuple[CopyWindow, int] | None = None
        self._querying = False
        self._priming = False
        self._draining = False
        self._urgent_waiters = 0
        self._urgent_active = False
        self._pause_depth = 0
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name=thread_name,
            daemon=True,
        )
        self._thread.start()

    def submit_window(
        self,
        num_items: int,
        issue: Callable[[int], CopyCompletionEvent | CopyIssueStatus | None],
        *,
        activate: bool = True,
        prime_runway: bool = False,
        on_discard: Callable[[int], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ) -> CopyWindow:
        """Queue a speculative window whose items are issued in ascending order."""
        if num_items < 0:
            raise ValueError("num_items must be non-negative")
        window = CopyWindow(
            self._owner,
            num_items,
            issue,
            on_discard if on_discard is not None else self._default_on_discard,
            on_error if on_error is not None else self._default_on_error,
        )
        if activate:
            self.activate_window(window, prime_runway=prime_runway)
        return window

    def activate_window(
        self,
        window: CopyWindow,
        *,
        prime_runway: bool = False,
    ) -> None:
        """Make a prepared window visible and optionally prime it inline."""
        self._check_window(window)
        with self._condition:
            self._raise_if_closed()
            if self._pause_depth:
                raise RuntimeError("CUDA copy scheduler is paused")
            if window._queued:
                raise RuntimeError("copy window is already active")
            if not window._accepting:
                return
            window._queued = True
            if prime_runway:
                self._condition.wait_for(lambda: not self._priming)
                self._priming = True
            self._windows.append(window)
            if not prime_runway:
                self._condition.notify_all()
        if prime_runway:
            try:
                self._prime_runway()
            finally:
                with self._condition:
                    self._priming = False
                    self._condition.notify_all()

    def pending_count(self, window: CopyWindow) -> int:
        """Return the number of not-yet-issued items in a window."""
        self._check_window(window)
        with self._condition:
            if not window._accepting:
                return 0
            return window.num_items - window._next_index

    def set_stop_event(
        self,
        window: CopyWindow,
        event: CopyCompletionEvent,
    ) -> None:
        """Stop admitting window items once a device event becomes ready."""
        self._check_window(window)
        with self._condition:
            if window._stop_event is not None:
                raise RuntimeError("copy window already has a stop event")
            window._stop_event = event
            self._condition.notify_all()

    def cancel_pending(
        self,
        window: CopyWindow,
        *,
        wait_for_issue: bool = True,
        defer_discard: bool = False,
    ) -> None:
        """Cancel unissued items, optionally waiting for an active callback."""
        self._check_window(window)
        with self._condition:
            discarded = self._discard_window_locked(window)
            self._condition.notify_all()
            if wait_for_issue:
                self._condition.wait_for(
                    lambda: self._issuing is None or self._issuing[0] is not window
                )
            if defer_discard and discarded:
                self._deferred_discards.append((window, discarded))
                self._condition.notify_all()
                discarded = 0
        self._notify_discard(window, discarded)

    def submit_urgent(self, callback: Callable[[], _T]) -> _T:
        """Run urgent work while blocking new speculative reservations.

        An already-running speculative callback may overlap this callback. This
        avoids lock inversion when urgent callers hold resources needed by the
        speculative callback. The copy backend must serialize commands issued
        to a shared communication stream.
        """
        if threading.current_thread() is self._thread:
            with self._condition:
                nested_in_issue = self._issuing is not None
            if nested_in_issue:
                return callback()

        with self._condition:
            self._raise_if_closed()
            if self._pause_depth:
                raise RuntimeError("CUDA copy scheduler is paused")
            self._urgent_waiters += 1
            self._condition.notify_all()
            try:
                self._condition.wait_for(
                    lambda: self._closed or not self._urgent_active
                )
                self._raise_if_closed()
                self._urgent_active = True
            finally:
                self._urgent_waiters -= 1
                self._condition.notify_all()

        try:
            return callback()
        finally:
            with self._condition:
                self._urgent_active = False
                self._condition.notify_all()

    def pause_and_drain(self) -> None:
        """Pause all admission and drain previously submitted work."""
        with self._condition:
            if self._closed:
                return
            self._pause_depth += 1
            discarded = self._discard_all_windows_locked()
            self._condition.notify_all()
        self._notify_discards(discarded)
        self._drain_submitted(wait_for_urgent=True)

    def resume(self) -> None:
        """Release one nesting level established by pause_and_drain."""
        with self._condition:
            if self._pause_depth == 0:
                if self._closed:
                    return
                raise RuntimeError("CUDA copy scheduler is not paused")
            self._pause_depth -= 1
            self._condition.notify_all()

    def close(self, *, wait: bool = True) -> None:
        """Stop and optionally drain scheduler-owned speculative copies."""
        with self._condition:
            self._closed = True
            discarded = self._discard_all_windows_locked()
            self._condition.notify_all()
        self._notify_discards(discarded)

        if not wait:
            return
        if threading.current_thread() is self._thread:
            raise RuntimeError("copy scheduler worker cannot wait for itself")
        self._drain_submitted(wait_for_urgent=True)
        self._thread.join()

    def _run(self) -> None:
        while True:
            reservation: tuple[CopyWindow, int] | None = None
            events: tuple[_InflightCopy, ...] | None = None
            deferred_discard: tuple[CopyWindow, int] | None = None
            deferred_error: tuple[CopyWindow, Exception] | None = None
            with self._condition:
                while (
                    reservation is None
                    and events is None
                    and deferred_discard is None
                    and deferred_error is None
                ):
                    if self._deferred_discards:
                        deferred_discard = self._deferred_discards.popleft()
                        break
                    if self._deferred_errors:
                        deferred_error = self._deferred_errors.popleft()
                        break
                    if self._closed:
                        self._condition.notify_all()
                        return
                    if (
                        self._pause_depth
                        or self._draining
                        or self._priming
                        or self._urgent_active
                        or self._urgent_waiters
                    ):
                        self._condition.wait()
                        continue

                    if len(self._inflight) < self._max_inflight:
                        reservation = self._reserve_next_locked()
                        if reservation is not None:
                            self._issuing = reservation
                            break

                    if self._inflight:
                        self._condition.wait(timeout=self._poll_interval_s)
                        if (
                            self._closed
                            or self._pause_depth
                            or self._draining
                            or self._priming
                            or self._urgent_active
                            or self._urgent_waiters
                        ):
                            continue
                        events = tuple(self._inflight)
                        self._querying = True
                        break
                    if self._deferred_discards:
                        continue
                    self._condition.wait()

            if deferred_discard is not None:
                self._notify_discard(*deferred_discard)
            elif deferred_error is not None:
                self._notify_error(*deferred_error)
            elif reservation is not None:
                self._issue_reserved(*reservation)
            else:
                assert events is not None
                self._query_events(events)

    def _reserve_next_locked(self) -> tuple[CopyWindow, int] | None:
        while self._windows:
            window = self._windows[0]
            if not window._accepting or window._next_index >= window.num_items:
                self._windows.popleft()
                continue
            if window._stop_event is not None:
                try:
                    stop = window._stop_event.query()
                except Exception as error:
                    count = self._discard_window_locked(window)
                    if count:
                        self._deferred_discards.append((window, count))
                    self._deferred_errors.append((window, error))
                    continue
                if stop:
                    count = self._discard_window_locked(window)
                    if count:
                        self._deferred_discards.append((window, count))
                    continue
            index = window._next_index
            window._next_index += 1
            if window._next_index >= window.num_items:
                self._windows.popleft()
            return window, index
        return None

    def _prime_runway(self) -> None:
        while True:
            with self._condition:
                if (
                    self._closed
                    or self._pause_depth
                    or self._urgent_active
                    or self._urgent_waiters
                    or len(self._inflight) >= self._max_inflight
                ):
                    return
                reservation = self._reserve_next_locked()
                if reservation is None:
                    return
                self._issuing = reservation
            self._issue_reserved(*reservation)

    def _issue_reserved(self, window: CopyWindow, index: int) -> None:
        event: CopyCompletionEvent | None = None
        skipped = False
        error: Exception | None = None
        try:
            result = window._issue(index)
            if result is CopyIssueStatus.SKIPPED:
                skipped = True
            else:
                event = result
        except Exception as exc:
            error = exc

        with self._condition:
            if self._issuing != (window, index):
                raise RuntimeError("copy scheduler lost its issue reservation")
            self._issuing = None
            if event is not None and error is None:
                self._inflight.append(_InflightCopy(window, event))
            discarded = 0
            if (event is None and not skipped) or error is not None:
                discarded = self._discard_window_locked(window)
            self._condition.notify_all()

        self._notify_discard(window, discarded)
        if error is not None:
            self._notify_error(window, error)

    def _query_events(self, events: tuple[_InflightCopy, ...]) -> None:
        completed: list[_InflightCopy] = []
        failures: list[tuple[_InflightCopy, Exception]] = []
        for copy in events:
            try:
                if copy.event.query():
                    completed.append(copy)
            except Exception as exc:
                failures.append((copy, exc))

        discarded: list[tuple[CopyWindow, int]] = []
        with self._condition:
            for copy in completed:
                self._remove_inflight_locked(copy)
            for copy, _ in failures:
                self._remove_inflight_locked(copy)
                count = self._discard_window_locked(copy.window)
                if count:
                    discarded.append((copy.window, count))
            self._querying = False
            self._condition.notify_all()

        self._notify_discards(discarded)
        for copy, error in failures:
            self._notify_error(copy.window, error)

    def _drain_submitted(self, *, wait_for_urgent: bool = False) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: (
                        self._issuing is None
                        and not self._querying
                        and not self._priming
                        and not self._draining
                        and (
                            not wait_for_urgent
                            or (not self._urgent_active and self._urgent_waiters == 0)
                        )
                    )
                )
                if not self._inflight:
                    return
                events = tuple(self._inflight)
                self._draining = True

            failures: list[tuple[_InflightCopy, Exception]] = []
            for copy in events:
                try:
                    copy.event.synchronize()
                except Exception as exc:
                    failures.append((copy, exc))

            with self._condition:
                for copy in events:
                    self._remove_inflight_locked(copy)
                self._draining = False
                self._condition.notify_all()

            for copy, error in failures:
                self._notify_error(copy.window, error)

    def _discard_all_windows_locked(self) -> list[tuple[CopyWindow, int]]:
        discarded = []
        for window in tuple(self._windows):
            count = self._discard_window_locked(window)
            if count:
                discarded.append((window, count))
        return discarded

    def _discard_window_locked(self, window: CopyWindow) -> int:
        if not window._accepting:
            return 0
        window._accepting = False
        discarded = window.num_items - window._next_index
        with suppress(ValueError):
            self._windows.remove(window)
        return discarded

    def _remove_inflight_locked(self, copy: _InflightCopy) -> None:
        for index, candidate in enumerate(self._inflight):
            if candidate is copy:
                self._inflight.pop(index)
                return

    def _check_window(self, window: CopyWindow) -> None:
        if not isinstance(window, CopyWindow) or window._owner is not self._owner:
            raise ValueError("copy window belongs to another scheduler")

    def _raise_if_closed(self) -> None:
        if self._closed:
            raise RuntimeError("CUDA copy scheduler is closed")

    def _notify_discards(self, discarded: list[tuple[CopyWindow, int]]) -> None:
        for window, count in discarded:
            self._notify_discard(window, count)

    def _notify_discard(self, window: CopyWindow, count: int) -> None:
        if count <= 0 or window._on_discard is None:
            return
        try:
            window._on_discard(count)
        except Exception as exc:
            self._notify_error(window, exc)

    def _notify_error(self, window: CopyWindow, error: Exception) -> None:
        if window._on_error is None:
            logger.exception(
                "CUDA copy scheduler callback failed",
                exc_info=(type(error), error, error.__traceback__),
            )
            return
        try:
            window._on_error(error)
        except Exception:
            logger.exception("CUDA copy scheduler error callback failed")
