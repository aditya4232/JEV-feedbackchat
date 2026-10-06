"""Bound caller waiting for synchronous providers, including body reads/retries."""

from __future__ import annotations

import math
from collections.abc import Callable
from concurrent.futures import Future, TimeoutError
from threading import Thread
from typing import TypeVar

T = TypeVar("T")


def validate_deadline(seconds: float) -> None:
    if isinstance(seconds, bool) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("deadline_seconds must be finite and positive")


def within_deadline(operation: Callable[[], T], seconds: float) -> T:
    """Fail on the overall deadline even when a transport stalls.

    A synchronous in-flight HTTP operation cannot be forcibly cancelled. Its
    socket timeout still applies; the daemon worker cannot block process exit.
    Never start another request after this function times out.
    """
    validate_deadline(seconds)
    future: Future[T] = Future()

    def run() -> None:
        try:
            future.set_result(operation())
        except BaseException as exc:
            future.set_exception(exc)

    Thread(target=run, daemon=True).start()
    return future.result(timeout=seconds)


__all__ = ["TimeoutError", "validate_deadline", "within_deadline"]
