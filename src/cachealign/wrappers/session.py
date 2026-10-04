# Copyright 2026 Invarcore Organization
# SPDX-License-Identifier: MIT

"""
Session Context Managers and Decorators for CacheAlign.
"""

from collections.abc import Callable
from functools import wraps
from typing import Any

from cachealign.telemetry.reporter import FinOpsReporter, SessionTelemetry


class CacheAlignSession:
    """Scoped session context manager tracking cache optimization across multiple calls."""

    def __init__(self, verbose: bool = True):
        self.reporter = FinOpsReporter(verbose=verbose)

    @property
    def telemetry(self) -> SessionTelemetry:
        return self.reporter.session

    def __enter__(self) -> "CacheAlignSession":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass


def session(verbose: bool = True) -> CacheAlignSession:
    """Creates a scoped session context manager for tracking CacheAlign metrics."""
    return CacheAlignSession(verbose=verbose)


def optimize(verbose: bool = True) -> Callable[..., Any]:
    """Decorator to scope and monitor prompt cache optimization for a specific function."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with session(verbose=verbose):
                return fn(*args, **kwargs)

        @wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            with session(verbose=verbose):
                return await fn(*args, **kwargs)

        import inspect

        if inspect.iscoroutinefunction(fn):
            return async_wrapper
        return sync_wrapper

    return decorator
