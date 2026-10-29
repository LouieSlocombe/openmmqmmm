"""Optional JSON-lines instrumentation for force callbacks and MD intervals."""

from __future__ import annotations

import functools
import json
import time
from collections.abc import Callable
from contextvars import ContextVar
from pathlib import Path
from typing import Any

_PHASES: ContextVar[dict[str, float] | None] = ContextVar("qmmm_profile_phases", default=None)


def _profile_phase(name: str) -> Callable:
    """Measure a wrapper phase only during an explicitly profiled force evaluation."""

    def decorate(function: Callable) -> Callable:
        @functools.wraps(function)
        def measured(*args: Any, **kwargs: Any) -> Any:
            phases = _PHASES.get()
            if phases is None:
                return function(*args, **kwargs)
            start = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                phases[name] = phases.get(name, 0.0) + time.perf_counter() - start

        return measured

    return decorate


class _MDProfile:
    """Append bounded-memory records without requesting an OpenMM energy or force."""

    def __init__(self, filename: str, interval: int, settings: dict[str, Any]) -> None:
        self.filename = str(Path(filename).resolve())
        self.interval = interval
        self.settings = settings
        self.callback_wall_s = 0.0
        self.evaluations = 0
        self.cache_hits = 0
        self.reset_interval(0)

    def emit(self, record: dict[str, Any]) -> None:
        with open(self.filename, "a") as output:
            output.write(json.dumps({"schema_version": 1, **self.settings, **record}, allow_nan=False) + "\n")

    def reset_interval(self, step: int) -> None:
        self._last_step = step
        self._last_time = time.perf_counter()
        self._last_callback_wall = self.callback_wall_s
        self._last_evaluations = self.evaluations
        self._last_cache_hits = self.cache_hits

    def report_step(self, step: int) -> None:
        elapsed = time.perf_counter() - self._last_time
        callback_wall = self.callback_wall_s - self._last_callback_wall
        self.emit(
            {
                "event": "md_step",
                "step": step,
                "steps": step - self._last_step,
                "wall_time_s": elapsed,
                "force_callback_wall_s": callback_wall,
                "other_wall_s": max(0.0, elapsed - callback_wall),
                "fresh_evaluations": self.evaluations - self._last_evaluations,
                "cache_hits": self.cache_hits - self._last_cache_hits,
            }
        )
        self.reset_interval(step)
