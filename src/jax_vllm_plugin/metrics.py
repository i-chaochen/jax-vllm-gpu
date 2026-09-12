# SPDX-License-Identifier: Apache-2.0
"""Per-step timing, recorded where the step actually happens.

Written because deriving inter-token latency by subtraction is unsound on this
platform and produced a table that could not be true: throughput *fell* from
batch 4 to batch 8 while step time rose 2.5x, which cannot happen when every
batch size reads the same weights per step. The cause is that an unseen shape
costs seconds to compile, so a single compile inside a timed window is smeared
across every step in it -- and the larger the batch, the more shapes, the worse
the smearing.

A median over recorded steps is immune to that in a way an average is not, and
separating the *first* step at each shape makes the compile visible instead of
merely diluted. So this records rather than infers.

The recorder is module-level and always on. Two `perf_counter` calls per step is
nothing against a 35 ms step, and a benchmark that has to enable instrumentation
is a benchmark that will one day be run without it.
"""

from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StepRecord:
    """One engine step: the forward pass and the sampling that follows it."""

    shape: Any
    seconds: float
    num_requests: int
    tokens: int
    first_of_shape: bool


class StepMetrics:
    """A bounded log of recent steps, plus the summary a benchmark wants.

    Bounded because this runs in a serving process that may never stop, and an
    unbounded list of every step ever taken is a slow memory leak rather than an
    observability feature.
    """

    def __init__(self, capacity: int = 8192) -> None:
        self._records: deque[StepRecord] = deque(maxlen=capacity)
        self._seen: set = set()

    def record(self, shape: Any, seconds: float, num_requests: int, tokens: int) -> None:
        key = (shape, num_requests)
        first = key not in self._seen
        self._seen.add(key)
        self._records.append(
            StepRecord(shape, seconds, num_requests, tokens, first)
        )

    def reset(self) -> None:
        """Clear the log but keep the set of shapes already compiled.

        Deliberately asymmetric: a benchmark resets between configurations and
        wants a clean set of timings, but a shape compiled during warm-up is
        still compiled, so forgetting that would report every warmed step as a
        first-of-shape and undo the point of warming.
        """
        self._records.clear()

    def forget_shapes(self) -> None:
        """Also forget which shapes were seen. For tests, not for benchmarks."""
        self._seen.clear()

    @property
    def records(self) -> tuple[StepRecord, ...]:
        return tuple(self._records)

    def summary(self) -> dict:
        """Median step time over steps that did not compile, plus the outliers.

        Median rather than mean, and repeats rather than all steps, because the
        question a benchmark is asking -- what does a step cost in steady state --
        is not the question an average over a window containing a compile answers.
        """
        repeats = [r for r in self._records if not r.first_of_shape]
        compiles = [r for r in self._records if r.first_of_shape]
        if not repeats:
            return {
                "steps": len(self._records),
                "repeats": 0,
                "compiles": len(compiles),
                "compile_seconds": sum(r.seconds for r in compiles),
            }

        times = sorted(r.seconds for r in repeats)
        tokens_per_step = statistics.median(r.tokens for r in repeats)
        median = statistics.median(times)
        return {
            "steps": len(self._records),
            "repeats": len(repeats),
            "compiles": len(compiles),
            "compile_seconds": sum(r.seconds for r in compiles),
            "median_s": median,
            "p10_s": times[len(times) // 10],
            "p90_s": times[(len(times) * 9) // 10],
            "min_s": times[0],
            "max_s": times[-1],
            "tokens_per_step": tokens_per_step,
            "tokens_per_second": tokens_per_step / median if median else float("nan"),
            "distinct_shapes": len(self._seen),
        }


# The process-wide recorder. A benchmark imports this; nothing else has to know.
STEPS = StepMetrics()
