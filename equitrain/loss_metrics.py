from __future__ import annotations

import math


class AverageMeter:
    """Backend-agnostic running average helper."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0.0
        self.avg = 0.0

    def update(self, value: float, n: float) -> None:
        if n <= 0 or not math.isfinite(value):
            return
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


__all__ = ['AverageMeter']
