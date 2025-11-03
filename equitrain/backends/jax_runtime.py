from __future__ import annotations

import multiprocessing as mp
import threading

_spawn_lock = threading.Lock()


def ensure_multiprocessing_spawn() -> None:
    """Ensure multiprocessing uses the 'spawn' start method to avoid JAX fork warnings."""

    with _spawn_lock:
        try:
            if mp.get_start_method(allow_none=True) != 'spawn':
                mp.set_start_method('spawn')
        except RuntimeError:
            # Start method already set in this process; nothing else to do.
            pass


# Ensure we configure the start method as soon as this module is imported.
ensure_multiprocessing_spawn()


__all__ = ['ensure_multiprocessing_spawn']
