"""Convenience re-exports for the data package."""

from .atomic import AtomicNumberTable
from .configuration import Configuration
from .statistics import (
    compute_atomic_numbers,
    compute_average_atomic_energies,
    compute_statistics,
)
from .statistics_data import Statistics, get_atomic_energies

__all__ = [
    'AtomicNumberTable',
    'Configuration',
    'compute_atomic_numbers',
    'compute_average_atomic_energies',
    'compute_statistics',
    'Statistics',
    'get_atomic_energies',
]
