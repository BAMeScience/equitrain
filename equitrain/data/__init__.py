"""Convenience re-exports for the data package."""

from .atomic import AtomicNumberTable
from .configuration import Configuration
from .statistics_data import Statistics, get_atomic_energies

__all__ = [
    'AtomicNumberTable',
    'Configuration',
    'Statistics',
    'get_atomic_energies',
]
