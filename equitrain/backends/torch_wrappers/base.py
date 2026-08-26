"""
Shared utilities for torch model wrappers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class AbstractWrapper(torch.nn.Module, ABC):
    """Common interface exposed by all torch wrappers."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    @abstractmethod
    def forward(self, *args):
        """Implement the model forward pass.

        Implementations must return a mapping with at least an ``'energy'``
        entry of shape ``[batch, 1]`` or ``[batch]``. Wrappers that produce
        forces or stresses should also return ``'forces'`` with shape
        ``[num_atoms, 3]`` and ``'stress'`` with shape ``[batch, 3, 3]``.
        Additional observables such as dipoles or virials can be included as
        extra keys; they are forwarded to the loss and metrics stack unchanged.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def atomic_numbers(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def atomic_energies(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def r_max(self):
        raise NotImplementedError

    @r_max.setter
    @abstractmethod
    def r_max(self, value):
        raise NotImplementedError


__all__ = ['AbstractWrapper']
