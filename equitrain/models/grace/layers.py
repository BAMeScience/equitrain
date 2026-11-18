"""Torch adaptation of GRACE's 1L operations, 2026-11-18.

Based on tensorpotential by Anton Bochkarev and Yury Lysogorskiy.
Distributed under the Academic Software Licence in LICENSE.md.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class NormalizedLinear(nn.Module):
    """Bias-free GRACE linear map, with explicit fan-in normalization."""

    def __init__(self, n_in, n_out):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_in, n_out))
        self.norm = n_in**-0.5

    def forward(self, x):
        return (x @ self.weight) * self.norm


class MLP(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.layers = nn.ModuleList(
            NormalizedLinear(a, b) for a, b in zip(sizes, sizes[1:])
        )

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.silu(layer(x))
        return self.layers[-1](x)


def spherical_harmonics(unit_vectors, lmax):
    """Real, sqrt(4*pi)-scaled harmonics in GRACE's (l, m) order.

    Cartesian polynomial recurrence avoids atan2 and pole singularities, so
    forces and force-training derivatives remain defined on the coordinate axes.
    """
    x, y, z = unit_vectors.unbind(-1)
    one = torch.ones_like(z)
    p = {(0, 0): one}
    if lmax:
        p[1, 0] = math.sqrt(3) * z
        p[1, 1] = -math.sqrt(1.5) * one
    for degree in range(2, lmax + 1):
        for m in range(degree + 1):
            if m == degree:
                p[degree, m] = -math.sqrt(1 + 0.5 / m) * p[degree - 1, m - 1]
            elif m == degree - 1:
                p[degree, m] = math.sqrt(2 * m + 3) * z * p[m, m]
            else:
                a = math.sqrt((4 * degree**2 - 1) / (degree**2 - m**2))
                b = -math.sqrt(((degree - 1) ** 2 - m**2) / (4 * (degree - 1) ** 2 - 1))
                p[degree, m] = a * (z * p[degree - 1, m] + b * p[degree - 2, m])
    phases = [(one, torch.zeros_like(z))]
    for _ in range(lmax):
        re, im = phases[-1]
        phases.append((re * x - im * y, re * y + im * x))
    values = []
    for degree in range(lmax + 1):
        for m in range(-degree, degree + 1):
            if m == 0:
                values.append(p[degree, 0])
            else:
                phase = phases[abs(m)][int(m < 0)]
                values.append(math.sqrt(2) * (-1) ** abs(m) * phase * p[degree, abs(m)])
    return torch.stack(values, -1)


def chebyshev_radial(distances, cutoff, n_basis, order):
    """Unnormalized first-kind Chebyshev basis with GRACE's polynomial cutoff."""
    r = distances / cutoff
    # Clamp before evaluating the polynomial to avoid large out-of-range values.
    r = r.clamp(max=1)
    x = 2 * r - 1
    prev, current = torch.ones_like(x), x
    values = [current]
    for _ in range(1, n_basis):
        prev, current = current, 2 * x * current - prev
        values.append(current)
    # Algebraically the reference polynomial, factored at its triple root r=1.
    # The expanded form loses precision in float32 forces near the cutoff.
    polynomial = torch.full_like(r, order * (order + 1) / 2)
    for degree in range(order - 2, -1, -1):
        polynomial = polynomial * r + (degree + 1) * (degree + 2) / 2
    envelope = (1 - r) ** 3 * polynomial
    return torch.stack(values, -1) * envelope.unsqueeze(-1)


def _groups(meta):
    groups = {}
    for index, (degree, _m, history, parity, _sum_l) in enumerate(meta):
        groups.setdefault((degree, parity, history), []).append(index)
    return dict(sorted(groups.items()))


class RightToLeft(nn.Module):
    """Channel mixing and equivariant residual addition (FCRight2Left)."""

    def __init__(self, left_meta, right_meta, n_left, n_right, n_out):
        super().__init__()
        left_groups, right_groups = _groups(left_meta), _groups(right_meta)
        left_tile = [0] * len(left_meta)
        right_tile, collect_from, collect_to, norm = [], [], [], [0.0] * len(left_meta)
        count = 0
        for group_index, ((degree, parity, _), indices) in enumerate(
            left_groups.items()
        ):
            matches = [
                v
                for (l_val, p, _), v in right_groups.items()
                if l_val == degree and p == parity
            ]
            for i in indices:
                left_tile[i] = group_index
                norm[i] = (1 + len(matches)) ** -0.5
            for right_indices in matches:
                collect_to.extend(indices)
                collect_from.extend(right_indices)
                right_tile.extend([count] * len(indices))
                count += 1
        self.weight_left = nn.Parameter(torch.randn(n_out, n_left, len(left_groups)))
        self.weight_right = nn.Parameter(torch.randn(n_out, n_right, count))
        for name, value in [
            ('left_tile', left_tile),
            ('right_tile', right_tile),
            ('collect_from', collect_from),
            ('collect_to', collect_to),
        ]:
            self.register_buffer(name, torch.tensor(value, dtype=torch.long))
        self.register_buffer('output_norm', torch.tensor(norm, dtype=torch.float64))
        self.norm_left, self.norm_right = n_left**-0.5, n_right**-0.5

    def forward(self, left, right):
        left = (
            torch.einsum('onk,ank->aok', self.weight_left[..., self.left_tile], left)
            * self.norm_left
        )
        right = (
            torch.einsum(
                'onk,ank->aok',
                self.weight_right[..., self.right_tile],
                right[..., self.collect_from],
            )
            * self.norm_right
        )
        return left.index_add(-1, self.collect_to, right) * self.output_norm.to(
            left.dtype
        )


class TensorProduct(nn.Module):
    """Sparse real Clebsch–Gordan contraction, including reference pruning."""

    def __init__(self, table):
        super().__init__()
        self.n_components = len(table['meta'])
        for key in ('left', 'right', 'output'):
            self.register_buffer(key, torch.tensor(table[key], dtype=torch.long))
        # Keep coefficients accurate when the default Torch dtype is float32.
        self.register_buffer(
            'coefficients', torch.tensor(table['coefficients'], dtype=torch.float64)
        )

    def forward(self, left, right):
        products = left[..., self.left] * right[..., self.right]
        products = products * self.coefficients.to(dtype=left.dtype)
        return products.new_zeros((*products.shape[:-1], self.n_components)).index_add(
            -1, self.output, products
        )


class ScalarReduce(nn.Module):
    def __init__(self, meta, n_types, n_in, n_out):
        super().__init__()
        indices = [
            i
            for _, group in _groups(meta).items()
            for i in group
            if meta[i][0] == 0 and meta[i][3] == 1
        ]
        self.register_buffer('indices', torch.tensor(indices, dtype=torch.long))
        self.weight = nn.Parameter(torch.randn(n_types, n_out, n_in, len(indices)))
        self.norm = n_in**-0.5

    def forward(self, x, species):
        return (
            torch.einsum('aonw,anw->ao', self.weight[species], x[..., self.indices])
            * self.norm
        )
