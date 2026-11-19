"""Backend-independent GRACE angular layout, ASL; see LICENSE.md.

Adapted from tensorpotential by Anton Bochkarev and Yury Lysogorskiy,
2026-11-18. No trained weights or tensor backend are used here.
"""

import json
from functools import lru_cache
from importlib.resources import files


@lru_cache(maxsize=5)
def angular_tables(lmax):
    return json.loads(files(__package__).joinpath('couplings.json').read_text())[
        'tables'
    ][str(lmax)]


def uncoupled_meta(lmax):
    return [
        [degree, m, '', (-1) ** degree, degree]
        for degree in range(lmax + 1)
        for m in range(-degree, degree + 1)
    ]


def groups(meta):
    result = {}
    for index, (degree, _m, history, parity, _sum_l) in enumerate(meta):
        result.setdefault((degree, parity, history), []).append(index)
    return dict(sorted(result.items()))


def scalar_indices(meta):
    return [
        i
        for _, group in groups(meta).items()
        for i in group
        if meta[i][0] == 0 and meta[i][3] == 1
    ]


def mixer_layout(left_meta, right_meta):
    left_groups, right_groups = groups(left_meta), groups(right_meta)
    left_tile = [0] * len(left_meta)
    right_tile, collect_from, collect_to, norm = [], [], [], [0.0] * len(left_meta)
    count = 0
    for group_index, ((degree, parity, _), indices) in enumerate(left_groups.items()):
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
    return dict(
        left_tile=left_tile,
        right_tile=right_tile,
        collect_from=collect_from,
        collect_to=collect_to,
        output_norm=norm,
        left_groups=len(left_groups),
        right_groups=count,
    )
