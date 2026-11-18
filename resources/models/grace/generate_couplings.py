"""Regenerate the fixed GRACE-1L angular tables from a local upstream checkout.

No model files, TensorFlow, or trained weights are used. Requires numpy, pandas,
and sympy. Generated tables inherit the upstream Academic Software Licence.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


def generate(upstream, output):
    path = upstream / 'tensorpotential/functions/couplings.py'
    spec = importlib.util.spec_from_file_location('grace_couplings_reference', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = {
        'source': 'tensorpotential 0.6.2.1; GRACE_1LAYER_v2_25; legacy coupling format',
        'license': 'Academic Software Licence (ASL)',
        'tables': {},
    }
    for lmax in range(5):
        a = pd.DataFrame(
            [
                [l_val, m, '', (-1) ** l_val, l_val]
                for l_val in range(lmax + 1)
                for m in range(-l_val, l_val + 1)
            ],
            columns=['l', 'm', 'hist', 'parity', 'sum_of_ls'],
        )

        def product(left, right, max_l, equal):
            return module.real_coupling_metainformation(
                A=left,
                B=right,
                lmax=lmax,
                Lmax=max_l,
                is_A_B_equal=equal,
                keep_parity=[[l_val, (-1) ** l_val] for l_val in range(lmax + 1)],
                normalize=True,
                legacy_format=True,
            )

        aa = product(a, a, lmax, True)
        tables = {}
        for name, frame in [
            ('AA', aa),
            ('AAA', product(aa, a, 0, False)),
            ('AAAA', product(aa, aa, 0, True)),
        ]:
            tables[name] = {
                'meta': frame[
                    ['l', 'm', 'hist', 'parity', 'sum_of_ls']
                ].values.tolist(),
                'left': np.concatenate(frame.left_inds).tolist(),
                'right': np.concatenate(frame.right_inds).tolist(),
                'coefficients': np.concatenate(frame.cg_list).tolist(),
                'output': [i for i, row in frame.iterrows() for _ in row.cg_list],
            }
        result['tables'][str(lmax)] = tables
    output.write_text(json.dumps(result, separators=(',', ':')) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('upstream', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    generate(args.upstream, args.output)
