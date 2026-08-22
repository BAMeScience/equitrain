# Data and Preprocessing

Equitrain stores training data as HDF5 files containing ASE structures plus the
target and metadata fields needed by the Torch and JAX backends.

## Input Formats

`equitrain-preprocess` accepts:

- `.xyz` and `.xyz.gz`: parsed through ASE and converted to Equitrain HDF5.
- `.lmdb` and `.aselmdb`: converted through FairChem's `AseDBDataset`.
- `.h5` and `.hdf5`: copied into the output directory if needed.

For Torch training, evaluation, and prediction, HDF5 input paths can be a
single file, directory, glob, or comma-separated list. Directories and globs are
sorted and concatenated in order. JAX CLI workflows currently expect explicit
HDF5 file paths; the lower-level JAX loader can also receive a Python list or
tuple of paths.

When an existing HDF5 file is passed to preprocessing, Equitrain copies it
without reinterpreting raw XYZ keys. Options such as `--energy-key` and
`--niggli-reduce` only affect raw XYZ or LMDB conversion.

## Output Files

The preprocessing command writes split files into `--output-dir`:

```text
data/
  train.h5
  valid.h5
  test.h5
  statistics.json
  preprocess_summary.json
```

`statistics.json` is written when `--compute-statistics` is enabled. It stores
the atomic number table, atomic energy offsets, cutoff radius, mean/std energy
statistics, and average neighbor count used by training. Use `--atomic-numbers`
when the element table should be fixed explicitly instead of inferred from the
training split.

`preprocess_summary.json` records target availability for each split. It is a
quick way to catch mismatched target keys or datasets where energies, forces, or
stress are missing.

## HDF5 Layout

Equitrain HDF5 files contain:

| Dataset | Contents |
| --- | --- |
| `/structures` | One row per configuration, including atom offset/length, cell, PBC, energy, stress, virials, dipole, total charge/spin, external field, reaction ids, and target weights. |
| `/positions` | Flat `(n_atoms_total, 3)` position array. |
| `/forces` | Flat `(n_atoms_total, 3)` force target array. |
| `/atomic_numbers` | Flat `(n_atoms_total,)` atomic number array. |

Per-configuration rows point into the flat per-atom arrays with `offset` and
`length`. This keeps random reads compact for large datasets.

The current persisted training layout stores total system charge and spin. It
does not store a separate per-atom charge array, even though `Configuration`
objects can carry `charges` while converting structures.

## Target Keys

Use these options when source XYZ fields do not use Equitrain's defaults:

| Option | Default | Source |
| --- | --- | --- |
| `--energy-key` | `energy` | `atoms.info`; with the default key, ASE calculator energy is used when present. |
| `--forces-key` | `forces` | `atoms.arrays`; with the default key, ASE calculator forces are used when present. |
| `--stress-key` | `stress` | `atoms.info`; with the default key, ASE calculator stress is used when present. |
| `--virials-key` | `virials` | `atoms.info`. |
| `--dipole-key` | `dipole` | `atoms.info`. |
| `--charges-key` | `charges` | `atoms.arrays`, parsed at the `Configuration` layer only. |

If energy, forces, stress, virials, or dipole are missing, Equitrain writes zero
placeholders and sets the corresponding per-configuration `*_weight` field to
`0.0`. Present targets receive weight `1.0`.

Global loss weights such as `--energy-weight`, `--forces-weight`, and
`--stress-weight` still decide which quantities participate in training.

## Metadata Keys

Use these options for system-level and reaction metadata:

| Option | Default | Stored field |
| --- | --- | --- |
| `--total-charge-key` | `charge` | `total_charge` and `charge` aliases on loaded ASE atoms. |
| `--total-spin-key` | `spin` | `total_spin` and `spin` aliases on loaded ASE atoms. |
| `--external-field-key` | `external_field` | `external_field`, reshaped to length 3. |
| `--source-id-key` | `source_id` | Integer source dataset id. |
| `--reaction-id-key` | `reaction_id` | Integer reaction group id. |
| `--state-id-key` | `state_id` | Integer reaction state id. |

`reaction_id=-1` marks ordinary non-reaction frames. See
[Reaction-Relative Losses](reaction-relative-losses.md) for the state-id
convention used by barrier and reaction-energy losses.

## Splits

If `--valid-file` is provided, it is converted to `valid.h5`. If it is omitted,
`--valid-fraction` controls how much of the training input is held out for
validation.

Pass `--test-file` when you want preprocessing to also write `test.h5`.

## Niggli Reduction

`--niggli-reduce` applies ASE Niggli reduction to periodic cells during raw
XYZ/LMDB conversion. It is skipped for existing HDF5 inputs because Equitrain
does not rewrite those structures during preprocessing.

Training, evaluation, and prediction do not expose a Niggli-reduction flag;
they use the cells stored in HDF5 or provided to the prediction input as-is.
Regenerate the HDF5 files from raw structures when reduced cells are required.

## HDF5 Utilities

Inspect one or more HDF5 files:

```bash
equitrain-hdf5-info data/train.h5
equitrain-hdf5-info 'data/*.h5' --max-entries 1000
```

Benchmark sequential reads:

```bash
equitrain-hdf5-benchmark data/train.h5 --repeat 3 --warmup 1 --touch
equitrain-hdf5-benchmark data/train.h5 --shuffle --count 10000
```

Use `--touch` when you want the benchmark to materialize positions, atomic
numbers, and forces rather than only constructing ASE objects.

## Python Helpers

The public data helpers are available from `equitrain.data`:

```python
from equitrain.data import AtomicNumberTable, Configuration, Statistics
from equitrain.data.format_hdf5 import HDF5Dataset, HDF5GraphDataset
from equitrain.data.format_lmdb import convert_lmdb_to_hdf5
```

`HDF5Dataset` reads and writes ASE `Atoms` objects. `HDF5GraphDataset` wraps the
same file with on-the-fly graph construction for the Torch backend.
