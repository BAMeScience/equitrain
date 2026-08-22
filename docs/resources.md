# Resources

The repository includes scripts and small examples that complement the main
package documentation.

## Data Preparation

Dataset preparation helpers live under `resources/data`.

- `resources/data/alexandria`
- `resources/data/alexandria+mptraj`
- `resources/data/mptraj`
- `resources/data/omat24`

These scripts show how to download or convert source datasets and write
Equitrain HDF5 files, including examples for Alexandria and MPTraj.

## Training

Distributed MACE training examples live under `resources/training`.

- `mace-alex-mptraj-multigpu.sh`
- `mace-alex-mptraj-multinode.sh`
- `mace-alex-mptraj-multinode-slurm.sh`
- corresponding Accelerate YAML files

The scripts are intended as templates for single-node multi-GPU and multi-node
jobs.

## Model Resources

Initial model helpers and wrapper-specific examples live under
`resources/models`.

- `resources/models/ani`: TorchANI export helper.
- `resources/models/mace-jax`: MACE foundation-to-JAX conversion helper.
- `resources/models/m3gnet`: M3GNet examples.
- `resources/models/orb`: ORB examples.
- `resources/models/sevennet`: SevenNet example configuration.

## HDF5 Inspection

Use `equitrain-hdf5-info` to inspect Equitrain HDF5 files:

```bash
equitrain-hdf5-info data/train.h5
```

Use `equitrain-hdf5-benchmark` to measure sequential read throughput:

```bash
equitrain-hdf5-benchmark data/train.h5
```
