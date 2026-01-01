# MACE-JAX Resources for Equitrain

Utilities and examples for working with pre-trained [MACE](https://github.com/ACEsuit/mace) foundation models in the JAX backend via the companion [mace-jax](https://github.com/ACEsuit/mace-jax) project.

## Contents

- `convert_foundation_to_jax.py` – downloads a Torch MACE foundation model (e.g. the `mp` “small” checkpoint), converts it to MACE-JAX parameters using `mace_jax.cli.mace_jax_from_torch`, and writes a ready-to-use bundle (`config.json` + `params.msgpack`).

## Usage

Activate an environment that has both `mace` and `mace-jax` installed (including the optional `cuequivariance` extras when available), then run:

```bash
python resources/models/mace-jax/convert_foundation_to_jax.py \
    --source mp \
    --model small \
    --output-dir resources/models/mace-jax/mp-small-jax
```

This produces a directory containing the serialized parameters and a JSON configuration that can be passed directly to Equitrain’s JAX backend (`--model path/to/bundle`) or loaded with the utilities in `mace_jax.tools`.

Use `--source` to pick a different foundation family (`mp`, `off`, `anicc`, `omol`) and `--model` to select a specific variant when multiple sizes exist.

## Dependencies

The script relies on the optional `mace` and `mace-jax` stacks, including their CUDA-enabled cuequivariance extensions. Install them via:

```bash
pip install equitrain[mace,jax]  # or the corresponding mace/mace-jax wheels
```

If the cuequivariance libraries are unavailable, the script will exit after downloading the Torch model; the export step itself requires the accelerated kernels to be importable. Run `python -c "import mace_jax, cuequivariance_ops_torch"` to check whether your environment is configured correctly.
