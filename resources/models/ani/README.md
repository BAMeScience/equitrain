# ANI Resources for Equitrain

This directory bundles helper utilities for working with [TorchANI](https://github.com/aiqm/torchani) models inside Equitrain.

## Contents

- `ani-initial-model.py` – convenience script that exports one of the pre-trained TorchANI model families (ANI1x, ANI1ccx, ANI2x, … depending on the installed TorchANI release) into a checkpoint that can be consumed by `equitrain.utility_test.AniWrapper`.

Running the script will default to the single-precision ANI1x model and produce `ani-initial.model` in the current directory:

```bash
python resources/models/ani/ani-initial-model.py
```

Use the command line switches to choose a different variant, file name, or precision:

```bash
python resources/models/ani/ani-initial-model.py \
    --variant ANI2x \
    --dtype float64 \
    --output resources/models/ani/ani2x-initial.model
```

The script automatically checks which factory functions are available in the locally installed TorchANI build and errors out with a clear message if none are present.

## Installation

TorchANI and its dependencies are optional requirements. Install them via the project extra:

```bash
pip install equitrain[ani]
```

or manually:

```bash
pip install torchani ase
```

No other resources are required—the official TorchANI weights are downloaded automatically when the script runs for the first time.
