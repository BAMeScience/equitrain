# Reaction-Relative Losses

Torch training can add relative energy targets for reaction data:

```bash
equitrain -v \
    --train-file data/train.h5 \
    --valid-file data/valid.h5 \
    --model path/to/mace.model \
    --model-wrapper mace \
    --barrier-weight 1.0 \
    --reaction-energy-weight 1.0 \
    --output-dir runs/reaction
```

The two relative losses are:

- `--barrier-weight`: `E_TS - E_reactant`
- `--reaction-energy-weight`: `E_product - E_reactant`

## Metadata

Reaction-relative data uses integer metadata:

- `source_id`: source dataset id, default `0`
- `reaction_id`: reaction group id, default `-1` for ordinary non-reaction
  frames
- `state_id`: frame role inside a reaction, default `-1`

The conventional state ids are:

| `state_id` | Meaning |
| --- | --- |
| `0` | Reactant |
| `1` | Transition state |
| `2` | Product |

Override input XYZ metadata keys during preprocessing with:

```bash
equitrain-preprocess \
    --source-id-key source_id \
    --reaction-id-key reaction_id \
    --state-id-key state_id \
    ...
```

## Batching Rules

Reaction-relative losses require complete reaction groups in each Torch batch.
Equitrain keeps reaction groups atomic during batching so the relative losses
are computed on complete groups. They are averaged once per reaction, not once
per frame.

Incomplete groups are skipped for the missing relative loss. If no complete
groups are available for a requested relative loss, training raises an error.

## Limitations

Reaction-relative losses are currently Torch-only. The JAX backend rejects
non-zero `--barrier-weight` or `--reaction-energy-weight`.

The weighted sampler is also incompatible with reaction-relative losses. Disable
`--weighted-sampler` when using these targets.
