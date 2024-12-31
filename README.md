# Equitrain: A Unified Framework for Training and Fine-tuning Machine Learning Interatomic Potentials

Equitrain is an open-source software package designed to simplify the training and fine-tuning of machine learning universal interatomic potentials (MLIPs). Equitrain addresses the challenges posed by the diverse and often complex training codes specific to each MLIP by providing a unified and efficient framework. This allows researchers to focus on model development rather than implementation details.


## Installation

```sh
pip install torch==2.3.0 torchvision torchaudio -f https://download.pytorch.org/whl/cu121
pip install torch-cluster torch-scatter -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
pip install -e .
```

## Execution

```sh
    mkdir data

    equitrain-preprocess \
           --train_file="data-train.xyz" \
           --valid_file="data-valid.xyz" \
           --atomic_numbers="[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 89, 90, 91, 92, 93, 94]" \
           --r_max=4.5 \
           --h5_prefix="data/" \
           --compute_statistics \
           --E0s="average" \
           --seed=42

```

Execute a mace model:
```sh
    mkdir result
    equitrain --train-file data/train.h5 --valid-file data/valid.h5  --statistics-file data/statistics.json --output-dir result --model mace.model
```

## Multi-GPU execution

*train_equiformer.py*
```python
from equitrain import get_args_parser
from equitrain import train

def main():

    r = 4.5

    args = get_args_parser().parse_args()
    args.train_file = f'data-r{r}/train.h5'
    args.valid_file = f'data-r{r}/valid.h5'
    args.statistics_file = f'data-r{r}/statistics.json'
    args.output_dir = 'result-test'

    train(args)

if __name__ == "__main__":
    main()
```

```sh
    accelerate launch --num-processes=2 train_equiformer.py
```

## TODO

* Implement stress predictions
