# Equitrain: A Unified Framework for Training and Fine-tuning Machine Learning Interatomic Potentials

Equitrain is an open-source software package designed to simplify the training and fine-tuning of machine learning universal interatomic potentials (MLIPs). Equitrain addresses the challenges posed by the diverse and often complex training codes specific to each MLIP by providing a unified and efficient framework. This allows researchers to focus on model development rather than implementation details.

---

## Key Features

- **Unified Framework**: Train and fine-tune MLIPs using a consistent interface.
- **Flexible Model Wrappers**: Support for different MLIP architectures through model-specific wrappers.
- **Efficient Preprocessing**: Automated preprocessing with options for computing statistics and managing data.
- **GPU/Node Scalability**: Seamless integration with multi-GPU and multi-node environments using `accelerate`.
- **Extensive Resources**: Includes scripts for dataset preparation, initial model setup, and training workflows.

---

## Installation

Install Equitrain via pip within the `equitrain` package directory:

```bash
pip install .
```

---

## Quickstart Guide

### 1. Preprocessing Data

Preprocess data files to compute necessary statistics and prepare for training:

#### Command Line:
```bash
equitrain-preprocess \
    --train_file="data-train.xyz" \
    --valid_file="data-valid.xyz" \
    --compute_statistics \
    --atomic-energies="average" \
    --output-dir="data/"
```

#### Python Script:
```python
from equitrain import get_args_parser_preprocess, preprocess

def test_preprocess():
    args = get_args_parser_preprocess().parse_args()
    args.train_file = 'data.xyz'
    args.valid_file = 'data.xyz'
    args.output_dir = 'test_preprocess/'
    args.compute_statistics = True
    args.atomic_energies = "average"
    args.r_max = 4.5

    preprocess(args)

if __name__ == "__main__":
    test_preprocess()
```

---

### 2. Training a Model

Train a model using the prepared dataset and specify the MLIP wrapper:

#### Command Line:
```bash
equitrain \
    --train-file data/train.h5 \
    --valid-file data/valid.h5 \
    --statistics-file data/statistics.json \
    --output-dir result \
    --model mace.model \
    --model-wrapper 'mace'
```

#### Python Script:
```python
from equitrain import get_args_parser_train, train
from equitrain.utility_test import MaceWrapper

def test_train_mace():
    args = get_args_parser_train().parse_args()
    args.train_file = 'data/train.h5'
    args.valid_file = 'data/valid.h5'
    args.statistics_file = 'data/statistics.json'
    args.output_dir = 'test_train_mace'
    args.model = MaceWrapper(args)
    args.epochs = 10
    args.batch_size = 64
    args.lr = 0.01
    args.verbose = 2
    args.tqdm = True

    train(args)

if __name__ == "__main__":
    test_train_mace()
```

---

### 3. Making Predictions

Use a trained model to make predictions on new data:

#### Python Script:
```python
from equitrain import get_args_parser_predict, predict
from equitrain.utility_test import MaceWrapper

def test_mace_predict():
    args = get_args_parser_predict().parse_args()
    args.predict_file = 'data/valid.h5'
    args.statistics_file = 'data/statistics.json'
    args.batch_size = 5
    args.model = MaceWrapper(args)

    energy_pred, forces_pred, stress_pred = predict(args)

    print(energy_pred)
    print(forces_pred)
    print(stress_pred)

if __name__ == "__main__":
    test_mace_predict()
```

---

## Advanced Features

### Multi-GPU and Multi-Node Training
Equitrain supports multi-GPU and multi-node training using `accelerate`. Example scripts are available in the `resources/training` directory.

### Dataset Preparation
Equitrain provides scripts for downloading and preparing popular datasets such as Alexandria and MPTraj. These scripts can be found in the `resources/data` directory.

### Pretrained Models
Initial model examples and configurations can be accessed in the `resources/models` directory.

---

## License
Equitrain is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## Acknowledgments
We thank the open-source community for supporting the development of machine learning interatomic potentials and related tools.
