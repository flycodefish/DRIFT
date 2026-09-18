# DE-ASFL

DE-ASFL is a research implementation of decentralized, hierarchical federated
learning under device, network, and data heterogeneity. The repository contains
the framework, baselines, experiment configurations, launchers, and regression
tests needed to run the code. Datasets, trained models, experiment outputs,
manuscripts, and review materials are intentionally not included.

## Repository layout

```text
baselines/   Baseline decentralized and hierarchical FL implementations
deasfl/      Election, clustering, aggregation, and DE-ASFL runtime
fl_core/     Shared data, runtime, monitoring, and simulation utilities
configs/     Dataset, method, and scenario configurations
scripts/     Training, experiment-matrix, analysis, and verification entrypoints
tests/       Unit and regression tests
```

## Environment

The frozen `environment.yml` matches the release validation environment:

- Python 3.10.20
- PyTorch 2.8.0 and torchvision 0.23.0
- CUDA runtime 12.8
- Ray 2.51.1

```bash
conda env create -f environment.yml
conda activate de-asfl
```

The validated GPU host used NVIDIA RTX 4090 GPUs with driver 550.127.05. CPU
unit tests do not require a GPU. Other compatible CUDA GPUs may be used.

## Data

The default data root is `./data` and can be overridden by launcher options.

- CIFAR-10/CIFAR-100 are downloaded by torchvision when absent.
- NEU-DET is downloaded through kagglehub when absent.
- UCI HAR must be downloaded separately and extracted as
  `data/har/UCI HAR Dataset/`.

Datasets are excluded from Git.

## Quick validation

Run the complete CPU-compatible regression suite:

```bash
python -m pytest -q
```

Run a one-round CIFAR-10 DE-ASFL check:

```bash
python scripts/test_30_clients.py \
  --config configs/cifar10/config_de_asfl_test.yaml \
  --rounds 1 \
  --data-dir ./data
```

Run the FedAvg baseline:

```bash
python scripts/run_fl.py \
  --config configs/cifar10/config_fedavg_test.yaml \
  --data-dir ./data
```

## Reproducible experiment matrices

`scripts/run_experiment_matrix.py` freezes data partitions, initial models,
physical graphs, resources, links, and failure traces for aligned comparisons.

```bash
python scripts/run_experiment_matrix.py \
  --run-smoke \
  --dataset cifar10 \
  --model resnet18 \
  --scenario reproducibility_smoke \
  --scenario-spec configs/scenarios/e2_static_tradeoff.json \
  --template-dir configs/cifar10 \
  --output-root outputs/reproducibility_smoke \
  --methods fedavg,drift \
  --seeds 42 \
  --num-clients 5 \
  --rounds 1 \
  --local-epochs 1 \
  --batch-size 32 \
  --partition dirichlet \
  --alpha 0.5 \
  --data-dir ./data
```

The centralized matched-cluster and JSD-quantile assignment modes are
experimental controls. They are opt-in and are not part of the deployable
distributed protocol. Cluster diagnostics and cluster-personalized evaluation
are also disabled by default.

## Outputs

Generated datasets, frozen inputs, checkpoints, logs, metrics, and figures are
written below `outputs/` and are ignored by Git.

## License

This project is released under the MIT License. See `LICENSE`.
