# DRIFT: Decentralized Routing-based Intelligent Federated Topology Learning in Dual Heterogeneous Edge Computing Networks

A dual-aware routing framework for heterogeneous decentralized federated learning.

## Overview

DE-ASFL addresses the dual heterogeneity challenge (device + data) in decentralized FL through two core components:

1. **Resource-Aware Distributed Election** — Trigger-time-based mechanism that elects the highest-capability node as the global aggregator in a mesh topology.
2. **Distribution-Aware Adaptive Clustering** — JSD-based clustering that organizes nodes with similar label distributions into tree-structured clusters for hierarchical aggregation.

## Project Structure

```
├── fl_core/          # Base FL framework (FedAvg, Gossip, DTD, topology, monitor)
├── deasfl/           # DE-ASFL extension (election, clustering, hierarchical aggregation)
├── configs/          # Experiment configuration files (YAML)
│   ├── cifar10/
│   ├── cifar100/
│   ├── neu/
│   └── har/
├── scripts/          # Experiment entry points and batch runners
├── figures/          # Paper figure generation scripts and outputs
├── paper/            # Paper manuscript and experiment notes
├── outputs/          # Experiment outputs (git-ignored)
└── data/             # Datasets (git-ignored, see below)
```

## Installation

```bash
conda env create -f environment.yml
conda activate fl_env
```

## Datasets

Download datasets to the `data/` directory:

- **CIFAR-10 / CIFAR-100**: Auto-downloaded by torchvision
- **NEU-DET**: Auto-downloaded by kagglehub
- **UCI HAR**: [Download](https://archive.ics.uci.edu/ml/machine-learning-databases/00240/) → `data/har/UCI HAR Dataset/`

## Quick Start

```bash
# DE-ASFL on CIFAR-10 (30 clients, Dirichlet α=0.1)
python scripts/test_30_clients.py --config configs/cifar10/config_de_asfl.yaml

# FedAvg baseline
python scripts/run_fl.py --config configs/cifar10/config_fedavg.yaml

# Gossip baseline
python scripts/run_fl.py --config configs/cifar10/config_gossip.yaml

# DTD baseline
python scripts/run_fl.py --config configs/cifar10/config_dtd_cifar10_alpha01.yaml
```

## Citation

```bibtex
@article{deasfl2026,
  title   = {Towards Robust Decentralized Federated Learning: A Dual-Aware Routing Framework and Its Industrial Validation},
  journal = {},
  year    = {2026},
}
```

## License

TBD
