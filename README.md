# SCARF - PyTorch

This repo reproduces the SCARF (Self-Supervised Contrastive Learning Using Random Feature Corruption) framework for self-supervised learning with tabular data.

*Authors: Dara Bahri, Heinrich Jiang, Yi Tay, Donald Metzler*

*Reference: Bahri, Dara, et al. "Scarf: Self-supervised contrastive learning using random feature corruption." arXiv preprint arXiv:2106.15147 (2021).*

Original paper: https://research.google/pubs/scarf-self-supervised-contrastive-learning-using-random-feature-corruption/

Original repo: --

## Install

Clone this repository, create a new Conda environment and 

```bash
git clone https://github.com/chris-santiago/scarf.git
conda env create -f environment.yml
cd scarf
pip install -e .
```

## Use

### Prerequisites

#### Hydra

This project uses [Hydra](https://hydra.cc/docs/intro/) for managing configuration CLI arguments. See `scarf/conf` for full
configuration details.

#### Task

This project uses [Task](https://taskfile.dev/) as a task runner. Though the underlying Python
commands can be executed without it, we recommend [installing Task](https://taskfile.dev/installation/)
for ease of use. Details located in `Taskfile.yml`.

#### Current commands

```bash
> task -l
task: Available tasks for this project:
* check-config:       Check Hydra configuration
* compare:            Compare using linear baselines
* train:              Train a model
* wandb:              Login to Weights & Biases
```

Example: Train model and for `adult-income` dataset experiment

*The `--` forwards CLI arguments to Hydra.*

```bash
task train -- experiment=income
```

#### PDM

This project was built using [this cookiecutter](https://github.com/chris-santiago/cookie) and is
setup to use [PDM](https://pdm.fming.dev/latest/) for dependency management, though it's not required
for package installation.

#### Weights and Biases

This project is set up to log experiment results with [Weights and Biases](https://wandb.ai/). It
expects an API key within a `.env` file in the root directory:

```toml
WANDB_KEY=<my-super-secret-key>
```

Users can configure different logger(s) within the `conf/trainer/default.yaml` file.
