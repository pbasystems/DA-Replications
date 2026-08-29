# Hydra Configuration & Experiment Guide

This guide explains how to use the Hydra configuration system to run, configure, sweep, and extend experiments in this repository.

---

## 1. Quick Start

Run the unified entrypoint `run.py`:

```bash
# Run default experiment (LSTM-DANN single trial on FD001 -> FD002)
python run.py

# Run pre-configured experiment recipes
python run.py experiment=lstm_dann_cmapss_single
python run.py experiment=lstm_dann_cmapss_10trials
python run.py experiment=ops_dann_ncmapss_single
python run.py experiment=ops_dann_ncmapss_10trials
```

---

## 2. Configuration Structure

All configuration files reside in `configs/`:

```
configs/
├── config.yaml                         # Global defaults & Hydra settings
├── experiment/                         # Complete experiment recipes
│   ├── lstm_dann_cmapss_single.yaml    # Single trial LSTM-DANN
│   ├── lstm_dann_cmapss_10trials.yaml  # 10-trial benchmark evaluation
│   ├── ops_dann_ncmapss_single.yaml    # Single trial OPS-DANN
│   └── ops_dann_ncmapss_10trials.yaml  # 10-trial benchmark evaluation
├── dataset/                            # Dataset specifications
│   ├── cmapss.yaml                     # C-MAPSS dataset settings (FD001-FD004)
│   └── n_cmapss.yaml                   # N-CMAPSS dataset settings (FC1-FC3)
├── model/                              # Model architectures
│   ├── lstm_dann.yaml                  # LSTM_DANN module parameters
│   └── ops_dann_hard.yaml              # OPSDANNHard module parameters
├── trainer/                            # Training loop parameters
│   ├── lstm_dann_trainer.yaml          # 2-stage alternating trainer
│   └── ops_dann_trainer.yaml           # Phase-averaged loss trainer
├── optimizer/                          # Optimizer & learning rate schedules
│   ├── sgd_lstm_dann.yaml              # Multi-group SGD + MultiStepLR
│   └── sgd_ops_dann.yaml               # SGD with momentum + polynomial decay
├── loss/                               # Loss functions & scoring
│   ├── lstm_dann_loss.yaml             # Lp regression loss + domain classification loss + Score
│   └── ops_dann_loss.yaml              # RUL loss + phase-averaged domain loss + Score
├── logger/                             # Metric logging & tracking
│   ├── console.yaml                    # Console-only logging
│   ├── wandb.yaml                      # Weights & Biases experiment tracking
│   └── disabled.yaml                   # Silent mode for testing & sweeps
└── paths/                              # Project paths
    └── default.yaml                    # Root and Data directory paths
```

---

## 3. Command-Line Overrides

Hydra allows overriding any parameter directly from the command line:

### A. Changing Domains & Datasets
```bash
# Evaluate on FD003 -> FD004 instead of FD001 -> FD002
python run.py experiment=lstm_dann_cmapss_single dataset.source_fd=FD003 dataset.target_fd=FD004

# Evaluate OPS-DANN on FC2 -> FC3
python run.py experiment=ops_dann_ncmapss_single dataset.source_fc=2 dataset.target_fc=3
```

### B. Modifying Hyperparameters
```bash
# Change feature size, dropout, and learning rates
python run.py experiment=lstm_dann_cmapss_single model.f_size=64 model.lstm_dropout=0.3 optimizer.lr_source_reg=0.005

# Adjust training epochs and batch size
python run.py experiment=lstm_dann_cmapss_single trainer.epochs=100 dataset.batch_size=128
```

### C. Controlling Multi-Trial Evaluation
```bash
# Run 5 trials instead of 10
python run.py experiment=lstm_dann_cmapss_10trials trials=5
```

### D. Enabling Weights & Biases (W&B) Tracking
```bash
python run.py experiment=lstm_dann_cmapss_single logger=wandb logger.wandb_project="My-RUL-Study" logger.wandb_run_name="lstm-f64-run1"
```

---

## 4. Hyperparameter Sweeps (Hydra Multirun)

You can run automated grid searches across any combination of parameters using `-m`:

```bash
# Sweep over learning rates and feature dimensions
python run.py -m experiment=lstm_dann_cmapss_single optimizer.lr_source_reg=0.01,0.001 model.f_size=16,32,64

# Sweep over domain pairs
python run.py -m experiment=lstm_dann_cmapss_single dataset.source_fd=FD001 dataset.target_fd=FD002,FD003,FD004
```

---

## 5. How to Add a New Model / Dataset / Experiment

### Step 1: Add Component Configs
- Add `configs/model/my_new_model.yaml`:
  ```yaml
  _target_: my_module.MyModel
  param1: 64
  param2: 0.1
  ```
- Add `configs/dataset/my_dataset.yaml` with required dataset parameters.

### Step 2: Implement or Wire Pipeline
- In `src/pipelines/`, either reuse an existing pipeline or subclass `BasePipeline` in `my_pipeline.py`.
- Register the new pipeline in `src/pipelines/__init__.py` under `PIPELINE_REGISTRY`.

### Step 3: Create an Experiment Recipe
- Add `configs/experiment/my_experiment.yaml`:
  ```yaml
  # @package _global_
  defaults:
    - override /dataset: my_dataset
    - override /model: my_new_model
    - override /trainer: my_trainer
    - override /optimizer: my_optimizer
    - override /loss: my_loss
    - override /logger: console

  seed: 42
  trials: 1
  pipeline: my_pipeline
  task_name: "my_experiment"
  ```

Run it immediately with:
```bash
python run.py experiment=my_experiment
```
