# Repository Guidelines

## Project Structure & Module Organization

This repository contains OAT, a Python 3.10 project for ordered action tokenization and policy training. Core source lives in `oat/`: shared utilities in `oat/common`, datasets in `oat/dataset`, environment runners in `oat/env_runner`, policies in `oat/policy`, tokenizers in `oat/tokenizer`, and Hydra workspaces in `oat/workspace`. Experiment configs are in `oat/config/*.yaml`. Operational scripts live in `scripts/`, Slurm launchers in `slurm/`, analysis utilities and generated plots in `analysis_scripts/` and `analysis_results/`, and vendored LIBERO code in `third_party/LIBERO`. Keep large datasets, checkpoints, and run outputs under `data/`, `tok_ckpt/`, or `output/`, not beside source modules.

## Build, Test, and Development Commands

- `uv sync`: install the project and workspace dependencies from `pyproject.toml` and `uv.lock`.
- `./setup_env.sh --cuda 12.9`: create a CUDA-ready virtual environment and install OAT/LIBERO editable; use this on fresh GPU machines.
- `pip install -e .`: install OAT editable inside an already prepared environment.
- `python scripts/run_workspace.py --config-name=train_oattok`: launch a Hydra training workspace; swap the config name with any file in `oat/config` without the `.yaml` suffix.
- `python scripts/smoke_test_so3_action_chunk_aug.py`: run the SO(3) augmentation smoke checks.

## Coding Style & Naming Conventions

Use 4-space indentation, type hints where they clarify tensor shapes or config objects, and concise docstrings for non-obvious math. Follow existing naming: modules and functions use `snake_case`, classes use `PascalCase`, configs use `train_<component>.yaml`, and policy/tokenizer variants include the feature name, for example `oat_policy_with_enriched_past.py`. Prefer Hydra/OmegaConf configuration over hard-coded experiment parameters.

## Testing Guidelines

Tests currently live as top-level `test_*.py` files. Add focused pytest-style functions named `test_<behavior>` for tokenizer, augmentation, and trajectory changes. Run `pytest test_a2lex.py test_zhill.py test_trajectory_smoothness.py` before touching quantizers or evaluation logic, and add small CPU-only checks when GPU coverage is impractical.

## Commit & Pull Request Guidelines

Recent commits are short, lowercase summaries such as `past2next` and `a2lex tokenization`. Keep commits focused and use imperative or descriptive one-line subjects. Pull requests should describe the changed model/tokenizer/config path, list commands run, call out dataset or checkpoint assumptions, and include plots or eval summaries when behavior changes.

## Security & Configuration Tips

Do not commit secrets, WandB credentials, local dataset paths, or bulky generated artifacts. Put machine-specific paths in Hydra overrides or local configs, and document required checkpoints or datasets in the PR.
