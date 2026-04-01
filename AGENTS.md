# Repository Guidelines

## Project Structure & Module Organization
Core Python entry points live at the repository root. Use `train.py` for training, `eval.py` for planning/evaluation, `jepa.py` for the JEPA world model, `module.py` for network building blocks, and `utils.py` for shared helpers and checkpoint callbacks. Hydra configs are under `config/`: training configs in `config/train/`, dataset variants in `config/train/data/`, and evaluation presets in `config/eval/` with solver and launcher overrides in subfolders. Visual assets such as demo media live in `assets/`.

## Build, Test, and Development Commands
Create the environment with `uv venv --python=3.10 && source .venv/bin/activate`, then install runtime dependencies with `uv pip install stable-worldmodel[train,env]`. Start training with `python train.py data=pusht`; swap `pusht` for another dataset config under `config/train/data/`. Run evaluation with `python eval.py --config-name=pusht.yaml policy=pusht/lewm`, where `policy` is the checkpoint path relative to `$STABLEWM_HOME` and omits `_object.ckpt`. Use `tar --zstd -xvf archive.tar.zst` to unpack dataset or checkpoint archives before placing them under `$STABLEWM_HOME`.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, PEP 8 spacing, and small top-level modules instead of deep package nesting. No formatter or linter config is checked in today, so match surrounding code instead of introducing a new tool-specific style. Use `snake_case` for functions, variables, and Hydra config names (`tworoom.yaml`, `lewm.yaml`), and `PascalCase` for classes (`JEPA`, `ARPredictor`, `ModelObjectCallBack`). Keep tensor shape assumptions explicit in code and prefer direct data-flow over fallback-heavy branching.

## Testing Guidelines
This repository currently has no dedicated `tests/` suite. Treat reproducible smoke runs as the minimum bar: launch one training config and one evaluation config relevant to your change. For config-only edits, verify the target command resolves correctly through Hydra. If you add automated tests, place them in a new `tests/` directory and name files `test_<feature>.py`.

## Commit & Pull Request Guidelines
Recent history favors short, imperative subjects with occasional conventional prefixes, for example `fix: use proj.device instead of hardcoded cuda`. Keep commits focused and describe the user-visible change first. Pull requests should include: the affected training/eval path, any required config or dataset assumptions, linked issues, and sample metrics or screenshots when behavior changes.

## Configuration & Data Tips
Set `STABLEWM_HOME` explicitly when datasets or checkpoints are stored outside `~/.stable-wm/`. Before training, update the WandB `entity` and `project` in `config/train/lewm.yaml`. Do not commit local cache paths, secrets, or large extracted `.h5` / checkpoint artifacts.
