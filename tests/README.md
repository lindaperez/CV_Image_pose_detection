# Testing Guide

This folder contains the lightweight `unittest` coverage for the project's script-level logic.

The tests are organized by purpose rather than by notebook stage:

- `data_prep`
  - Countix label mapping and manifest normalization helpers
- `evaluation`
  - bootstrap confidence intervals
  - experiment registry and routed prediction assembly
- `review`
  - hard-case review manifest tools
  - hard-case review local server
- `runtime`
  - squat offline runtime
  - live squat runtime logic

## Quick Commands

From the repo root:

```bash
source .venv/bin/activate
python tests/run_tests.py --list
python tests/run_tests.py data_prep
python tests/run_tests.py review
python tests/run_tests.py runtime
python tests/run_tests.py all
```

For the full legacy discovery run:

```bash
source .venv/bin/activate
python -m unittest discover -s tests -p 'test_*.py'
```

## Scope

These tests are meant to protect the script-level contracts in the repo:

- file/path resolution
- manifest building
- summary generation
- runtime helper behavior
- routing and registry updates

They do **not** replace the Colab experiment runs. The notebooks remain the empirical validation layer for model training and benchmark results.
