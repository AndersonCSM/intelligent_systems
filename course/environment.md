# Environment Setup Guide

This document is the single source of truth for Python environment setup in this repository.

## Supported Approaches

1. Python `venv` (built-in, lightweight);
2. Miniconda environment (recommended for ML/Data Science).

## Python venv

### Create

```bash
cd /path/to/project
python -m venv venv
```

### Activate

```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Deactivate

```bash
deactivate
```

## Miniconda (recommended)

### Create named environment

```bash
conda create -n intelligent_systems python=3.11 numpy pandas scikit-learn matplotlib jupyter tensorflow -y
```

### Activate

```bash
conda activate intelligent_systems
```

### Install dependencies

```bash
conda install <package-name>
# or
pip install <package-name>
```

### Export environment

```bash
conda env export -n intelligent_systems --no-builds > environment.yml
```

### Remove environment

```bash
conda env remove -n intelligent_systems
```

## Notes

- Use `venv` for simple Python-only projects;
- Use Miniconda for ML and heavier scientific stacks;
- Do not commit environment folders to Git.
