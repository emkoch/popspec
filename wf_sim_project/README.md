# WF-Sim-Tools: High-Performance Wright-Fisher Simulations

A robust, C-accelerated Python package for simulating Wright-Fisher populations with recombination, selection (polygenic & underdominant), and complex demographies.

Built for high-performance clusters (O2) with a user-friendly Python interface.

## Features

- **Fast:** Core simulation engine written in C (GSL-optimized).
- **Flexible:** Supports time-varying demography via input files.
- **Easy:** Pure Python interface returns Pandas DataFrames directly.
- **Reproducible:** Strict random seeding and GSL RNG usage.

## Installation

### Prerequisites
- GSL (GNU Scientific Library)
- Python 3.8+
- GCC Compiler

### Install
```bash
# Clone the repository
git clone [https://github.com/emkoch/popspec.git](https://github.com/emkoch/popspec.git)

# Navigate to the project subdirectory
cd popspec/wf_sim_project

# Install in editable mode
pip install -e .
```

## Usage

### 1. Command Line Interface (CLI)

Run simulations directly from the terminal:

```bash
# Run 100 sims with N=1000, theta=50, saving to results.tsv
wf-sim --n-sims 100 --pop-size 1000 --theta 50 --output results.tsv

# Run with specific demography file
wf-sim --n-sims 100 --pop-size 1000 --demography-file demo_Ns.txt
```

### 2. Python API

Integrate directly into your analysis notebooks:

```python
import pandas as pd
from wf_sim_tools.api import run_sims

# Run simulations
df = run_sims(
    n_sims=1000,
    pop_size=5000,
    theta=50.0,
    beta1=0.01,
    polygenic=True
)

# Analyze immediately
print(f"Mean Fixation: {df['N11'].mean()}")
```

## Parameters

| Argument | Description | Default |
|----------|-------------|---------|
| `--n-sims` | Number of simulations to run | **Required** |
| `--pop-size` | Effective Population Size (N) | **Required** |
| `--theta` | Mutation rate (2*Ne*mu) | `50.0` |
| `--rho` | Recombination rate | `50.0` |
| `--run-mult` | Duration multiplier (simulates for `run_mult * N` generations) | `2` |
| `--beta1` | Selection coefficient for derived allele at locus 1 | `0.0` |
| `--beta2` | Selection coefficient for derived allele at locus 2 | `0.0` |
| `--symmetry` | Probability that selection is positive (vs negative) | `0.5` |
| `--seed` | Random seed offset | `123` |
| `--polygenic` | Flag: Use polygenic selection (vs quadratic) | `False` |
| `--demography-file` | Path to text file containing integer list of pop sizes | `None` |
| `--output` / `-o` | Output file path (if ignored, prints to stdout) | `None` |
