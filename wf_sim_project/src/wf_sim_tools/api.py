import subprocess
import sys
import os
import io
import shutil
from pathlib import Path
import pandas as pd

# Paths
PACKAGE_DIR = Path(__file__).parent
BINARY_DIR = PACKAGE_DIR / "bin"
BINARY_NAME = "wf_sim"
BINARY_PATH = BINARY_DIR / BINARY_NAME
SOURCE_FILE = BINARY_DIR / "sim.c"

def get_gsl_flags():
    """
    Attempts to find GSL compilation flags dynamically.
    1. Tries 'gsl-config' (standard on Linux/Mac).
    2. Tries O2 Cluster hardcoded defaults.
    """
    # 1. Try asking the system where GSL is
    gsl_config = shutil.which("gsl-config")
    if gsl_config:
        try:
            cflags = subprocess.check_output([gsl_config, "--cflags"], text=True).strip().split()
            libs = subprocess.check_output([gsl_config, "--libs"], text=True).strip().split()
            return cflags, libs
        except subprocess.CalledProcessError:
            pass # Fall through to fallback

    # 2. Fallback: O2 Cluster Defaults
    print("⚠️  'gsl-config' not found. Using O2 Cluster defaults.", file=sys.stderr)
    o2_cflags = ["-I/n/app/gsl/2.8-gcc-14.2.0/include"]
    o2_libs = [
        "-L/n/app/gsl/2.8-gcc-14.2.0/lib",
        "-lgsl", "-lgslcblas", "-lm",
        "-Wl,-rpath,/n/app/gsl/2.8-gcc-14.2.0/lib"
    ]
    return o2_cflags, o2_libs

def check_and_compile():
    """Checks for binary, compiles if missing."""
    if not BINARY_PATH.exists():
        if not SOURCE_FILE.exists():
            raise FileNotFoundError(f"Source file not found at {SOURCE_FILE}")

        print(f"🔨 Compiling C binary...", file=sys.stderr)
        
        cflags, libs = get_gsl_flags()

        cmd = ["gcc", "-std=c99", "-O3"] + cflags + [str(SOURCE_FILE), "-o", str(BINARY_PATH)] + libs
        
        # Print the command for debugging
        # print(" ".join(cmd), file=sys.stderr) 

        subprocess.run(cmd, check=True)

def run_sims(n_sims: int, pop_size: int, 
             theta: float = 50.0, rho: float = 50.0, 
             run_mult: int = 2, beta1: float = 0.0, beta2: float = 0.0, 
             symmetry: float = 0.5, seed: int = 123, 
             polygenic: bool = False, demography_file: str = None) -> pd.DataFrame:
    """
    Runs the C simulation and returns results as a Pandas DataFrame.
    
    Args:
        n_sims (int): Number of simulations to run.
        pop_size (int): Effective population size (N).
        theta (float): Mutation rate parameter (2*Ne*mu).
        rho (float): Recombination rate parameter.
        run_mult (int): Multiplier for generation run time (runs for run_mult * N gens).
        beta1 (float): Selection coefficient 1.
        beta2 (float): Selection coefficient 2.
        symmetry (float): Probability selection is positive vs negative (0.5 = random).
        seed (int): Random seed offset.
        polygenic (bool): If True, uses underdominant selection. If False, quadratic.
        demography_file (str): Path to file containing population sizes (optional).

    Returns:
        pd.DataFrame: Simulation results.
    """
    # Ensure binary exists
    check_and_compile()

    # Prepare Arguments
    demo_arg = demography_file if demography_file else "0"
    
    c_args = [
        str(n_sims), str(pop_size), str(theta), str(rho),
        str(run_mult), str(beta1), str(beta2),
        str(symmetry), str(seed), str(int(polygenic)),
        demo_arg
    ]

    cmd = [str(BINARY_PATH)] + c_args

    # Run and capture output
    # We capture stdout because the C program prints the TSV data there
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    # Use io.StringIO to treat the string output like a file object
    csv_data = io.StringIO(result.stdout)
    
    # Read into Pandas (C output is tab-separated)
    df = pd.read_csv(csv_data, sep="\t")
    
    return df