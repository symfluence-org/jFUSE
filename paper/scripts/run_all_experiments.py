#!/usr/bin/env python3
"""
Run all experiments for the jFUSE paper.

This script orchestrates all experiments and generates results for the paper.
Results are saved to paper/results/ and figures to paper/figures/.

Usage:
    python run_all_experiments.py [--exp EXP_NUM] [--gpu] [--quick]
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Ensure we're in the right directory
SCRIPT_DIR = Path(__file__).parent.absolute()
PAPER_DIR = SCRIPT_DIR.parent
PROJECT_DIR = PAPER_DIR.parent
RESULTS_DIR = PAPER_DIR / "results"
FIGURES_DIR = PAPER_DIR / "figures"

# Create output directories
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def run_script(script_name: str, args: list = None, env: dict = None):
    """Run a Python script with optional arguments and environment."""
    script_path = SCRIPT_DIR / script_name
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)

    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, env=full_env, cwd=str(SCRIPT_DIR))
    if result.returncode != 0:
        print(f"ERROR: {script_name} failed with return code {result.returncode}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Run jFUSE paper experiments")
    parser.add_argument(
        "--exp", type=int, choices=[1, 2, 3, 4], help="Run specific experiment (1-4)"
    )
    parser.add_argument("--gpu", action="store_true", help="Enable GPU acceleration")
    parser.add_argument(
        "--quick", action="store_true", help="Run quick versions of experiments for testing"
    )
    args = parser.parse_args()

    # Set up environment
    env = {}
    if args.gpu:
        env["JAX_PLATFORM_NAME"] = "gpu"
    else:
        env["JAX_PLATFORM_NAME"] = "cpu"

    script_args = []
    if args.quick:
        script_args.append("--quick")

    experiments = {
        1: "exp1_model_fidelity.py",
        2: "exp2_gradient_verification.py",
        3: "exp3_calibration_efficiency.py",
        4: "exp4_distributed_calibration.py",
    }

    if args.exp:
        # Run specific experiment
        scripts_to_run = [experiments[args.exp]]
    else:
        # Run all experiments
        scripts_to_run = list(experiments.values())

    print("jFUSE Paper Experiments")
    print("=======================")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Figures directory: {FIGURES_DIR}")
    print(f"GPU enabled: {args.gpu}")
    print(f"Quick mode: {args.quick}")
    print(f"Experiments to run: {scripts_to_run}")

    # Run experiments
    success = True
    for script in scripts_to_run:
        if not run_script(script, script_args, env):
            success = False

    # Generate plots
    if success:
        print("\nGenerating figures...")
        run_script("plot_results.py", env=env)

    # Summary
    print("\n" + "=" * 60)
    if success:
        print("All experiments completed successfully!")
        print(f"Results saved to: {RESULTS_DIR}")
        print(f"Figures saved to: {FIGURES_DIR}")
    else:
        print("Some experiments failed. Check output above for details.")
    print("=" * 60)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
