#!/usr/bin/env python3
"""
Generate figures for the jFUSE paper from experiment results.

Outputs:
    - figures/calibration_convergence.pdf
    - figures/model_fidelity.pdf
    - figures/gradient_verification.pdf
    - figures/distributed_results.pdf
"""

from pathlib import Path

import numpy as np
import pandas as pd

PAPER_DIR = Path(__file__).parent.parent
RESULTS_DIR = PAPER_DIR / "results"
FIGURES_DIR = PAPER_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Check if matplotlib is available
try:
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Generating placeholder files.")


def plot_calibration_convergence():
    """Plot calibration convergence curves."""
    history_file = RESULTS_DIR / "exp3_convergence_history.csv"

    if not history_file.exists():
        print("  Skipping: exp3_convergence_history.csv not found")
        return

    if not HAS_MATPLOTLIB:
        # Create placeholder
        (FIGURES_DIR / "calibration_convergence.pdf").touch()
        return

    df = pd.read_csv(history_file)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        "jFUSE_Adam": "#1f77b4",
        "jFUSE_LBFGS": "#2ca02c",
        "SCE_UA": "#d62728",
        "DDS": "#ff7f0e",
    }

    for method in df["method"].unique():
        method_df = df[df["method"] == method]
        ax.plot(
            method_df["func_evals"],
            method_df["nse"],
            label=method,
            color=colors.get(method, "gray"),
            linewidth=2,
        )

    # Target lines
    for target in [0.7, 0.8, 0.85]:
        ax.axhline(y=target, color="gray", linestyle="--", alpha=0.5)
        ax.text(100, target + 0.01, f"NSE={target}", fontsize=9, alpha=0.7)

    ax.set_xlabel("Function Evaluations", fontsize=12)
    ax.set_ylabel("Nash-Sutcliffe Efficiency", fontsize=12)
    ax.set_title("Calibration Convergence: Gradient-based vs Derivative-free", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xscale("log")
    ax.set_xlim(1, 25000)
    ax.set_ylim(0.3, 0.95)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "calibration_convergence.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    print("  Generated: calibration_convergence.pdf")


def plot_model_fidelity():
    """Plot model fidelity comparison."""
    fidelity_file = RESULTS_DIR / "exp1_fidelity_by_structure.csv"

    if not fidelity_file.exists():
        print("  Skipping: exp1_fidelity_by_structure.csv not found")
        return

    if not HAS_MATPLOTLIB:
        (FIGURES_DIR / "model_fidelity.pdf").touch()
        return

    df = pd.read_csv(fidelity_file)
    df = df.dropna()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # RMSE comparison
    ax1 = axes[0]
    x = np.arange(len(df))
    width = 0.35

    ax1.bar(x - width / 2, df["vs_fortran_rmse"], width, label="vs Fortran FUSE", color="#1f77b4")
    ax1.bar(x + width / 2, df["vs_dfuse_rmse"], width, label="vs dFUSE", color="#2ca02c")

    ax1.set_xlabel("Model Configuration", fontsize=12)
    ax1.set_ylabel("RMSE (mm/day)", fontsize=12)
    ax1.set_title("RMSE by Model Structure", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.set_xticks([])

    # Correlation comparison
    ax2 = axes[1]
    ax2.bar(x - width / 2, df["vs_fortran_corr"], width, label="vs Fortran FUSE", color="#1f77b4")
    ax2.bar(x + width / 2, df["vs_dfuse_corr"], width, label="vs dFUSE", color="#2ca02c")

    ax2.set_xlabel("Model Configuration", fontsize=12)
    ax2.set_ylabel("Correlation", fontsize=12)
    ax2.set_title("Correlation by Model Structure", fontsize=14)
    ax2.legend(fontsize=10)
    ax2.set_xticks([])
    ax2.set_ylim(0.7, 1.0)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_fidelity.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    print("  Generated: model_fidelity.pdf")


def plot_gradient_verification():
    """Plot gradient verification results."""
    grad_file = RESULTS_DIR / "exp2_gradient_verification.csv"

    if not grad_file.exists():
        print("  Skipping: exp2_gradient_verification.csv not found")
        return

    if not HAS_MATPLOTLIB:
        (FIGURES_DIR / "gradient_verification.pdf").touch()
        return

    df = pd.read_csv(grad_file)
    df = df.dropna()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Gradient comparison scatter
    ax1 = axes[0]
    ax1.scatter(df["fd_gradient"], df["jax_gradient"], c="#1f77b4", s=100, alpha=0.7)

    # 1:1 line
    lims = [
        min(df["fd_gradient"].min(), df["jax_gradient"].min()) * 1.1,
        max(df["fd_gradient"].max(), df["jax_gradient"].max()) * 1.1,
    ]
    ax1.plot(lims, lims, "k--", alpha=0.5, label="1:1 line")

    for i, row in df.iterrows():
        ax1.annotate(
            row["parameter"], (row["fd_gradient"], row["jax_gradient"]), fontsize=8, alpha=0.7
        )

    ax1.set_xlabel("Finite Difference Gradient", fontsize=12)
    ax1.set_ylabel("JAX Automatic Gradient", fontsize=12)
    ax1.set_title("Gradient Verification", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Relative error bar chart
    ax2 = axes[1]
    colors = ["#2ca02c" if r["status"] == "PASS" else "#d62728" for _, r in df.iterrows()]
    ax2.barh(df["parameter"], df["rel_error_pct"], color=colors)
    ax2.axvline(x=5.0, color="red", linestyle="--", label="5% threshold")

    ax2.set_xlabel("Relative Error (%)", fontsize=12)
    ax2.set_title("Gradient Relative Error by Parameter", fontsize=14)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "gradient_verification.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    print("  Generated: gradient_verification.pdf")


def plot_distributed_results():
    """Plot distributed calibration results."""
    dist_file = RESULTS_DIR / "exp4_distributed_calibration.csv"

    if not dist_file.exists():
        print("  Skipping: exp4_distributed_calibration.csv not found")
        return

    if not HAS_MATPLOTLIB:
        (FIGURES_DIR / "distributed_results.pdf").touch()
        return

    df = pd.read_csv(dist_file)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # NSE by configuration
    ax1 = axes[0]
    bars = ax1.bar(df["configuration"], df["final_nse"], color="#1f77b4")

    ax1.set_xlabel("Parameter Configuration", fontsize=12)
    ax1.set_ylabel("Final NSE", fontsize=12)
    ax1.set_title("Calibration Performance by Configuration", fontsize=14)
    ax1.set_ylim(0.7, 1.0)
    ax1.tick_params(axis="x", rotation=45)

    # Add parameter count labels
    for bar, n_params in zip(bars, df["n_total_params"]):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"n={n_params}",
            ha="center",
            fontsize=9,
        )

    # Time vs parameters scatter
    ax2 = axes[1]
    ax2.scatter(
        df["n_total_params"],
        df["time_s"],
        s=200,
        c=df["final_nse"],
        cmap="RdYlGn",
        vmin=0.7,
        vmax=1.0,
    )

    for i, row in df.iterrows():
        ax2.annotate(
            row["configuration"],
            (row["n_total_params"], row["time_s"]),
            fontsize=9,
            ha="center",
            va="bottom",
        )

    ax2.set_xlabel("Number of Parameters", fontsize=12)
    ax2.set_ylabel("Calibration Time (s)", fontsize=12)
    ax2.set_title("Scalability: Time vs Parameters", fontsize=14)

    cbar = plt.colorbar(ax2.collections[0], ax=ax2)
    cbar.set_label("Final NSE")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "distributed_results.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    print("  Generated: distributed_results.pdf")


def main():
    """Generate all figures."""
    print("Generating paper figures...")
    print(f"  Results directory: {RESULTS_DIR}")
    print(f"  Figures directory: {FIGURES_DIR}")

    plot_calibration_convergence()
    plot_model_fidelity()
    plot_gradient_verification()
    plot_distributed_results()

    print("\nDone!")


if __name__ == "__main__":
    main()
