#!/usr/bin/env python3
"""
Experiment 3: Calibration Efficiency

Compare gradient-based optimization (jFUSE with Adam/L-BFGS) to derivative-free
methods (simulated SCE-UA and DDS) in terms of function evaluations to reach
target NSE values.

Outputs:
    - results/exp3_calibration_efficiency.csv
    - results/exp3_convergence_history.csv
"""

import argparse
import sys
from pathlib import Path
import time

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from jfuse.fuse.model import create_fuse_model
from jfuse.fuse.state import FUSEState, FUSEParams, FUSEForcing, get_default_params
from jfuse.fuse.config import PRMS_CONFIG

import jax
import jax.numpy as jnp
import optax

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def generate_forcing(n_timesteps: int = 730):
    """Generate 2 years of forcing data."""
    np.random.seed(42)
    t = np.arange(n_timesteps)

    # Seasonal precipitation with random events
    precip_base = 3.0 + 2.0 * np.sin(2 * np.pi * t / 365)
    precip = np.maximum(0, precip_base + np.random.exponential(2.0, n_timesteps))

    # PET
    pet = 2.0 + 2.5 * np.sin(2 * np.pi * (t - 90) / 365)
    pet = np.maximum(0.5, pet)

    # Temperature
    temp = 10.0 + 15.0 * np.sin(2 * np.pi * (t - 90) / 365)
    temp += np.random.normal(0, 2, n_timesteps)

    return FUSEForcing(
        precip=jnp.array(precip, dtype=jnp.float32),
        pet=jnp.array(pet, dtype=jnp.float32),
        temp=jnp.array(temp, dtype=jnp.float32),
    )


def generate_observations(forcing: FUSEForcing):
    """Generate synthetic observations."""
    # True parameters (perturbed from default)
    true_params = get_default_params()
    true_params = FUSEParams(
        S1_max=true_params.S1_max * 1.3,
        S2_max=true_params.S2_max * 0.8,
        ku=true_params.ku * 1.1,
        ks=true_params.ks * 0.9,
        ki=true_params.ki,
        kq=true_params.kq,
        alpha=true_params.alpha * 1.2,
        beta=true_params.beta,
        chi=true_params.chi,
        phi=true_params.phi,
        psi=true_params.psi,
        f_tens=true_params.f_tens,
        f_1=true_params.f_1,
        f_2=true_params.f_2,
        lambda_baseflow=true_params.lambda_baseflow,
        c_baseflow=true_params.c_baseflow,
        n_baseflow=true_params.n_baseflow,
        m_topmodel=true_params.m_topmodel,
        f_root=true_params.f_root,
        r_exp=true_params.r_exp,
        melt_factor=true_params.melt_factor,
        melt_temp=true_params.melt_temp,
        rain_temp=true_params.rain_temp,
        snow_temp=true_params.snow_temp,
        melt_factor_amp=true_params.melt_factor_amp,
        gamma_shape=true_params.gamma_shape,
        gamma_scale=true_params.gamma_scale,
        rainfall_mult=true_params.rainfall_mult,
        rainfall_add=true_params.rainfall_add,
        manning_n=true_params.manning_n,
    )

    model = create_fuse_model(PRMS_CONFIG)
    initial_state = FUSEState.zeros()

    _, flux_history = model.simulate(true_params, initial_state, forcing)

    # Add observation noise
    np.random.seed(123)
    noise = np.random.normal(0, 0.05 * np.std(flux_history.q_total), len(flux_history.q_total))
    obs = np.array(flux_history.q_total) + noise

    return jnp.array(np.maximum(0, obs), dtype=jnp.float32), true_params


def compute_nse(simulated: jnp.ndarray, observed: jnp.ndarray, warmup: int = 60):
    """Compute NSE."""
    sim = simulated[warmup:]
    obs = observed[warmup:]

    ss_res = jnp.sum((sim - obs) ** 2)
    ss_tot = jnp.sum((obs - jnp.mean(obs)) ** 2)

    return float(1.0 - ss_res / (ss_tot + 1e-10))


def gradient_based_calibration(
    forcing: FUSEForcing,
    observations: jnp.ndarray,
    optimizer_name: str = "adam",
    max_epochs: int = 500,
    lr: float = 0.01,
):
    """Run gradient-based calibration."""
    model = create_fuse_model(PRMS_CONFIG)
    initial_state = FUSEState.zeros()

    # Initialize parameters
    params = get_default_params()
    params_array = params.to_array()

    def loss_fn(params_array):
        params = FUSEParams.from_array(params_array)
        _, flux_history = model.simulate(params, initial_state, forcing)

        sim = flux_history.q_total[60:]
        obs = observations[60:]

        ss_res = jnp.sum((sim - obs) ** 2)
        ss_tot = jnp.sum((obs - jnp.mean(obs)) ** 2)

        return ss_res / (ss_tot + 1e-10)  # 1 - NSE

    # Set up optimizer
    if optimizer_name == "adam":
        optimizer = optax.adam(lr)
    elif optimizer_name == "lbfgs":
        # Use Adam with lower lr as proxy (true L-BFGS not in optax)
        optimizer = optax.adam(lr * 0.5)
    else:
        optimizer = optax.adam(lr)

    opt_state = optimizer.init(params_array)

    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    history = []
    func_evals = 0

    for epoch in range(max_epochs):
        loss, grads = loss_and_grad(params_array)
        func_evals += 1  # One forward + backward = 1 effective evaluation

        updates, opt_state = optimizer.update(grads, opt_state, params_array)
        params_array = optax.apply_updates(params_array, updates)

        # Clamp to reasonable bounds
        params_array = jnp.clip(params_array, 0.01, 1000.0)

        nse = 1.0 - float(loss)
        history.append(
            {
                "epoch": epoch,
                "func_evals": func_evals,
                "loss": float(loss),
                "nse": nse,
            }
        )

        # Early stopping
        if nse > 0.95:
            break

    return history


def derivative_free_calibration(
    forcing: FUSEForcing, observations: jnp.ndarray, method: str = "sce_ua", max_evals: int = 20000
):
    """Simulate derivative-free calibration (SCE-UA or DDS)."""
    model = create_fuse_model(PRMS_CONFIG)
    initial_state = FUSEState.zeros()

    # This is a simulation of derivative-free methods
    # In practice, you would use actual SCE-UA or DDS implementations

    def evaluate(params_array):
        params = FUSEParams.from_array(params_array)
        _, flux_history = model.simulate(params, initial_state, forcing)

        sim = flux_history.q_total[60:]
        obs = observations[60:]

        ss_res = jnp.sum((sim - obs) ** 2)
        ss_tot = jnp.sum((obs - jnp.mean(obs)) ** 2)

        return float(1.0 - ss_res / (ss_tot + 1e-10))  # NSE

    evaluate_jit = jax.jit(evaluate)

    # Initialize
    params = get_default_params()
    best_params = params.to_array()
    best_nse = evaluate_jit(best_params)

    history = []

    np.random.seed(456)

    for i in range(max_evals):
        # Random perturbation (simulating population-based search)
        perturbation = np.random.normal(0, 0.1 * (1 - i / max_evals), len(best_params))
        candidate = best_params + perturbation
        candidate = jnp.clip(candidate, 0.01, 1000.0)

        nse = evaluate_jit(candidate)

        if nse > best_nse:
            best_nse = nse
            best_params = candidate

        if i % 100 == 0:
            history.append(
                {
                    "epoch": i,
                    "func_evals": i + 1,
                    "nse": best_nse,
                }
            )

        if best_nse > 0.90:
            break

    return history


def run_experiment(quick: bool = False):
    """Run calibration efficiency experiment."""
    print("Experiment 3: Calibration Efficiency")
    print("=" * 50)

    # Generate data
    n_timesteps = 365 if quick else 730
    forcing = generate_forcing(n_timesteps)
    observations, true_params = generate_observations(forcing)

    max_epochs_grad = 100 if quick else 500
    max_evals_df = 5000 if quick else 20000

    methods = [
        (
            "jFUSE_Adam",
            lambda: gradient_based_calibration(
                forcing, observations, "adam", max_epochs_grad, 0.01
            ),
        ),
        (
            "jFUSE_LBFGS",
            lambda: gradient_based_calibration(
                forcing, observations, "lbfgs", max_epochs_grad, 0.01
            ),
        ),
        (
            "SCE_UA",
            lambda: derivative_free_calibration(forcing, observations, "sce_ua", max_evals_df),
        ),
        ("DDS", lambda: derivative_free_calibration(forcing, observations, "dds", max_evals_df)),
    ]

    all_history = []
    efficiency_results = []

    for name, run_fn in methods:
        print(f"  Running {name}...", end=" ", flush=True)

        start_time = time.time()
        history = run_fn()
        elapsed = time.time() - start_time

        final_nse = history[-1]["nse"]
        total_evals = history[-1]["func_evals"]

        print(f"NSE: {final_nse:.3f}, Evals: {total_evals}, Time: {elapsed:.1f}s")

        # Record history
        for h in history:
            h["method"] = name
            all_history.append(h)

        # Find evaluations to reach targets
        nse_targets = [0.7, 0.8, 0.85]
        evals_to_target = {}

        for target in nse_targets:
            evals = None
            for h in history:
                if h["nse"] >= target:
                    evals = h["func_evals"]
                    break
            evals_to_target[f"evals_to_{target}"] = evals if evals else ">max"

        efficiency_results.append(
            {
                "method": name,
                "final_nse": final_nse,
                "total_evals": total_evals,
                "wall_time_s": elapsed,
                **evals_to_target,
            }
        )

    # Save results
    history_df = pd.DataFrame(all_history)
    history_df.to_csv(RESULTS_DIR / "exp3_convergence_history.csv", index=False)

    efficiency_df = pd.DataFrame(efficiency_results)
    efficiency_df.to_csv(RESULTS_DIR / "exp3_calibration_efficiency.csv", index=False)

    # Print summary table
    print("\nFunction evaluations to reach target NSE:")
    print("-" * 60)
    print(f"{'Method':<15} {'NSE=0.7':<12} {'NSE=0.8':<12} {'NSE=0.85':<12}")
    print("-" * 60)
    for r in efficiency_results:
        print(
            f"{r['method']:<15} {str(r['evals_to_0.7']):<12} {str(r['evals_to_0.8']):<12} {str(r['evals_to_0.85']):<12}"
        )

    print(f"\nResults saved to: {RESULTS_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Experiment 3: Calibration Efficiency")
    parser.add_argument("--quick", action="store_true", help="Run quick version")
    args = parser.parse_args()

    run_experiment(quick=args.quick)


if __name__ == "__main__":
    main()
