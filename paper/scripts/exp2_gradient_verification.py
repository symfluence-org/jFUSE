#!/usr/bin/env python3
"""
Experiment 2: Gradient Verification

Verify JAX automatic gradients against finite differences for all parameter types.

Outputs:
    - results/exp2_gradient_verification.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from jfuse.fuse.model import create_fuse_model
from jfuse.fuse.state import FUSEState, FUSEParams, FUSEForcing, get_default_params
from jfuse.fuse.config import PRMS_CONFIG

import jax
import jax.numpy as jnp

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def generate_forcing(n_timesteps: int = 365):
    """Generate forcing data."""
    np.random.seed(42)
    t = np.arange(n_timesteps)

    precip = 3.0 + 2.0 * np.sin(2 * np.pi * t / 365)
    precip = np.maximum(0, precip + np.random.exponential(2.0, n_timesteps))

    pet = 2.0 + 2.5 * np.sin(2 * np.pi * (t - 90) / 365)
    pet = np.maximum(0.5, pet)

    temp = 10.0 + 15.0 * np.sin(2 * np.pi * (t - 90) / 365)

    return FUSEForcing(
        precip=jnp.array(precip, dtype=jnp.float32),
        pet=jnp.array(pet, dtype=jnp.float32),
        temp=jnp.array(temp, dtype=jnp.float32),
    )


def generate_synthetic_observations(forcing: FUSEForcing, params: FUSEParams):
    """Generate synthetic observations by running model with true parameters."""
    model = create_fuse_model(PRMS_CONFIG)
    initial_state = FUSEState.zeros()

    _, flux_history = model.simulate(params, initial_state, forcing)

    # Add observation noise
    np.random.seed(123)
    noise = np.random.normal(0, 0.1, len(flux_history.q_total))
    obs = np.array(flux_history.q_total) + noise

    return jnp.array(np.maximum(0, obs), dtype=jnp.float32)


def create_loss_function(forcing: FUSEForcing, observations: jnp.ndarray, warmup: int = 30):
    """Create a loss function for parameter optimization."""
    model = create_fuse_model(PRMS_CONFIG)
    initial_state = FUSEState.zeros()

    def loss_fn(params: FUSEParams) -> float:
        _, flux_history = model.simulate(params, initial_state, forcing)
        simulated = flux_history.q_total[warmup:]
        observed = observations[warmup:]

        # NSE loss
        ss_res = jnp.sum((simulated - observed) ** 2)
        ss_tot = jnp.sum((observed - jnp.mean(observed)) ** 2)
        nse = 1.0 - ss_res / (ss_tot + 1e-10)

        return 1.0 - nse  # Minimize (1 - NSE)

    return loss_fn


def finite_difference_gradient(loss_fn, params: FUSEParams, param_name: str, eps: float = 1e-4):
    """Compute gradient using central finite differences."""
    params_array = params.to_array()
    param_names = [
        "S1_max",
        "S2_max",
        "ku",
        "ks",
        "ki",
        "kq",
        "alpha",
        "beta",
        "chi",
        "phi",
        "psi",
        "f_tens",
        "f_1",
        "f_2",
        "lambda_baseflow",
        "c_baseflow",
        "n_baseflow",
        "m_topmodel",
        "f_root",
        "r_exp",
        "melt_factor",
        "melt_temp",
        "rain_temp",
        "snow_temp",
        "melt_factor_amp",
        "gamma_shape",
        "gamma_scale",
        "rainfall_mult",
        "rainfall_add",
        "manning_n",
    ]

    try:
        idx = param_names.index(param_name)
    except ValueError:
        return np.nan

    # Forward difference
    params_plus = params_array.at[idx].set(params_array[idx] + eps)
    params_plus_obj = FUSEParams.from_array(params_plus)
    loss_plus = float(loss_fn(params_plus_obj))

    # Backward difference
    params_minus = params_array.at[idx].set(params_array[idx] - eps)
    params_minus_obj = FUSEParams.from_array(params_minus)
    loss_minus = float(loss_fn(params_minus_obj))

    # Central difference
    grad = (loss_plus - loss_minus) / (2 * eps)

    return grad


def jax_gradient(loss_fn, params: FUSEParams, param_name: str):
    """Compute gradient using JAX automatic differentiation."""
    param_names = [
        "S1_max",
        "S2_max",
        "ku",
        "ks",
        "ki",
        "kq",
        "alpha",
        "beta",
        "chi",
        "phi",
        "psi",
        "f_tens",
        "f_1",
        "f_2",
        "lambda_baseflow",
        "c_baseflow",
        "n_baseflow",
        "m_topmodel",
        "f_root",
        "r_exp",
        "melt_factor",
        "melt_temp",
        "rain_temp",
        "snow_temp",
        "melt_factor_amp",
        "gamma_shape",
        "gamma_scale",
        "rainfall_mult",
        "rainfall_add",
        "manning_n",
    ]

    try:
        idx = param_names.index(param_name)
    except ValueError:
        return np.nan

    def loss_from_array(params_array):
        params_obj = FUSEParams.from_array(params_array)
        return loss_fn(params_obj)

    grad_fn = jax.grad(loss_from_array)
    grads = grad_fn(params.to_array())

    return float(grads[idx])


def run_experiment(quick: bool = False):
    """Run gradient verification experiment."""
    print("Experiment 2: Gradient Verification")
    print("=" * 50)

    # Parameters to test
    params_to_test = [
        "S1_max",
        "S2_max",
        "ku",
        "ks",
        "ki",
        "alpha",
        "beta",
        "f_tens",
        "lambda_baseflow",
        "manning_n",
    ]

    if quick:
        params_to_test = params_to_test[:5]

    # Generate data
    n_timesteps = 100 if quick else 365
    forcing = generate_forcing(n_timesteps)
    params = get_default_params()
    observations = generate_synthetic_observations(forcing, params)

    # Create loss function
    loss_fn = create_loss_function(forcing, observations)
    loss_fn_jit = jax.jit(loss_fn)

    print(f"Testing {len(params_to_test)} parameters...")

    results = []

    for param_name in params_to_test:
        print(f"  {param_name}...", end=" ", flush=True)

        try:
            # Compute JAX gradient
            jax_grad = jax_gradient(loss_fn_jit, params, param_name)

            # Compute finite difference gradient
            fd_grad = finite_difference_gradient(loss_fn_jit, params, param_name)

            # Compute relative error
            if abs(fd_grad) > 1e-10:
                rel_error = abs(jax_grad - fd_grad) / abs(fd_grad) * 100
            else:
                rel_error = abs(jax_grad - fd_grad) * 100

            results.append(
                {
                    "parameter": param_name,
                    "jax_gradient": jax_grad,
                    "fd_gradient": fd_grad,
                    "abs_error": abs(jax_grad - fd_grad),
                    "rel_error_pct": rel_error,
                    "status": "PASS" if rel_error < 5.0 else "FAIL",
                }
            )

            print(f"JAX: {jax_grad:.6f}, FD: {fd_grad:.6f}, Error: {rel_error:.2f}%")

        except Exception as e:
            print(f"FAILED: {e}")
            results.append(
                {
                    "parameter": param_name,
                    "jax_gradient": np.nan,
                    "fd_gradient": np.nan,
                    "abs_error": np.nan,
                    "rel_error_pct": np.nan,
                    "status": "ERROR",
                }
            )

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "exp2_gradient_verification.csv", index=False)

    # Summary
    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_total = len(results)

    print("\nSummary:")
    print(f"  Parameters tested: {n_total}")
    print(f"  Passed (<5% error): {n_pass}/{n_total}")
    print(
        f"  Max relative error: {max(r['rel_error_pct'] for r in results if not np.isnan(r['rel_error_pct'])):.2f}%"
    )

    print(f"\nResults saved to: {RESULTS_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Experiment 2: Gradient Verification")
    parser.add_argument("--quick", action="store_true", help="Run quick version")
    args = parser.parse_args()

    run_experiment(quick=args.quick)


if __name__ == "__main__":
    main()
