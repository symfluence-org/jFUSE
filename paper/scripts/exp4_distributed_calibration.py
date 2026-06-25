#!/usr/bin/env python3
"""
Experiment 4: Distributed Calibration

Calibrate coupled FUSE+routing model with spatially-varying parameters
across multiple HRUs.

Outputs:
    - results/exp4_distributed_calibration.csv
    - results/exp4_parameter_sensitivity.csv
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


def generate_distributed_forcing(n_hrus: int, n_timesteps: int = 730):
    """Generate forcing for multiple HRUs."""
    np.random.seed(42)

    forcings = []

    for hru in range(n_hrus):
        t = np.arange(n_timesteps)

        # Add spatial variability
        precip_mult = 0.8 + 0.4 * np.random.random()
        pet_mult = 0.9 + 0.2 * np.random.random()
        temp_offset = -2 + 4 * np.random.random()

        precip = precip_mult * (3.0 + 2.0 * np.sin(2 * np.pi * t / 365))
        precip = np.maximum(0, precip + np.random.exponential(2.0, n_timesteps))

        pet = pet_mult * (2.0 + 2.5 * np.sin(2 * np.pi * (t - 90) / 365))
        pet = np.maximum(0.5, pet)

        temp = 10.0 + temp_offset + 15.0 * np.sin(2 * np.pi * (t - 90) / 365)

        forcings.append(
            FUSEForcing(
                precip=jnp.array(precip, dtype=jnp.float32),
                pet=jnp.array(pet, dtype=jnp.float32),
                temp=jnp.array(temp, dtype=jnp.float32),
            )
        )

    return forcings


def generate_hru_areas(n_hrus: int):
    """Generate HRU areas (km^2)."""
    np.random.seed(789)
    areas = 50 + 100 * np.random.random(n_hrus)
    return areas


def generate_network_topology(n_hrus: int):
    """Generate simple river network topology."""
    # Simple linear network: HRU i drains to reach i, reach i drains to reach i+1
    downstream = list(range(1, n_hrus)) + [-1]  # -1 = outlet
    return downstream


def runoff_to_discharge(runoff_mm: jnp.ndarray, area_km2: float) -> jnp.ndarray:
    """Convert runoff (mm/day) to discharge (m^3/s)."""
    # Q (m^3/s) = R (mm/day) * A (km^2) / 86400 / 1000 * 1e6
    return runoff_mm * area_km2 * 1000.0 / 86400.0


def muskingum_cunge_route(
    inflows: jnp.ndarray, downstream: list, manning_n: jnp.ndarray, dt: float = 86400.0
):
    """Simple Muskingum-Cunge routing through network."""
    n_reaches = len(downstream)
    n_timesteps = inflows.shape[1]

    outflows = jnp.zeros_like(inflows)

    # Route in topological order (assuming linear network)
    for t in range(1, n_timesteps):
        for i in range(n_reaches):
            # Lateral inflow from HRU
            lateral = inflows[i, t]

            # Upstream inflow (if any)
            upstream_inflow = 0.0
            for j in range(n_reaches):
                if downstream[j] == i:
                    upstream_inflow += outflows[j, t - 1]

            # Simple routing with Manning's n effect
            K = 0.5 * manning_n[i]  # Storage constant proportional to n
            X = 0.2  # Weighting factor

            C1 = (dt - 2 * K * X) / (2 * K * (1 - X) + dt)
            C2 = (dt + 2 * K * X) / (2 * K * (1 - X) + dt)
            C3 = (2 * K * (1 - X) - dt) / (2 * K * (1 - X) + dt)

            total_inflow = lateral + upstream_inflow

            outflows = outflows.at[i, t].set(
                C1 * total_inflow + C2 * inflows[i, t - 1] + C3 * outflows[i, t - 1]
            )

    return outflows


def run_distributed_simulation(
    forcings: list, params_list: list, areas: np.ndarray, downstream: list, manning_n: jnp.ndarray
):
    """Run distributed FUSE + routing simulation."""
    model = create_fuse_model(PRMS_CONFIG)
    n_hrus = len(forcings)

    # Run FUSE for each HRU
    runoffs = []
    for i in range(n_hrus):
        initial_state = FUSEState.zeros()
        _, flux_history = model.simulate(params_list[i], initial_state, forcings[i])
        runoffs.append(flux_history.q_total)

    runoffs = jnp.stack(runoffs)  # (n_hrus, n_timesteps)

    # Convert to discharge
    discharges = jnp.zeros_like(runoffs)
    for i in range(n_hrus):
        discharges = discharges.at[i].set(runoff_to_discharge(runoffs[i], areas[i]))

    # Route through network
    routed = muskingum_cunge_route(discharges, downstream, manning_n)

    # Outlet discharge (last reach)
    outlet_idx = downstream.index(-1)
    outlet_discharge = routed[outlet_idx]

    return outlet_discharge, runoffs


def generate_observations(forcings: list, areas: np.ndarray, downstream: list):
    """Generate synthetic observations."""
    n_hrus = len(forcings)

    # True parameters with spatial variation
    np.random.seed(321)
    true_params = []
    for i in range(n_hrus):
        base = get_default_params()
        # Add spatial variation to key parameters
        true_params.append(
            FUSEParams(
                S1_max=base.S1_max * (0.8 + 0.4 * np.random.random()),
                S2_max=base.S2_max * (0.8 + 0.4 * np.random.random()),
                ku=base.ku * (0.9 + 0.2 * np.random.random()),
                ks=base.ks * (0.8 + 0.4 * np.random.random()),
                ki=base.ki,
                kq=base.kq,
                alpha=base.alpha,
                beta=base.beta,
                chi=base.chi,
                phi=base.phi,
                psi=base.psi,
                f_tens=base.f_tens,
                f_1=base.f_1,
                f_2=base.f_2,
                lambda_baseflow=base.lambda_baseflow,
                c_baseflow=base.c_baseflow,
                n_baseflow=base.n_baseflow,
                m_topmodel=base.m_topmodel,
                f_root=base.f_root,
                r_exp=base.r_exp,
                melt_factor=base.melt_factor,
                melt_temp=base.melt_temp,
                rain_temp=base.rain_temp,
                snow_temp=base.snow_temp,
                melt_factor_amp=base.melt_factor_amp,
                gamma_shape=base.gamma_shape,
                gamma_scale=base.gamma_scale,
                rainfall_mult=base.rainfall_mult,
                rainfall_add=base.rainfall_add,
                manning_n=base.manning_n,
            )
        )

    true_manning = jnp.array([0.03 + 0.02 * np.random.random() for _ in range(n_hrus)])

    outlet_q, _ = run_distributed_simulation(forcings, true_params, areas, downstream, true_manning)

    # Add noise
    np.random.seed(654)
    noise = np.random.normal(0, 0.05 * np.std(outlet_q), len(outlet_q))
    obs = np.array(outlet_q) + noise

    return jnp.array(np.maximum(0, obs), dtype=jnp.float32), true_params, true_manning


def calibrate_distributed(
    forcings: list,
    observations: jnp.ndarray,
    areas: np.ndarray,
    downstream: list,
    spatial_params: list,
    max_epochs: int = 200,
    lr: float = 0.01,
):
    """Calibrate distributed model."""
    n_hrus = len(forcings)

    # Initialize parameters
    base_params = get_default_params()
    base_array = base_params.to_array()

    # Determine which parameters are spatially varying
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

    spatial_indices = [param_names.index(p) for p in spatial_params if p in param_names]
    global_indices = [i for i in range(len(param_names)) if i not in spatial_indices]

    # Parameter arrays
    # Global params: shared across HRUs
    n_global = len(global_indices)
    n_spatial = len(spatial_indices)

    global_params = base_array[jnp.array(global_indices)]
    spatial_params_arr = jnp.tile(base_array[jnp.array(spatial_indices)], (n_hrus, 1))
    manning_params = jnp.ones(n_hrus) * 0.035

    # Combine into single array for optimization
    all_params = jnp.concatenate([global_params, spatial_params_arr.flatten(), manning_params])

    n_params = len(all_params)

    def loss_fn(all_params):
        # Unpack parameters
        global_p = all_params[:n_global]
        spatial_p = all_params[n_global : n_global + n_spatial * n_hrus].reshape(n_hrus, n_spatial)
        manning = all_params[-n_hrus:]

        # Reconstruct parameter objects
        params_list = []
        for i in range(n_hrus):
            full_array = jnp.zeros(len(param_names))
            full_array = full_array.at[jnp.array(global_indices)].set(global_p)
            full_array = full_array.at[jnp.array(spatial_indices)].set(spatial_p[i])
            params_list.append(FUSEParams.from_array(full_array))

        # Run simulation
        outlet_q, _ = run_distributed_simulation(forcings, params_list, areas, downstream, manning)

        # NSE loss
        warmup = 60
        sim = outlet_q[warmup:]
        obs = observations[warmup:]

        ss_res = jnp.sum((sim - obs) ** 2)
        ss_tot = jnp.sum((obs - jnp.mean(obs)) ** 2)

        return ss_res / (ss_tot + 1e-10)

    # Optimization
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(all_params)

    loss_and_grad = jax.value_and_grad(loss_fn)

    history = []

    for epoch in range(max_epochs):
        loss, grads = loss_and_grad(all_params)
        updates, opt_state = optimizer.update(grads, opt_state, all_params)
        all_params = optax.apply_updates(all_params, updates)

        # Clamp
        all_params = jnp.clip(all_params, 0.001, 1000.0)

        nse = 1.0 - float(loss)
        history.append(
            {
                "epoch": epoch,
                "nse": nse,
                "n_params": n_params,
            }
        )

        if epoch % 50 == 0:
            print(f"    Epoch {epoch}: NSE = {nse:.4f}")

        if nse > 0.95:
            break

    return history, n_params


def run_experiment(quick: bool = False):
    """Run distributed calibration experiment."""
    print("Experiment 4: Distributed Calibration")
    print("=" * 50)

    n_hrus = 10 if quick else 29
    n_timesteps = 365 if quick else 730
    max_epochs = 100 if quick else 300

    print(f"  HRUs: {n_hrus}")
    print(f"  Timesteps: {n_timesteps}")

    # Generate data
    print("  Generating forcing data...")
    forcings = generate_distributed_forcing(n_hrus, n_timesteps)
    areas = generate_hru_areas(n_hrus)
    downstream = generate_network_topology(n_hrus)

    print("  Generating observations...")
    observations, true_params, true_manning = generate_observations(forcings, areas, downstream)

    # Test different parameter configurations
    configurations = [
        ("Global", []),  # All parameters global
        ("Spatial_key", ["S1_max", "S2_max", "ks"]),  # Key params spatial
        ("Spatial_full", ["S1_max", "S2_max", "ku", "ks", "ki", "alpha"]),
        ("Spatial_all", ["S1_max", "S2_max", "ku", "ks", "ki", "kq", "alpha", "beta", "f_tens"]),
    ]

    if quick:
        configurations = configurations[:2]

    results = []
    all_history = []

    for config_name, spatial_params in configurations:
        print(f"\n  Configuration: {config_name}")

        start_time = time.time()
        history, n_params = calibrate_distributed(
            forcings, observations, areas, downstream, spatial_params, max_epochs
        )
        elapsed = time.time() - start_time

        final_nse = history[-1]["nse"]

        results.append(
            {
                "configuration": config_name,
                "n_spatial_params": len(spatial_params),
                "n_total_params": n_params,
                "final_nse": final_nse,
                "epochs": len(history),
                "time_s": elapsed,
            }
        )

        for h in history:
            h["configuration"] = config_name
            all_history.append(h)

        print(f"    Final NSE: {final_nse:.4f}, Params: {n_params}, Time: {elapsed:.1f}s")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / "exp4_distributed_calibration.csv", index=False)

    history_df = pd.DataFrame(all_history)
    history_df.to_csv(RESULTS_DIR / "exp4_convergence_history.csv", index=False)

    # Summary
    print("\nSummary:")
    print("-" * 70)
    print(f"{'Config':<20} {'Params':<10} {'NSE':<10} {'Time (s)':<10}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['configuration']:<20} {r['n_total_params']:<10} {r['final_nse']:<10.3f} {r['time_s']:<10.1f}"
        )

    print(f"\nResults saved to: {RESULTS_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Experiment 4: Distributed Calibration")
    parser.add_argument("--quick", action="store_true", help="Run quick version")
    args = parser.parse_args()

    run_experiment(quick=args.quick)


if __name__ == "__main__":
    main()
