"""
jFUSE Command Line Interface

Run jFUSE simulations using FUSE-compatible file manager configuration.

Usage:
    jfuse run <filemanager> <basin_id> [--mode=sim|calib] [--verbose]
    jfuse run <filemanager> <basin_id> --mode=calib --method=gradient
    jfuse info <filemanager>
    jfuse structures
"""

# Enable 64-bit precision for JAX - MUST be done before any JAX imports
import jax

jax.config.update("jax_enable_x64", True)

import argparse
import sys
import time
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from jfuse.io.filemanager import FileManagerConfig, ForcingInfo


def load_observations_csv(
    filepath: str,
    times: np.ndarray,
    basin_area_m2: float,
    datetime_col: str = "datetime",
    discharge_col: str = "discharge_cms",
    datetime_format: str = None,
) -> jnp.ndarray:
    """Load observations from CSV file and align with forcing timestamps.

    Args:
        filepath: Path to CSV file with datetime and discharge columns
        times: Array of forcing timestamps (daily)
        basin_area_m2: Total basin area in m² (for unit conversion)
        datetime_col: Column name for datetime
        discharge_col: Column name for discharge (in m³/s)
        datetime_format: Datetime format string (auto-detected if None)

    Returns:
        Observations array aligned with forcing times [n_timesteps] in mm/day
    """
    import pandas as pd

    df = pd.read_csv(filepath)

    # Auto-detect datetime format
    if datetime_format is None:
        # Try common formats
        sample = df[datetime_col].iloc[0]
        formats_to_try = [
            "%d/%m/%Y %H:%M",  # 14/11/1999 10:00
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d %H:%M",
            "%m/%d/%Y %H:%M",
            "%d-%m-%Y %H:%M",
        ]
        for fmt in formats_to_try:
            try:
                pd.to_datetime(sample, format=fmt)
                datetime_format = fmt
                break
            except (ValueError, TypeError):
                continue

    if datetime_format:
        df[datetime_col] = pd.to_datetime(df[datetime_col], format=datetime_format)
    else:
        df[datetime_col] = pd.to_datetime(df[datetime_col])

    df = df.set_index(datetime_col)

    # Resample to daily mean if hourly/sub-daily
    df_daily = df.resample("1D").mean()

    # Convert m³/s to mm/day: Q_mm = Q_m3s * 86400 / area_m2 * 1000
    # = Q_m3s * 86400000 / area_m2
    df_daily["q_mm_day"] = df_daily[discharge_col] * 86400.0 * 1000.0 / basin_area_m2

    # Align with forcing times
    forcing_times = pd.DatetimeIndex(times)
    obs_aligned = np.full(len(forcing_times), np.nan)

    for i, t in enumerate(forcing_times):
        t_date = t.normalize()  # Get date only
        if t_date in df_daily.index:
            obs_aligned[i] = df_daily.loc[t_date, "q_mm_day"]

    n_valid = np.sum(~np.isnan(obs_aligned))
    print(f"  Loaded {n_valid}/{len(obs_aligned)} observations from CSV")
    print(f"  Obs range: {np.nanmin(obs_aligned):.4f} - {np.nanmax(obs_aligned):.4f} mm/day")
    print(f"  Obs mean: {np.nanmean(obs_aligned):.4f} mm/day")

    return jnp.array(obs_aligned)


def plot_hydrograph(
    times: np.ndarray,
    simulated: np.ndarray,
    observed: Optional[np.ndarray] = None,
    output_path: str = None,
    title: str = "Hydrograph",
    warmup_days: int = 0,
    metrics: Optional[Dict[str, float]] = None,
) -> None:
    """Plot simulated vs observed hydrograph.

    Args:
        times: Array of timestamps
        simulated: Simulated discharge [n_timesteps] in mm/day
        observed: Observed discharge [n_timesteps] in mm/day (optional)
        output_path: Path to save plot (if None, displays interactively)
        title: Plot title
        warmup_days: Number of warmup days to mark
        metrics: Dictionary of performance metrics to display
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("  WARNING: matplotlib not installed, skipping plot")
        print("           Install with: pip install matplotlib")
        return

    fig, ax = plt.subplots(figsize=(12, 5))

    # Convert times to matplotlib format
    times_plot = np.array(times, dtype="datetime64[D]").astype("datetime64[us]").astype(datetime)

    # Plot simulated
    ax.plot(times_plot, simulated, "b-", label="Simulated", linewidth=0.8, alpha=0.8)

    # Plot observed if available
    if observed is not None:
        ax.plot(times_plot, observed, "k-", label="Observed", linewidth=0.8, alpha=0.8)

    # Mark warmup period
    if warmup_days > 0 and warmup_days < len(times):
        ax.axvline(
            times_plot[warmup_days],
            color="r",
            linestyle="--",
            alpha=0.5,
            label=f"Warmup ({warmup_days} days)",
        )

    # Add metrics text box
    if metrics:
        metrics_text = "\n".join([f"{k.upper()}: {v:.3f}" for k, v in metrics.items()])
        ax.text(
            0.02,
            0.98,
            metrics_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Discharge (mm/day)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved to: {output_path}")
    else:
        plt.show()

    plt.close()


def load_forcing_netcdf(
    filepath: str,
    forcing_info: "ForcingInfo",
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> Tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    np.ndarray,
    Optional[jnp.ndarray],
    Dict[str, Optional[str]],
]:
    """Load forcing data from NetCDF file.

    Args:
        filepath: Path to forcing NetCDF file
        forcing_info: ForcingInfo with variable names
        start_date: Start date for subsetting
        end_date: End date for subsetting

    Returns:
        Tuple of (precip, pet, temp, times, obs) arrays
    """
    import xarray as xr

    ds = xr.open_dataset(filepath)

    # Common variable name aliases
    PRECIP_ALIASES = [
        "pr",
        "pptrate",
        "precip",
        "precipitation",
        "prcp",
        "ppt",
        "rain",
        "rainfall",
        "P",
    ]
    PET_ALIASES = ["pet", "evspsblpot", "potevap", "pevap", "pe", "ET0", "PET"]
    TEMP_ALIASES = ["temp", "airtemp", "tas", "tair", "t2m", "temperature", "T", "tmean"]
    OBS_ALIASES = ["q_obs", "qobs", "streamflow", "discharge", "runoff", "Q", "q"]
    TIME_ALIASES = ["time", "Time", "TIME", "t"]

    def find_var(ds, preferred, aliases):
        """Find variable in dataset, trying preferred name first, then aliases."""
        if preferred in ds:
            return preferred
        for alias in aliases:
            if alias in ds:
                return alias
        return None

    # Find time coordinate
    time_var = find_var(ds, forcing_info.time_var, TIME_ALIASES)
    if time_var is None:
        # Try to find any dimension that looks like time
        for dim in ds.dims:
            if "time" in dim.lower():
                time_var = dim
                break
        if time_var is None:
            time_var = list(ds.dims.keys())[0]  # Fall back to first dimension

    times = ds[time_var].values

    # Convert to datetime if needed
    if hasattr(times[0], "astype"):
        times = times.astype("datetime64[D]")

    # Subset by date if requested
    if start_date is not None or end_date is not None:
        time_mask = np.ones(len(times), dtype=bool)
        if start_date:
            start_np = np.datetime64(start_date.strftime("%Y-%m-%d"))
            time_mask &= times >= start_np
        if end_date:
            end_np = np.datetime64(end_date.strftime("%Y-%m-%d"))
            time_mask &= times <= end_np

        times = times[time_mask]
        ds = ds.isel({time_var: time_mask})

    # Find forcing variables with auto-detection
    precip_var = find_var(ds, forcing_info.precip_var, PRECIP_ALIASES)
    pet_var = find_var(ds, forcing_info.pet_var, PET_ALIASES)
    temp_var = find_var(ds, forcing_info.temp_var, TEMP_ALIASES)
    obs_var = find_var(ds, forcing_info.obs_var if forcing_info.obs_var else "q_obs", OBS_ALIASES)

    if precip_var is None:
        raise KeyError(f"Could not find precipitation variable. Available: {list(ds.data_vars)}")
    if pet_var is None:
        raise KeyError(f"Could not find PET variable. Available: {list(ds.data_vars)}")
    if temp_var is None:
        raise KeyError(f"Could not find temperature variable. Available: {list(ds.data_vars)}")

    # Extract forcing variables
    precip = ds[precip_var].values
    pet = ds[pet_var].values
    temp = ds[temp_var].values

    # Smart unit conversion based on attributes
    def get_unit(var_name):
        if var_name in ds:
            return ds[var_name].attrs.get("units", "").lower()
        return ""

    # Precipitation conversion
    precip_units = get_unit(precip_var)
    if "kg" in precip_units and "s" in precip_units:  # kg/m2/s
        precip = precip * 86400.0
    elif "mm" in precip_units and "s" in precip_units:  # mm/s
        precip = precip * 86400.0
    # else assume already in mm/day

    # PET conversion
    pet_units = get_unit(pet_var)
    if "kg" in pet_units and "s" in pet_units:  # kg/m2/s
        pet = pet * 86400.0
    elif "mm" in pet_units and "s" in pet_units:  # mm/s
        pet = pet * 86400.0
    # else assume already in mm/day

    # Temperature conversion (K to C if needed)
    temp_units = get_unit(temp_var)
    if "k" in temp_units.lower() and "c" not in temp_units.lower():
        temp = temp - 273.15
    elif np.nanmean(temp) > 100:  # Likely Kelvin based on value
        temp = temp - 273.15
    # else assume already in Celsius

    # Handle dimensions - ensure shape is (time,) or (time, hru)
    # NetCDF files might have dimensions in different orders
    if precip.ndim == 1:
        # Already 1D - lumped case
        pass
    elif precip.ndim == 2:
        # Check if time is first dimension (should be longer)
        if precip.shape[0] < precip.shape[1]:
            # Likely (hru, time) - transpose to (time, hru)
            precip = precip.T
            pet = pet.T
            temp = temp.T

        # Squeeze if only 1 HRU
        if precip.shape[-1] == 1:
            precip = precip.squeeze(-1)
            pet = pet.squeeze(-1)
            temp = temp.squeeze(-1)
    elif precip.ndim == 3:
        # Could be (time, lat, lon) or similar - flatten spatial dims
        orig_shape = precip.shape
        precip = precip.reshape(orig_shape[0], -1)
        pet = pet.reshape(orig_shape[0], -1)
        temp = temp.reshape(orig_shape[0], -1)

    # Get observations if available
    # Observations should always be 1D (outlet streamflow only)
    obs = None
    if obs_var is not None and obs_var in ds:
        obs = ds[obs_var].values
        # Ensure 1D - observations are outlet only
        if obs.ndim > 1:
            if obs.shape[-1] == 1 or obs.shape[0] == 1:
                obs = obs.squeeze()
            else:
                # Multiple spatial points - take first column (assume outlet)
                # or mean if that's more appropriate
                obs = obs[:, 0] if obs.shape[0] > obs.shape[1] else obs[0, :]
        obs = jnp.array(obs)

    ds.close()

    # Store found variable names for verbose output
    found_vars = {
        "precip": precip_var,
        "pet": pet_var,
        "temp": temp_var,
        "obs": obs_var if obs is not None else None,
        "time": time_var,
    }

    return (
        jnp.array(precip),
        jnp.array(pet),
        jnp.array(temp),
        times,
        obs,
        found_vars,
    )


def run_simulation(
    fm_config: "FileManagerConfig",
    basin_id: str,
    verbose: bool = True,
    network_path: Optional[str] = None,
    obs_agg: str = "first",
    obs_file: Optional[str] = None,
    plot: bool = False,
) -> Dict[str, Any]:
    """Run a jFUSE simulation.

    Args:
        fm_config: File manager configuration
        basin_id: Basin identifier
        verbose: Print progress messages
        network_path: Path to network topology file (overrides default naming)
        obs_agg: How to aggregate 2D observations ('first', 'last', 'mean', 'sum')
        obs_file: Path to CSV file with outlet observations (overrides NetCDF obs)
        plot: Generate hydrograph plot

    Returns:
        Dictionary with simulation results
    """
    from jfuse import FUSEModel
    from jfuse.fuse import FUSEConfig
    from jfuse.fuse.config import RoutingType
    from jfuse.io.filemanager import ForcingInfo, parse_forcing_info

    if verbose:
        print(f"\n{'='*60}")
        print("jFUSE Simulation")
        print(f"{'='*60}")
        print(f"Basin: {basin_id}")
        print(f"Model ID: {fm_config.model_id}")

    # Load model configuration from decisions file
    if verbose:
        print(f"\nLoading decisions from: {fm_config.decisions_path}")

    config = FUSEConfig.from_file(str(fm_config.decisions_path))

    if verbose:
        print(config.describe())

    # Load forcing info
    forcing_info = ForcingInfo()
    if fm_config.forcing_info_path.exists():
        forcing_info = parse_forcing_info(str(fm_config.forcing_info_path))

    # Load forcing data
    forcing_file = fm_config.forcing_file(basin_id)
    if verbose:
        print(f"\nLoading forcing from: {forcing_file}")

    precip, pet, temp, times, obs, found_vars = load_forcing_netcdf(
        str(forcing_file),
        forcing_info,
        fm_config.date_start_sim,
        fm_config.date_end_sim,
    )

    # Detect number of HRUs and spatial mode
    if precip.ndim == 1:
        n_hrus = 1
        is_distributed = False
    else:
        n_hrus = precip.shape[1]
        is_distributed = n_hrus > 1  # Only distributed if MORE than 1 HRU

        # If only 1 HRU but 2D array, squeeze to 1D (lumped)
        if n_hrus == 1:
            precip = precip.squeeze(-1)
            pet = pet.squeeze(-1)
            temp = temp.squeeze(-1)

    # Check if CHANNEL routing should be enabled
    # Channel routing requires: distributed mode + network file exists
    # Q_TDH (hillslope routing) is handled internally by FUSE
    use_channel_routing = False
    if network_path:
        network_file = Path(network_path)
    else:
        network_file = fm_config.network_file(basin_id)

    if is_distributed:
        network_exists = network_file.exists()

        if network_exists:
            use_channel_routing = True
            if verbose:
                print(f"\nNetwork file found: {network_file}")
        else:
            if verbose:
                print("\n  No network file found - using area-weighted HRU aggregation")

    # Create model based on spatial mode
    if use_channel_routing:
        # Use CoupledModel for FUSE + Muskingum-Cunge channel routing
        from jfuse.coupled import CoupledModel
        from jfuse.io import load_network

        if verbose:
            print(f"Loading network topology from: {network_file}")

        try:
            network, hru_areas = load_network(str(network_file))
        except KeyError as e:
            print(f"  ERROR: Could not load network file: {e}")
            print("  Falling back to area-weighted HRU aggregation")
            use_channel_routing = False
        except Exception as e:
            print(f"  ERROR: Failed to load network: {e}")
            print("  Falling back to area-weighted HRU aggregation")
            use_channel_routing = False

    if use_channel_routing:
        # Check HRU count matches
        if len(network.reaches) != n_hrus:
            if verbose:
                print(
                    f"  WARNING: Network has {len(network.reaches)} reaches but forcing has {n_hrus} HRUs"
                )
                print(f"  Using min({len(network.reaches)}, {n_hrus}) for simulation")

        # Debug: Check outlet detection
        network_arrays = network.to_arrays()
        n_outlets = int(jnp.sum(network_arrays.is_outlet))
        outlet_indices = jnp.where(network_arrays.is_outlet)[0]
        if verbose:
            print(f"  Outlets detected: {n_outlets} at indices {list(outlet_indices)}")
            if n_outlets == 0:
                print("  WARNING: No outlets found in network! Check downstream_id values.")
                # Show some downstream IDs for debugging
                print(f"  DEBUG: First 5 downstream_idx: {list(network_arrays.downstream_idx[:5])}")

        model = CoupledModel(
            fuse_config=config,
            network=network_arrays,
            hru_areas=hru_areas,
            n_hrus=n_hrus,
        )

        # Final sanity check on outlet detection
        if n_outlets == 0:
            print("  CRITICAL: No outlets in network! Routing will not work.")
            print("  Check your network file's downstream_id/downSegId values.")
            print(
                "  Outlets are identified when downstream_id = 0, -1, or points to non-existent reach."
            )

        params = model.default_params()
        initial_state = None  # CoupledModel will create default

        if verbose:
            print(f"  Reaches: {len(network.reaches)}")
            print("  Channel routing: Muskingum-Cunge")
    else:
        # Use FUSEModel only (no channel routing)
        model = FUSEModel(config, n_hrus=n_hrus)

        # Initialize params and state with correct shape
        from jfuse.fuse.state import State, Parameters

        params = Parameters.default(n_hrus=n_hrus)
        initial_state = State.default(n_hrus=n_hrus)

    forcing = (precip, pet, temp)

    # Load observations from CSV if provided AND we have a network for unit conversion
    # For lumped mode, always use NetCDF observations (already in mm/day)
    if obs_file and use_channel_routing:
        if verbose:
            print(f"\nLoading observations from CSV: {obs_file}")

        # Get basin area for unit conversion (m³/s -> mm/day)
        basin_area = float(jnp.sum(hru_areas))

        obs = load_observations_csv(
            obs_file,
            times,
            basin_area,
        )
    elif obs_file and not use_channel_routing:
        if verbose:
            print("\n  NOTE: --obs-file ignored for lumped mode (no network for unit conversion)")
            print("        Using observations from NetCDF file (already in mm/day)")

    # Determine hillslope routing type from config
    hillslope_routing = "gamma" if config.routing == RoutingType.GAMMA else "none"

    if verbose:
        print(
            f"  Variables found: pr={found_vars['precip']}, pet={found_vars['pet']}, temp={found_vars['temp']}, obs={found_vars['obs']}"
        )
        print(f"  Period: {times[0]} to {times[-1]}")
        print(f"  Timesteps: {len(times)}")
        if is_distributed:
            print(f"  HRUs: {n_hrus} (distributed mode)")
            print(f"  Hillslope routing (Q_TDH): {hillslope_routing}")
            if use_channel_routing:
                print("  Channel routing: Muskingum-Cunge")
            else:
                print("  Channel routing: none (area-weighted aggregation)")
        else:
            print("  Mode: lumped")
            print(f"  Hillslope routing (Q_TDH): {hillslope_routing}")
        print(f"  Mean precip: {float(jnp.nanmean(precip)):.2f} mm/day")
        print(f"  Mean PET: {float(jnp.nanmean(pet)):.2f} mm/day")
        print(f"  Mean temp: {float(jnp.nanmean(temp)):.1f} °C")
        if obs is not None:
            obs_mean = float(jnp.nanmean(obs[~jnp.isnan(obs)]))
            print(f"  Mean observed Q: {obs_mean:.2f} mm/day")
            if obs.ndim > 1:
                # Check if observations are tiled (same value across all HRUs)
                obs_col0 = obs[:, 0]
                obs_col_last = obs[:, -1]
                is_tiled = jnp.allclose(obs_col0, obs_col_last, equal_nan=True)

                # Show aggregation info
                print(f"  2D obs shape: {obs.shape}, using --obs-agg={obs_agg}")
                if obs_agg == "first":
                    agg_val = float(jnp.nanmean(obs_col0))
                elif obs_agg == "last":
                    agg_val = float(jnp.nanmean(obs[:, -1]))
                elif obs_agg == "mean":
                    agg_val = float(jnp.nanmean(jnp.mean(obs, axis=1)))
                elif obs_agg == "sum":
                    agg_val = float(jnp.nanmean(jnp.sum(obs, axis=1)))
                else:
                    agg_val = obs_mean
                print(f"  Aggregated obs mean: {agg_val:.4f} mm/day")

                if is_tiled:
                    print("  INFO: Observations appear tiled (same across HRUs)")
                    sum_val = float(jnp.nanmean(obs_col0 * obs.shape[1]))
                    print(
                        f"        Try --obs-agg=sum if obs was divided by n_hrus ({sum_val:.4f} mm/day)"
                    )

    # Run simulation
    if verbose:
        print("\nRunning simulation...")

    t0 = time.time()

    if use_channel_routing:
        # CoupledModel returns (outlet_Q, runoff) - NOT (runoff, Q_outlet)!
        Q_outlet, runoff_hru = model.simulate(forcing, params, initial_state)
        # For metrics, use aggregated runoff in mm/day (same units as obs)
        runoff = runoff_hru
        final_state = None

        # Debug: Check routing output
        if verbose:
            from jfuse.coupled import runoff_to_inflow

            network_arrays = model.network
            print(f"  DEBUG: is_outlet sum = {int(jnp.sum(network_arrays.is_outlet))}")
            print(f"  DEBUG: outlet indices = {list(jnp.where(network_arrays.is_outlet)[0])}")
            print(
                f"  DEBUG: hru_areas range = {float(model.hru_areas.min()):.2e} - {float(model.hru_areas.max()):.2e} m²"
            )

            # Check lateral inflow
            lateral_inflow = runoff_to_inflow(runoff_hru, model.hru_areas, 86400.0)
            print(
                f"  DEBUG: lateral_inflow range = {float(lateral_inflow.min()):.4f} - {float(lateral_inflow.max()):.4f} m³/s"
            )
            print(
                f"  DEBUG: Q_outlet range = {float(Q_outlet.min()):.4f} - {float(Q_outlet.max()):.4f} m³/s"
            )
    else:
        runoff, final_state = model.simulate(forcing, params, initial_state)

    elapsed = time.time() - t0

    if verbose:
        print(f"  Completed in {elapsed:.2f}s")
        print(f"  Mean runoff: {float(jnp.nanmean(runoff)):.2f} mm/day")
        if runoff.ndim > 1:
            print(f"  Runoff shape: {runoff.shape} (time, hru)")
        if use_channel_routing:
            print(
                f"  Outlet Q range: {float(Q_outlet.min()):.2f} - {float(Q_outlet.max()):.2f} m³/s"
            )

    # Compute metrics if observations available
    metrics = {}
    if obs is not None:
        # Determine warmup period
        if fm_config.date_start_eval and fm_config.date_start_sim:
            warmup_days = (fm_config.date_start_eval - fm_config.date_start_sim).days
        else:
            warmup_days = 365

        # For distributed case, aggregate runoff across HRUs
        if runoff.ndim > 1:
            runoff_agg = jnp.mean(runoff, axis=1)  # Mean across HRUs
        else:
            runoff_agg = runoff

        # Ensure observations are 1D (outlet only)
        obs_1d = obs
        if obs.ndim > 1:
            if obs.shape[1] == 1:
                obs_1d = obs.squeeze()
            else:
                # Apply aggregation method
                if obs_agg == "first":
                    obs_1d = obs[:, 0]
                elif obs_agg == "last":
                    obs_1d = obs[:, -1]
                elif obs_agg == "mean":
                    obs_1d = jnp.mean(obs, axis=1)
                elif obs_agg == "sum":
                    obs_1d = jnp.sum(obs, axis=1)
                else:
                    obs_1d = obs[:, 0]  # Default to first

        # Compute metrics after warmup
        sim_eval = runoff_agg[warmup_days:]
        obs_eval = obs_1d[warmup_days:]

        # Ensure both are 1D
        sim_eval = jnp.atleast_1d(sim_eval).flatten()
        obs_eval = jnp.atleast_1d(obs_eval).flatten()

        # Remove NaNs
        valid = ~(jnp.isnan(sim_eval) | jnp.isnan(obs_eval))
        sim_valid = sim_eval[valid]
        obs_valid = obs_eval[valid]

        if len(sim_valid) > 0:
            # NSE (undefined when observations are constant -> ss_tot == 0)
            ss_res = jnp.sum((obs_valid - sim_valid) ** 2)
            ss_tot = jnp.sum((obs_valid - jnp.mean(obs_valid)) ** 2)
            nse = jnp.where(ss_tot > 0, 1 - ss_res / ss_tot, jnp.nan)

            # KGE (alpha/beta undefined when obs std/mean are zero)
            std_obs = jnp.std(obs_valid)
            mean_obs = jnp.mean(obs_valid)
            r = jnp.corrcoef(sim_valid, obs_valid)[0, 1]
            alpha = jnp.where(std_obs > 0, jnp.std(sim_valid) / std_obs, jnp.nan)
            beta = jnp.where(mean_obs != 0, jnp.mean(sim_valid) / mean_obs, jnp.nan)
            kge = 1 - jnp.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

            metrics["nse"] = float(nse)
            metrics["kge"] = float(kge)
            metrics["r"] = float(r)
            metrics["alpha"] = float(alpha)
            metrics["beta"] = float(beta)

            if verbose:
                print(f"\nPerformance (after {warmup_days} day warmup):")
                print(f"  NSE: {metrics['nse']:.4f}")
                print(f"  KGE: {metrics['kge']:.4f}")
                print(f"  r: {metrics['r']:.4f}")

    # Save output
    fm_config.output_path.mkdir(parents=True, exist_ok=True)
    output_file = fm_config.output_file("_output.nc")

    if verbose:
        print(f"\nSaving output to: {output_file}")

    save_output_netcdf(
        str(output_file),
        times,
        runoff,
        obs,
        final_state,
        metrics,
        fm_config,
        config,
    )

    # Generate plot if requested
    if plot:
        if verbose:
            print("\nGenerating hydrograph plot...")

        # Get 1D arrays for plotting
        if runoff.ndim > 1:
            runoff_plot = np.array(jnp.mean(runoff, axis=1))
        else:
            runoff_plot = np.array(runoff)

        obs_plot = None
        if obs is not None:
            if obs.ndim > 1:
                obs_plot = np.array(obs[:, 0])
            else:
                obs_plot = np.array(obs)

        # Determine warmup for plot
        if fm_config.date_start_eval and fm_config.date_start_sim:
            warmup_plot = (fm_config.date_start_eval - fm_config.date_start_sim).days
        else:
            warmup_plot = 365

        plot_file = output_file.parent / f"{fm_config.model_id}_hydrograph.png"
        plot_hydrograph(
            times,
            runoff_plot,
            obs_plot,
            output_path=str(plot_file),
            title=f"{basin_id} - {fm_config.model_id}",
            warmup_days=warmup_plot,
            metrics=metrics if metrics else None,
        )

    return {
        "runoff": runoff,
        "times": times,
        "obs": obs,
        "metrics": metrics,
        "final_state": final_state,
        "params": params,
        "elapsed": elapsed,
    }


def run_calibration(
    fm_config: "FileManagerConfig",
    basin_id: str,
    method: str = "gradient",
    verbose: bool = True,
    network_path: Optional[str] = None,
    obs_agg: str = "first",
    obs_file: Optional[str] = None,
    learning_rate: float = 0.01,
    epochs: int = 500,
    loss_fn: str = "kge",
    plot: bool = False,
) -> Dict[str, Any]:
    """Run jFUSE calibration.

    Args:
        fm_config: File manager configuration
        basin_id: Basin identifier
        method: Calibration method ('gradient' or 'sce')
        verbose: Print progress messages
        network_path: Path to network topology file (overrides default naming)
        obs_agg: How to aggregate 2D observations ('first', 'last', 'mean', 'sum')
        obs_file: Path to CSV file with outlet observations (overrides NetCDF obs)
        learning_rate: Learning rate for gradient-based calibration
        epochs: Number of calibration iterations
        loss_fn: Loss function(s) - single or comma-separated for multi-objective
        plot: Generate hydrograph plot

    Returns:
        Dictionary with calibration results
    """
    from jfuse import FUSEModel
    from jfuse.fuse import FUSEConfig
    from jfuse.fuse.config import RoutingType
    from jfuse.optim import Calibrator, CalibrationConfig
    from jfuse.io.filemanager import ForcingInfo, parse_forcing_info

    if verbose:
        print(f"\n{'='*60}")
        print("jFUSE Calibration")
        print(f"{'='*60}")
        print(f"Basin: {basin_id}")
        print(f"Method: {method}")
        print(f"Metric: {fm_config.metric}")

    # Load model configuration
    config = FUSEConfig.from_file(str(fm_config.decisions_path))

    if verbose:
        print("\nModel structure:")
        print(config.describe())

    # Load forcing
    forcing_info = ForcingInfo()
    if fm_config.forcing_info_path.exists():
        forcing_info = parse_forcing_info(str(fm_config.forcing_info_path))

    forcing_file = fm_config.forcing_file(basin_id)
    precip, pet, temp, times, obs, found_vars = load_forcing_netcdf(
        str(forcing_file),
        forcing_info,
        fm_config.date_start_sim,
        fm_config.date_end_sim,
    )

    # Detect number of HRUs and spatial mode
    if precip.ndim == 1:
        n_hrus = 1
        is_distributed = False
    else:
        n_hrus = precip.shape[1]
        is_distributed = n_hrus > 1  # Only distributed if MORE than 1 HRU

        # If only 1 HRU but 2D array, squeeze to 1D (lumped)
        if n_hrus == 1:
            precip = precip.squeeze(-1)
            pet = pet.squeeze(-1)
            temp = temp.squeeze(-1)

    # Check if CHANNEL routing should be enabled
    use_channel_routing = False
    if network_path:
        network_file = Path(network_path)
    else:
        network_file = fm_config.network_file(basin_id)

    if is_distributed:
        network_exists = network_file.exists()

        if network_exists:
            use_channel_routing = True
        else:
            if verbose:
                print("\n  No network file found - calibrating with area-weighted HRU aggregation")

    # Determine hillslope routing type from config
    hillslope_routing = "gamma" if config.routing == RoutingType.GAMMA else "none"

    if use_channel_routing:
        # Use CoupledModel for FUSE + Muskingum-Cunge channel routing
        from jfuse.coupled import CoupledModel
        from jfuse.io import load_network

        if verbose:
            print(f"\nLoading network topology from: {network_file}")

        try:
            network, hru_areas = load_network(str(network_file))
        except KeyError as e:
            print(f"  ERROR: Could not load network file: {e}")
            print("  Falling back to area-weighted HRU aggregation")
            use_channel_routing = False
        except Exception as e:
            print(f"  ERROR: Failed to load network: {e}")
            print("  Falling back to area-weighted HRU aggregation")
            use_channel_routing = False

    if use_channel_routing:
        # Check HRU count matches
        if len(network.reaches) != n_hrus:
            if verbose:
                print(
                    f"  WARNING: Network has {len(network.reaches)} reaches but forcing has {n_hrus} HRUs"
                )

        # Debug: Check outlet detection
        network_arrays = network.to_arrays()
        n_outlets = int(jnp.sum(network_arrays.is_outlet))
        if verbose:
            print(f"  Outlets detected: {n_outlets}")
            if n_outlets == 0:
                print("  WARNING: No outlets found in network! Check downstream_id values.")

        model = CoupledModel(
            fuse_config=config,
            network=network_arrays,
            hru_areas=hru_areas,
            n_hrus=n_hrus,
        )

        if verbose:
            print(f"  Reaches: {len(network.reaches)}")
            print("  Channel routing: Muskingum-Cunge")
    else:
        # Use FUSEModel only (no channel routing)
        model = FUSEModel(config, n_hrus=n_hrus)

    forcing = (precip, pet, temp)

    # Load observations from CSV if provided AND we have a network for unit conversion
    # For lumped mode, always use NetCDF observations (already in mm/day)
    if obs_file and use_channel_routing:
        if verbose:
            print(f"\nLoading observations from CSV: {obs_file}")

        # Get basin area for unit conversion (m³/s -> mm/day)
        basin_area = float(jnp.sum(hru_areas))

        obs = load_observations_csv(
            obs_file,
            times,
            basin_area,
        )
    elif obs_file and not use_channel_routing:
        if verbose:
            print("\n  NOTE: --obs-file ignored for lumped mode (no network for unit conversion)")
            print("        Using observations from NetCDF file (already in mm/day)")

    if obs is None:
        raise ValueError("No observations available for calibration!")

    # Preprocess observations to 1D if needed
    if obs.ndim > 1:
        if obs.shape[1] == 1:
            obs = obs.squeeze(-1)
        else:
            # Apply aggregation method
            if obs_agg == "first":
                obs = obs[:, 0]
            elif obs_agg == "last":
                obs = obs[:, -1]
            elif obs_agg == "mean":
                obs = jnp.mean(obs, axis=1)
            elif obs_agg == "sum":
                obs = jnp.sum(obs, axis=1)
            else:
                obs = obs[:, 0]

            if verbose:
                print(f"\n  2D obs aggregated using --obs-agg={obs_agg}")
                print(f"  Aggregated obs mean: {float(jnp.nanmean(obs)):.4f} mm/day")

    if verbose:
        print("\nData loaded:")
        print(f"  Timesteps: {len(times)}")
        if is_distributed:
            print(f"  HRUs: {n_hrus} (distributed mode)")
            print(f"  Hillslope routing (Q_TDH): {hillslope_routing}")
            if use_channel_routing:
                print("  Channel routing: Muskingum-Cunge")
            else:
                print("  Channel routing: none (area-weighted aggregation)")
        else:
            print("  Mode: lumped")
            print(f"  Hillslope routing (Q_TDH): {hillslope_routing}")
        print(f"  Mean observed: {float(jnp.nanmean(obs)):.2f} mm/day")

    # Determine warmup
    if fm_config.date_start_eval and fm_config.date_start_sim:
        warmup_days = (fm_config.date_start_eval - fm_config.date_start_sim).days
    else:
        warmup_days = 365

    if verbose:
        print(f"  Warmup period: {warmup_days} days")

    # Parse and validate loss function(s)
    loss_types = [lt.strip().lower() for lt in loss_fn.split(",")]
    valid_losses = ["kge", "nse", "rmse", "mse", "mae"]
    for lt in loss_types:
        if lt not in valid_losses:
            raise ValueError(f"Unknown loss function: {lt}. Available: {valid_losses}")

    # Format for display
    if len(loss_types) == 1:
        loss_display = loss_types[0].upper()
    else:
        loss_display = f"Multi({', '.join(lt.upper() for lt in loss_types)})"

    if method == "gradient":
        calib_config = CalibrationConfig(
            max_iterations=epochs,
            learning_rate=learning_rate,
            optimizer="adam",
            patience=50,
            log_every=max(1, epochs // 20),
        )

        calibrator = Calibrator(model, calib_config)

        if verbose:
            print("\nStarting gradient-based calibration...")
            print(f"  Max iterations: {calib_config.max_iterations}")
            print(f"  Learning rate: {calib_config.learning_rate}")
            print(f"  Optimizer: {calib_config.optimizer}")
            print(f"  Loss function: {loss_display}")

        t0 = time.time()
        result = calibrator.calibrate(
            forcing=forcing,
            observed=obs,
            loss_fn=loss_fn,
            warmup_steps=warmup_days,
            verbose=verbose,
        )
        elapsed = time.time() - t0

        best_params = result["best_params"]
        best_loss = result["best_loss"]

    else:
        raise ValueError(f"Unknown calibration method: {method}")

    if verbose:
        print(f"\nCalibration complete in {elapsed:.1f}s")
        print(f"  Best loss ({loss_display}): {best_loss:.4f}")

    # Run final simulation with best parameters
    from jfuse.coupled import CoupledModel

    if isinstance(model, CoupledModel):
        # CoupledModel returns (outlet_Q, runoff)
        outlet_Q, runoff = model.simulate(forcing, best_params)
        # outlet_Q is in m³/s, runoff is in mm/day
        # For metrics, use aggregated runoff in mm/day (same units as obs)
        runoff_agg = jnp.mean(runoff, axis=1) if runoff.ndim > 1 else runoff
        final_state = None
    else:
        # FUSEModel
        state = model.default_state()
        runoff, final_state = model.simulate(forcing, best_params, state)

        # For distributed case, aggregate runoff across HRUs
        if runoff.ndim > 1:
            runoff_agg = jnp.mean(runoff, axis=1)
        else:
            runoff_agg = runoff

    # Ensure obs is 1D
    obs_1d = obs
    if obs.ndim > 1:
        if obs.shape[1] == 1:
            obs_1d = obs.squeeze()
        else:
            # Apply aggregation method
            if obs_agg == "first":
                obs_1d = obs[:, 0]
            elif obs_agg == "last":
                obs_1d = obs[:, -1]
            elif obs_agg == "mean":
                obs_1d = jnp.mean(obs, axis=1)
            elif obs_agg == "sum":
                obs_1d = jnp.sum(obs, axis=1)
            else:
                obs_1d = obs[:, 0]  # Default to first

    # Compute final metrics
    sim_eval = runoff_agg[warmup_days:]
    obs_eval = obs_1d[warmup_days:]

    # Ensure both are 1D
    sim_eval = jnp.atleast_1d(sim_eval).flatten()
    obs_eval = jnp.atleast_1d(obs_eval).flatten()

    valid = ~(jnp.isnan(sim_eval) | jnp.isnan(obs_eval))
    sim_valid = sim_eval[valid]
    obs_valid = obs_eval[valid]

    if len(sim_valid) > 0:
        # NSE (undefined when observations are constant -> ss_tot == 0)
        ss_res = jnp.sum((obs_valid - sim_valid) ** 2)
        ss_tot = jnp.sum((obs_valid - jnp.mean(obs_valid)) ** 2)
        nse = jnp.where(ss_tot > 0, 1 - ss_res / ss_tot, jnp.nan)

        # KGE (alpha/beta undefined when obs std/mean are zero)
        std_obs = jnp.std(obs_valid)
        mean_obs = jnp.mean(obs_valid)
        r = jnp.corrcoef(sim_valid, obs_valid)[0, 1]
        alpha = jnp.where(std_obs > 0, jnp.std(sim_valid) / std_obs, jnp.nan)
        beta = jnp.where(mean_obs != 0, jnp.mean(sim_valid) / mean_obs, jnp.nan)
        kge = 1 - jnp.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

        metrics = {
            "nse": float(nse),
            "kge": float(kge),
            "r": float(r),
            "alpha": float(alpha),
            "beta": float(beta),
        }
    else:
        warnings.warn(
            "No valid (non-NaN) overlapping timesteps between simulation and "
            "observations after warmup; metrics are undefined.",
            UserWarning,
        )
        metrics = {k: float("nan") for k in ("nse", "kge", "r", "alpha", "beta")}

    if verbose:
        print("\nFinal performance:")
        print(f"  NSE: {metrics['nse']:.4f}")
        print(f"  KGE: {metrics['kge']:.4f}")

    # Save output
    fm_config.output_path.mkdir(parents=True, exist_ok=True)
    output_file = fm_config.output_file("_calib_output.nc")

    if verbose:
        print(f"\nSaving output to: {output_file}")

    save_output_netcdf(
        str(output_file),
        times,
        runoff,
        obs,
        final_state,
        metrics,
        fm_config,
        config,
        calibrated_params=best_params,
    )

    # Save parameters
    params_file = fm_config.output_file("_calib_params.txt")
    save_parameters(str(params_file), best_params)

    if verbose:
        print(f"Parameters saved to: {params_file}")

    # Generate plot if requested
    if plot:
        if verbose:
            print("\nGenerating hydrograph plot...")

        # Get 1D arrays for plotting
        if runoff.ndim > 1:
            runoff_plot = np.array(jnp.mean(runoff, axis=1))
        else:
            runoff_plot = np.array(runoff)

        obs_plot = None
        if obs is not None:
            if obs.ndim > 1:
                obs_plot = np.array(obs[:, 0])
            else:
                obs_plot = np.array(obs)

        plot_file = output_file.parent / f"{fm_config.model_id}_calib_hydrograph.png"
        plot_hydrograph(
            times,
            runoff_plot,
            obs_plot,
            output_path=str(plot_file),
            title=f"{basin_id} - Calibrated ({fm_config.model_id})",
            warmup_days=warmup_days,
            metrics=metrics,
        )

    return {
        "runoff": runoff,
        "times": times,
        "obs": obs,
        "metrics": metrics,
        "params": best_params,
        "loss_history": result.get("loss_history", []),
        "elapsed": elapsed,
    }


def save_output_netcdf(
    filepath: str,
    times: np.ndarray,
    runoff: jnp.ndarray,
    obs: Optional[jnp.ndarray],
    final_state: Any,
    metrics: Dict[str, float],
    fm_config: "FileManagerConfig",
    model_config: Any,
    calibrated_params: Any = None,
):
    """Save simulation output to NetCDF."""
    import xarray as xr

    runoff_arr = np.array(runoff)

    # Handle dimensions based on runoff shape
    if runoff_arr.ndim == 1:
        # Lumped case: (time,)
        data_vars = {
            "q_sim": (["time"], runoff_arr),
        }
        coords = {"time": times}
    else:
        # Distributed case: (time, hru)
        n_hrus = runoff_arr.shape[1]
        data_vars = {
            "q_sim": (["time", "hru"], runoff_arr),
            "q_sim_mean": (["time"], runoff_arr.mean(axis=1)),
        }
        coords = {
            "time": times,
            "hru": np.arange(n_hrus),
        }

    # Create dataset
    ds = xr.Dataset(
        data_vars,
        coords=coords,
        attrs={
            "title": "jFUSE simulation output",
            "model_id": fm_config.model_id,
            "created": datetime.now().isoformat(),
            "upper_arch": model_config.upper_arch.name,
            "lower_arch": model_config.lower_arch.name,
            "baseflow": model_config.baseflow.name,
            "percolation": model_config.percolation.name,
            "surface_runoff": model_config.surface_runoff.name,
        },
    )

    ds["q_sim"].attrs = {
        "long_name": "Simulated runoff",
        "units": "mm/day",
    }

    if "q_sim_mean" in ds:
        ds["q_sim_mean"].attrs = {
            "long_name": "Mean simulated runoff across HRUs",
            "units": "mm/day",
        }

    if obs is not None:
        obs_arr = np.array(obs)
        # Ensure obs is 1D for saving
        if obs_arr.ndim == 1:
            obs_1d = obs_arr
        elif obs_arr.ndim == 2:
            if obs_arr.shape[1] == 1:
                obs_1d = obs_arr.squeeze()
            elif obs_arr.shape[0] == 1:
                obs_1d = obs_arr.squeeze()
            else:
                # 2D obs with multiple columns - take first column (outlet)
                obs_1d = obs_arr[:, 0]
        else:
            obs_1d = obs_arr.flatten()[: len(times)]

        ds["q_obs"] = (["time"], obs_1d)
        ds["q_obs"].attrs = {
            "long_name": "Observed runoff",
            "units": "mm/day",
        }

    # Add metrics as attributes
    for name, value in metrics.items():
        ds.attrs[name] = value

    # Save
    ds.to_netcdf(filepath)
    ds.close()


def save_parameters(filepath: str, params: Any):
    """Save calibrated parameters to text file."""
    with open(filepath, "w") as f:
        f.write("# jFUSE Calibrated Parameters\n")
        f.write(f"# Created: {datetime.now().isoformat()}\n")
        f.write("#" + "=" * 50 + "\n")

        for name in dir(params):
            if not name.startswith("_"):
                try:
                    value = getattr(params, name)
                    if not callable(value):
                        f.write(f"{name} = {float(value):.6f}\n")
                except (TypeError, ValueError):
                    pass


def print_info(fm_config: "FileManagerConfig"):
    """Print file manager configuration info."""
    print(f"\n{'='*60}")
    print("jFUSE File Manager Configuration")
    print(f"{'='*60}")

    print("\nPaths:")
    print(f"  Settings: {fm_config.settings_path}")
    print(f"  Input: {fm_config.input_path}")
    print(f"  Output: {fm_config.output_path}")

    print("\nSettings files:")
    print(f"  Decisions: {fm_config.decisions_file}")
    print(f"  Forcing info: {fm_config.forcing_info_file}")
    print(f"  Constraints: {fm_config.constraints_file}")

    print("\nSimulation period:")
    print(f"  Start: {fm_config.date_start_sim}")
    print(f"  End: {fm_config.date_end_sim}")

    print("\nEvaluation:")
    print(f"  Start: {fm_config.date_start_eval}")
    print(f"  End: {fm_config.date_end_eval}")
    print(f"  Metric: {fm_config.metric}")

    # Check if files exist
    print("\nFile status:")
    decisions_path = fm_config.decisions_path
    print(f"  Decisions file: {'✓' if decisions_path.exists() else '✗'} {decisions_path}")

    # Try to load and show model structure
    if decisions_path.exists():
        from jfuse.fuse import FUSEConfig

        config = FUSEConfig.from_file(str(decisions_path))
        print(f"\n{config.describe()}")


def print_structures():
    """Print all available model structures."""
    from jfuse.fuse import enumerate_all_configs

    configs = enumerate_all_configs()

    print(f"\n{'='*60}")
    print("jFUSE Available Model Structures")
    print(f"{'='*60}")
    print(f"\nTotal combinations: {len(configs)}")
    print("\nFirst 20 structures:")

    for i, (name, config) in enumerate(list(configs.items())[:20]):
        print(f"  {i+1}. {name}")

    print(f"\n... and {len(configs) - 20} more")

    print("\nTo create a custom structure, edit the decisions file with:")
    print("  ARCH1: onestate_1, tension1_1, tension2_1")
    print("  ARCH2: unlimfrc_2, unlimpow_2, fixedsiz_2, tens2pll_2")
    print("  QSURF: prms_varnt, arno_x_vic, tmdl_param")
    print("  QPERC: perc_f2sat, perc_lower, perc_w2sat")
    print("  ESOIL: sequential, rootweight")
    print("  QINTF: intflwnone, intflwsome")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="jFUSE: JAX Implementation of FUSE Hydrological Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run simulation
  jfuse run fm_catch.txt bow_at_banff
  
  # Run simulation with routing (requires network file and Q_TDH=rout_gamma in decisions)
  jfuse run fm_catch.txt bow_at_banff --network=basin_network.nc
  
  # Run calibration with gradient descent
  jfuse run fm_catch.txt bow_at_banff --mode=calib --method=gradient
  
  # Show file manager info
  jfuse info fm_catch.txt
  
  # List all model structures
  jfuse structures

Routing:
  For distributed simulations, routing is automatically enabled when:
  1. Q_TDH is set to 'rout_gamma' in the decisions file
  2. A network topology file exists (basin_id_network.nc)
  
  Use --network to specify a custom network file path.
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run simulation or calibration")
    run_parser.add_argument("filemanager", help="Path to file manager file")
    run_parser.add_argument("basin_id", help="Basin identifier")
    run_parser.add_argument(
        "--mode",
        choices=["sim", "calib"],
        default="sim",
        help="Run mode: sim (simulation) or calib (calibration)",
    )
    run_parser.add_argument(
        "--method",
        choices=["gradient", "sce"],
        default="gradient",
        help="Calibration method (for --mode=calib)",
    )
    run_parser.add_argument(
        "--network",
        type=str,
        default=None,
        help="Path to network topology file (overrides default naming)",
    )
    run_parser.add_argument(
        "--obs-agg",
        type=str,
        default="first",
        choices=["first", "last", "mean", "sum"],
        help="How to aggregate 2D observations: first/last column, mean, or sum",
    )
    run_parser.add_argument(
        "--obs-file",
        type=str,
        default=None,
        help="CSV file with outlet observations (columns: datetime, discharge_cms). "
        "Only used for distributed mode with network file (converts m³/s to mm/day). "
        "For lumped mode, uses q_obs from NetCDF file.",
    )
    run_parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate for gradient-based calibration (default: 0.01)",
    )
    run_parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Number of iterations for calibration (default: 500)",
    )
    run_parser.add_argument(
        "--loss",
        type=str,
        default="kge",
        help="Loss function(s) for calibration. Single: kge, nse, rmse, mse, mae. "
        'Multi-objective: comma-separated, e.g., "kge,nse" (default: kge)',
    )
    run_parser.add_argument(
        "--plot", action="store_true", help="Generate hydrograph plot after run"
    )
    run_parser.add_argument(
        "--verbose", "-v", action="store_true", default=True, help="Print progress messages"
    )
    run_parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress messages")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show file manager configuration")
    info_parser.add_argument("filemanager", help="Path to file manager file")

    # Structures command
    subparsers.add_parser("structures", help="List available model structures")

    # Parse arguments
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Import here to avoid slow startup
    from jfuse.io.filemanager import parse_filemanager

    if args.command == "run":
        fm_config = parse_filemanager(args.filemanager)
        verbose = args.verbose and not args.quiet
        network_path = getattr(args, "network", None)
        obs_agg = getattr(args, "obs_agg", "first")
        obs_file = getattr(args, "obs_file", None)
        do_plot = getattr(args, "plot", False)
        learning_rate = getattr(args, "lr", 0.01)
        epochs = getattr(args, "epochs", 500)
        loss = getattr(args, "loss", "kge")

        if args.mode == "sim":
            run_simulation(
                fm_config,
                args.basin_id,
                verbose=verbose,
                network_path=network_path,
                obs_agg=obs_agg,
                obs_file=obs_file,
                plot=do_plot,
            )
        else:
            run_calibration(
                fm_config,
                args.basin_id,
                method=args.method,
                verbose=verbose,
                network_path=network_path,
                obs_agg=obs_agg,
                obs_file=obs_file,
                learning_rate=learning_rate,
                epochs=epochs,
                loss_fn=loss,
                plot=do_plot,
            )

        if verbose:
            print(f"\n{'='*60}")
            print("Done!")
            print(f"{'='*60}")

    elif args.command == "info":
        fm_config = parse_filemanager(args.filemanager)
        print_info(fm_config)

    elif args.command == "structures":
        print_structures()


if __name__ == "__main__":
    main()
