"""
Coupled FUSE + Routing Model

Combines the FUSE rainfall-runoff model with river routing for end-to-end
differentiable watershed simulation. This enables gradient-based calibration
of both hydrological and routing parameters simultaneously.

The coupling handles:
- Conversion of runoff (mm/day) to lateral inflow (m³/s)
- HRU-to-reach mapping for spatial distribution
- Proper gradient flow through both components
"""

from typing import Tuple, Optional, Dict, Any, NamedTuple
from functools import partial
import math
import warnings

import jax
import jax.numpy as jnp
from jax import Array, lax
import equinox as eqx

from .fuse import (
    FUSEModel,
    State as FUSEState,
    Parameters as FUSEParameters,
    Forcing,
    ModelConfig,
    PRMS_CONFIG,
    fuse_simulate,
)
from .routing import (
    MuskingumCungeRouter,
    NetworkArrays,
    RiverNetwork,
)


class CoupledState(NamedTuple):
    """Combined state for coupled model.
    
    Attributes:
        fuse_state: FUSE model state
        router_Q: Discharge at each reach (m³/s)
    """
    fuse_state: FUSEState
    router_Q: Array


class CoupledParams(NamedTuple):
    """Combined parameters for coupled model.
    
    Attributes:
        fuse_params: FUSE model parameters [n_hrus, n_params] or [n_params]
        manning_n: Manning's n for each reach [n_reaches]
        geometry: Optional geometry parameters (width_coef, etc.)
    """
    fuse_params: FUSEParameters
    manning_n: Array
    width_coef: Optional[Array] = None
    width_exp: Optional[Array] = None
    depth_coef: Optional[Array] = None
    depth_exp: Optional[Array] = None


def runoff_to_inflow(
    runoff_mm: Array,
    areas_m2: Array,
    dt_seconds: float = 86400.0,
) -> Array:
    """Convert runoff depth to volumetric flow rate.
    
    Q [m³/s] = runoff [mm/day] * area [m²] / 1000 / 86400
    
    Args:
        runoff_mm: Runoff in mm/day [n_timesteps, n_hrus]
        areas_m2: HRU areas in m² [n_hrus]
        dt_seconds: Timestep in seconds (default 1 day)
        
    Returns:
        Volumetric flow rate in m³/s [n_timesteps, n_hrus]
    """
    # mm/day to m/day: divide by 1000
    # m³/day = m/day * m²
    # m³/s = m³/day / 86400
    return runoff_mm * areas_m2 / 1000.0 / dt_seconds


def coupled_simulate(
    forcing_series: Tuple[Array, Array, Array],
    fuse_params: FUSEParameters,
    manning_n: Array,
    network: NetworkArrays,
    hru_areas: Array,
    fuse_config: ModelConfig,
    initial_fuse_state: Optional[FUSEState] = None,
    initial_Q: Optional[Array] = None,
    fuse_dt: float = 1.0,
    routing_dt: Optional[float] = None,
    n_substeps: int = 1,
    start_doy: int = 1,
) -> Tuple[Array, Array, FUSEState]:
    """Run coupled FUSE + routing simulation.
    
    This is the main simulation function that:
    1. Runs FUSE to generate runoff for each HRU
    2. Converts runoff to lateral inflow (m³/s)
    3. Routes through the river network
    
    Args:
        forcing_series: Tuple of (precip, pet, temp) arrays [n_timesteps, n_hrus]
        fuse_params: FUSE model parameters
        manning_n: Manning's n for each reach [n_reaches]
        network: River network topology
        hru_areas: HRU areas in m² [n_hrus]
        fuse_config: FUSE model configuration
        initial_fuse_state: Initial FUSE state (optional)
        initial_Q: Initial discharge [n_reaches] (optional)
        fuse_dt: FUSE timestep in days (default 1)
        routing_dt: Routing timestep in seconds. Defaults to the FUSE step
            interval (fuse_dt * 86400) so the router advances in step with the
            inflow series it is given. Routing a daily inflow series at a
            smaller dt (e.g. 3600 s) with no sub-stepping over-attenuates and
            over-lags the hydrograph by fuse_dt*86400 / routing_dt.
        n_substeps: Number of Muskingum sub-steps per FUSE timestep for routing
            stability (see route_network). Default 1 (no sub-stepping).
        start_doy: Starting day of year
        
    Returns:
        Tuple of (outlet_Q, runoff, final_fuse_state)
        - outlet_Q: Outlet discharge [n_timesteps] in m³/s
        - runoff: Runoff from each HRU [n_timesteps, n_hrus] in mm/day
        - final_fuse_state: Final FUSE state
    """
    precip, pet, temp = forcing_series
    n_timesteps = precip.shape[0]
    n_hrus = hru_areas.shape[0]
    n_reaches = network.n_reaches

    # Route in step with the inflow series: one routing step per FUSE timestep.
    if routing_dt is None:
        routing_dt = fuse_dt * 86400.0
    
    # Initialize FUSE state
    if initial_fuse_state is None:
        initial_fuse_state = FUSEState.default(n_hrus)
    
    # Initialize routing
    if initial_Q is None:
        initial_Q = jnp.full(n_reaches, 0.1)
    
    # Run FUSE simulation
    runoff, final_fuse_state = fuse_simulate(
        forcing_series,
        initial_fuse_state,
        fuse_params,
        fuse_config,
        fuse_dt,
        start_doy,
    )
    
    # Convert runoff to lateral inflow (m³/s)
    # Assume HRU i maps to reach i (can be customized via hru_to_reach mapping)
    lateral_inflow = runoff_to_inflow(runoff, hru_areas, fuse_dt * 86400.0)

    # Handle HRU-to-reach dimension mismatch
    if n_hrus != n_reaches:
        warnings.warn(
            f"HRU count ({n_hrus}) does not match reach count ({n_reaches}). "
            f"Using automatic mapping: {'padding with zeros' if n_hrus < n_reaches else 'aggregating to last reach'}. "
            f"For explicit control, provide hru_to_reach mapping.",
            UserWarning,
        )

    if n_hrus < n_reaches:
        # Pad with zeros for reaches without HRU inflow
        lateral_inflow = jnp.pad(
            lateral_inflow,
            ((0, 0), (0, n_reaches - n_hrus)),
            mode='constant',
            constant_values=0.0,
        )
    elif n_hrus > n_reaches:
        # Aggregate excess HRUs to last reach to preserve water balance
        # This maintains total water volume while fitting to network structure
        base_inflow = lateral_inflow[:, :n_reaches - 1]
        excess_inflow = jnp.sum(lateral_inflow[:, n_reaches - 1:], axis=1, keepdims=True)
        lateral_inflow = jnp.concatenate([base_inflow, excess_inflow], axis=1)
    
    # Update network with current manning_n
    # Create updated network arrays with calibrated Manning's n
    updated_network = network._replace(manning_n=manning_n)
    
    # Route through network
    from .routing import route_network
    outlet_Q = route_network(
        lateral_inflow, updated_network, routing_dt, initial_Q, n_substeps=n_substeps
    )
    
    return outlet_Q, runoff, final_fuse_state


def _resolve_n_substeps(
    method: str,
    max_substeps: int,
    network: Optional[NetworkArrays],
    dt: float,
) -> int:
    """Resolve a static Muskingum sub-step count for routing stability.

    'fixed' uses ``max_substeps`` directly. 'adaptive' targets a sub-step close
    to the shortest reach travel time (evaluated at a reference discharge) so
    the Courant number stays near 1, capped at ``max_substeps``. Returns 1 when
    sub-stepping is disabled or no network is available.
    """
    from .routing.router import compute_celerity

    max_substeps = int(max_substeps)
    if network is None or max_substeps <= 1:
        return 1
    if method == 'fixed':
        return max(1, max_substeps)

    # adaptive: estimate the shortest reach travel time at a reference discharge.
    celerity = compute_celerity(
        1.0,
        network.slopes,
        network.manning_n,
        network.width_coef,
        network.width_exp,
        network.depth_coef,
        network.depth_exp,
    )
    travel_times = network.lengths / jnp.maximum(celerity, 1e-3)
    k_min = float(jnp.min(travel_times))
    if not (k_min > 0.0):
        return max(1, max_substeps)
    n = math.ceil(dt / k_min)
    return int(min(max(n, 1), max_substeps))


class CoupledModel(eqx.Module):
    """Coupled FUSE + routing model for end-to-end simulation and calibration.
    
    This class provides a convenient interface for:
    - Running coupled simulations
    - Computing gradients for calibration
    - Loading from NetCDF files
    
    Attributes:
        fuse_model: FUSE rainfall-runoff model
        router: Muskingum-Cunge router
        network: River network topology
        hru_areas: HRU contributing areas (m²)
        hru_to_reach: Mapping from HRU indices to reach indices
    """
    fuse_model: FUSEModel
    network: NetworkArrays
    hru_areas: Array
    routing_dt: Optional[float] = eqx.field(static=True)
    n_substeps: int = eqx.field(static=True)

    def __init__(
        self,
        fuse_config: ModelConfig = None,
        network: NetworkArrays = None,
        hru_areas: Array = None,
        n_hrus: int = 1,
        routing_dt: Optional[float] = None,
        routing_substep_method: str = 'adaptive',
        routing_max_substeps: int = 10,
    ):
        """Initialize coupled model.
        
        Args:
            fuse_config: FUSE model configuration (default PRMS)
            network: River network topology
            hru_areas: HRU areas in m²
            n_hrus: Number of HRUs (used if hru_areas not provided)
            routing_dt: Routing timestep in seconds. Defaults to None, which
                routes one step per FUSE day (86400 s); see coupled_simulate.
            routing_substep_method: 'adaptive' (sub-steps chosen from reach
                travel times) or 'fixed' (always routing_max_substeps).
            routing_max_substeps: Maximum Muskingum sub-steps per FUSE step.
                Set to 1 to disable sub-stepping.
        """
        if fuse_config is None:
            fuse_config = PRMS_CONFIG
        
        if hru_areas is None:
            hru_areas = jnp.ones(n_hrus) * 1e6  # Default 1 km²
        
        self.fuse_model = FUSEModel(config=fuse_config, n_hrus=len(hru_areas))
        self.network = network
        self.hru_areas = hru_areas
        self.routing_dt = routing_dt
        # Resolve a static sub-step count from the (concrete) network geometry.
        self.n_substeps = _resolve_n_substeps(
            routing_substep_method,
            routing_max_substeps,
            network,
            routing_dt if routing_dt is not None else 86400.0,
        )
    
    @classmethod
    def from_netcdf(
        cls,
        forcing_path: str,
        network_path: str,
        config: ModelConfig = None,
    ) -> "CoupledModel":
        """Create model from NetCDF files.
        
        Args:
            forcing_path: Path to forcing NetCDF (contains HRU info)
            network_path: Path to network topology NetCDF
            config: FUSE model configuration
            
        Returns:
            Configured CoupledModel
        """
        from .io import load_forcing, load_network
        
        # Load data
        forcing_data = load_forcing(forcing_path)
        network, hru_areas = load_network(network_path)
        
        return cls(
            fuse_config=config,
            network=network.to_arrays(),
            hru_areas=hru_areas,
            n_hrus=len(hru_areas),
        )
    
    def default_params(self) -> CoupledParams:
        """Get default parameters for both FUSE and routing."""
        fuse_params = self.fuse_model.default_params()
        manning_n = self.network.manning_n
        
        return CoupledParams(
            fuse_params=fuse_params,
            manning_n=manning_n,
        )
    
    def simulate(
        self,
        forcing_series: Tuple[Array, Array, Array],
        params: CoupledParams,
        initial_state: Optional[CoupledState] = None,
        start_doy: int = 1,
    ) -> Tuple[Array, Array]:
        """Run coupled simulation.
        
        Args:
            forcing_series: Tuple of (precip, pet, temp) [n_timesteps, n_hrus]
            params: Coupled parameters
            initial_state: Initial state (optional)
            start_doy: Starting day of year
            
        Returns:
            Tuple of (outlet_Q, runoff) where:
            - outlet_Q: [n_timesteps] in m³/s
            - runoff: [n_timesteps, n_hrus] in mm/day
        """
        initial_fuse_state = None
        initial_Q = None
        
        if initial_state is not None:
            initial_fuse_state = initial_state.fuse_state
            initial_Q = initial_state.router_Q
        
        # Update network Manning's n
        network = self.network._replace(manning_n=params.manning_n)
        
        outlet_Q, runoff, _ = coupled_simulate(
            forcing_series,
            params.fuse_params,
            params.manning_n,
            network,
            self.hru_areas,
            self.fuse_model.config,
            initial_fuse_state,
            initial_Q,
            fuse_dt=1.0,
            routing_dt=self.routing_dt,
            n_substeps=self.n_substeps,
            start_doy=start_doy,
        )
        
        return outlet_Q, runoff
    
    @property
    def n_hrus(self) -> int:
        """Number of HRUs."""
        return self.fuse_model.n_hrus
    
    @property
    def n_reaches(self) -> int:
        """Number of reaches."""
        return self.network.n_reaches


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def nse_loss(simulated: Array, observed: Array, warmup: int = 0) -> Array:
    """Nash-Sutcliffe Efficiency loss (1 - NSE).
    
    Args:
        simulated: Simulated discharge [n_timesteps]
        observed: Observed discharge [n_timesteps]
        warmup: Number of warmup timesteps to exclude
        
    Returns:
        1 - NSE (lower is better)
    """
    sim = simulated[warmup:]
    obs = observed[warmup:]
    
    # Handle NaN in observations
    valid = ~jnp.isnan(obs)
    sim_v = jnp.where(valid, sim, 0.0)
    obs_v = jnp.where(valid, obs, 0.0)
    n_valid = jnp.sum(valid)
    
    obs_mean = jnp.sum(obs_v) / jnp.maximum(n_valid, 1.0)
    
    ss_res = jnp.sum(jnp.where(valid, (sim_v - obs_v) ** 2, 0.0))
    ss_tot = jnp.sum(jnp.where(valid, (obs_v - obs_mean) ** 2, 0.0))
    
    return ss_res / jnp.maximum(ss_tot, 1e-10)


def mse_loss(simulated: Array, observed: Array, warmup: int = 0) -> Array:
    """Mean Squared Error loss.
    
    Args:
        simulated: Simulated discharge [n_timesteps]
        observed: Observed discharge [n_timesteps]
        warmup: Number of warmup timesteps to exclude
        
    Returns:
        MSE (lower is better)
    """
    sim = simulated[warmup:]
    obs = observed[warmup:]
    
    valid = ~jnp.isnan(obs)
    sim_v = jnp.where(valid, sim, 0.0)
    obs_v = jnp.where(valid, obs, 0.0)
    n_valid = jnp.sum(valid)
    
    return jnp.sum(jnp.where(valid, (sim_v - obs_v) ** 2, 0.0)) / jnp.maximum(n_valid, 1.0)


def rmse_loss(simulated: Array, observed: Array, warmup: int = 0) -> Array:
    """Root Mean Squared Error loss.
    
    Args:
        simulated: Simulated discharge [n_timesteps]
        observed: Observed discharge [n_timesteps]
        warmup: Number of warmup timesteps to exclude
        
    Returns:
        RMSE (lower is better)
    """
    return jnp.sqrt(mse_loss(simulated, observed, warmup))


def mae_loss(simulated: Array, observed: Array, warmup: int = 0) -> Array:
    """Mean Absolute Error loss.
    
    Args:
        simulated: Simulated discharge [n_timesteps]
        observed: Observed discharge [n_timesteps]
        warmup: Number of warmup timesteps to exclude
        
    Returns:
        MAE (lower is better)
    """
    sim = simulated[warmup:]
    obs = observed[warmup:]
    
    valid = ~jnp.isnan(obs)
    sim_v = jnp.where(valid, sim, 0.0)
    obs_v = jnp.where(valid, obs, 0.0)
    n_valid = jnp.sum(valid)
    
    return jnp.sum(jnp.where(valid, jnp.abs(sim_v - obs_v), 0.0)) / jnp.maximum(n_valid, 1.0)


def kge_loss(simulated: Array, observed: Array, warmup: int = 0) -> Array:
    """Kling-Gupta Efficiency loss (1 - KGE).
    
    KGE = 1 - sqrt((r-1)² + (α-1)² + (β-1)²)
    
    where:
        r = correlation coefficient
        α = σ_sim / σ_obs (variability ratio)
        β = μ_sim / μ_obs (bias ratio)
    """
    sim = simulated[warmup:]
    obs = observed[warmup:]
    
    valid = ~jnp.isnan(obs)
    sim_v = jnp.where(valid, sim, 0.0)
    obs_v = jnp.where(valid, obs, 0.0)
    n_valid = jnp.sum(valid)
    
    sim_mean = jnp.sum(sim_v) / jnp.maximum(n_valid, 1.0)
    obs_mean = jnp.sum(obs_v) / jnp.maximum(n_valid, 1.0)
    
    sim_std = jnp.sqrt(jnp.sum(jnp.where(valid, (sim_v - sim_mean) ** 2, 0.0)) / jnp.maximum(n_valid - 1, 1.0))
    obs_std = jnp.sqrt(jnp.sum(jnp.where(valid, (obs_v - obs_mean) ** 2, 0.0)) / jnp.maximum(n_valid - 1, 1.0))
    
    # Correlation
    cov = jnp.sum(jnp.where(valid, (sim_v - sim_mean) * (obs_v - obs_mean), 0.0)) / jnp.maximum(n_valid - 1, 1.0)
    r = cov / jnp.maximum(sim_std * obs_std, 1e-10)
    
    # Variability and bias ratios
    alpha = sim_std / jnp.maximum(obs_std, 1e-10)
    beta = sim_mean / jnp.maximum(obs_mean, 1e-10)
    
    return jnp.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def coupled_loss(
    params: CoupledParams,
    model: CoupledModel,
    forcing: Tuple[Array, Array, Array],
    observed: Array,
    warmup: int = 365,
    loss_type: str = "nse",
) -> Array:
    """Compute loss for coupled model.
    
    This function is designed to be differentiated with jax.grad().
    
    Args:
        params: Model parameters to optimize
        model: Coupled model
        forcing: Forcing data (precip, pet, temp)
        observed: Observed discharge
        warmup: Warmup period
        loss_type: "nse" or "kge"
        
    Returns:
        Loss value (scalar)
    """
    outlet_Q, _ = model.simulate(forcing, params)
    
    if loss_type == "nse":
        return nse_loss(outlet_Q, observed, warmup)
    else:
        return kge_loss(outlet_Q, observed, warmup)


def value_and_grad_loss(
    model: CoupledModel,
    params: CoupledParams,
    forcing: Tuple[Array, Array, Array],
    observed: Array,
    warmup: int = 365,
    loss_type: str = "nse",
) -> Tuple[Array, CoupledParams]:
    """Compute loss and gradients for calibration.
    
    Args:
        model: Coupled model
        params: Current parameters
        forcing: Forcing data
        observed: Observed discharge
        warmup: Warmup period
        loss_type: "nse" or "kge"
        
    Returns:
        Tuple of (loss_value, parameter_gradients)
    """
    loss_fn = partial(coupled_loss, model=model, forcing=forcing, 
                      observed=observed, warmup=warmup, loss_type=loss_type)
    
    return jax.value_and_grad(loss_fn)(params)
