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


class LakeRuleParams(NamedTuple):
    """Global multipliers on the per-reach lake/reservoir operating rules.

    HydroLAKES gives the spatial pattern (per-lake ``q_ref`` from ``Dis_avg``,
    capacity from ``Vol_total``); these scalars let calibration tune the rules
    globally against observed downstream flow, without per-lake operation data:

        q_ref_eff  = q_ref_base  * q_ref_mult
        q_min_eff  = q_min_frac  * q_ref_eff      (min release as a frac of q_ref)
        exp_eff    = exp                          (global rating exponent)
        spill_eff  = spill_base  * spill_mult

    All AD-active. Applied only to reaches flagged ``is_lake``.
    """
    q_ref_mult: Array
    q_min_frac: Array
    exp: Array
    spill_mult: Array


# Bounds for the lake operating-rule multipliers (used by the calibrator).
# Order defines the layout of the trailing lake block in a calibration vector.
LAKE_RULE_BOUNDS = {
    "LAKE_Q_REF_MULT": (0.2, 5.0),
    "LAKE_Q_MIN_FRAC": (0.0, 0.9),
    "LAKE_EXP": (0.5, 5.0),
    "LAKE_SPILL_MULT": (0.1, 10.0),
}
LAKE_RULE_NAMES = tuple(LAKE_RULE_BOUNDS.keys())

# Neutral starting values: q_ref/spill unchanged (×1), a modest min release,
# the standard weir exponent — i.e. "use HydroLAKES defaults as-is" until the
# calibrator moves them.
LAKE_RULE_DEFAULTS = {
    "LAKE_Q_REF_MULT": 1.0,
    "LAKE_Q_MIN_FRAC": 0.1,
    "LAKE_EXP": 2.0,
    "LAKE_SPILL_MULT": 1.0,
}


def build_lake_rules(values) -> "LakeRuleParams":
    """Build :class:`LakeRuleParams` from a length-4 array/sequence ordered as
    :data:`LAKE_RULE_NAMES` (q_ref_mult, q_min_frac, exp, spill_mult).

    Lets a calibrator carry the four lake multipliers as a trailing block of its
    parameter vector and map them back with one call.
    """
    return LakeRuleParams(values[0], values[1], values[2], values[3])


def apply_lake_rules(network: NetworkArrays, lake_rules: Optional["LakeRuleParams"]) -> NetworkArrays:
    """Apply global operating-rule multipliers to a network's lake reaches.

    Returns ``network`` unchanged when ``lake_rules`` is None or the network has
    no lake attributes.
    """
    if lake_rules is None or getattr(network, "is_lake", None) is None:
        return network
    is_lake = network.is_lake
    q_ref = network.lake_q_ref * lake_rules.q_ref_mult
    q_min = lake_rules.q_min_frac * q_ref
    exp = jnp.broadcast_to(lake_rules.exp, network.lake_exp.shape)
    spill = network.lake_spill_coef * lake_rules.spill_mult
    return network._replace(
        lake_q_ref=jnp.where(is_lake, q_ref, network.lake_q_ref),
        lake_q_min=jnp.where(is_lake, q_min, network.lake_q_min),
        lake_exp=jnp.where(is_lake, exp, network.lake_exp),
        lake_spill_coef=jnp.where(is_lake, spill, network.lake_spill_coef),
    )


class CoupledParams(NamedTuple):
    """Combined parameters for coupled model.

    Attributes:
        fuse_params: FUSE model parameters [n_hrus, n_params] or [n_params]
        manning_n: Manning's n for each reach [n_reaches]
        geometry: Optional geometry parameters (width_coef, etc.)
        lake_rules: Optional global lake/reservoir operating-rule multipliers.
    """
    fuse_params: FUSEParameters
    manning_n: Array
    width_coef: Optional[Array] = None
    width_exp: Optional[Array] = None
    depth_coef: Optional[Array] = None
    depth_exp: Optional[Array] = None
    lake_rules: Optional[LakeRuleParams] = None


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
    glacier_frac: Optional[Array] = None,
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
    
    # Run FUSE simulation (glacier_frac, when given and enabled in fuse_config,
    # area-weights each HRU's runoff between soil column and glacier component).
    runoff, final_fuse_state = fuse_simulate(
        forcing_series,
        initial_fuse_state,
        fuse_params,
        fuse_config,
        fuse_dt,
        start_doy,
        glacier_frac=glacier_frac,
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


def coupled_simulate_full(
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
    glacier_frac: Optional[Array] = None,
) -> Tuple[Array, Array, FUSEState]:
    """Coupled FUSE + routing simulation that also returns per-reach discharge.

    Identical to :func:`coupled_simulate` but routes with
    :func:`route_network_full`, exposing ``Q_all`` ``[n_timesteps, n_reaches]``.
    Used by multi-gauge calibration, which reads simulated flow at each gauge's
    reach.

    Returns:
        Tuple ``(outlet_Q, Q_all, final_fuse_state)``.
    """
    if routing_dt is None:
        routing_dt = fuse_dt * 86400.0
    n_hrus = hru_areas.shape[0]
    n_reaches = network.n_reaches
    if initial_fuse_state is None:
        initial_fuse_state = FUSEState.default(n_hrus)
    if initial_Q is None:
        initial_Q = jnp.full(n_reaches, 0.1)

    runoff, final_fuse_state = fuse_simulate(
        forcing_series, initial_fuse_state, fuse_params, fuse_config,
        fuse_dt, start_doy, glacier_frac=glacier_frac,
    )

    lateral_inflow = runoff_to_inflow(runoff, hru_areas, fuse_dt * 86400.0)
    if n_hrus < n_reaches:
        lateral_inflow = jnp.pad(
            lateral_inflow, ((0, 0), (0, n_reaches - n_hrus)),
            mode='constant', constant_values=0.0)
    elif n_hrus > n_reaches:
        base_inflow = lateral_inflow[:, :n_reaches - 1]
        excess_inflow = jnp.sum(lateral_inflow[:, n_reaches - 1:], axis=1, keepdims=True)
        lateral_inflow = jnp.concatenate([base_inflow, excess_inflow], axis=1)

    updated_network = network._replace(manning_n=manning_n)
    from .routing import route_network_full
    outlet_Q, Q_all = route_network_full(
        lateral_inflow, updated_network, routing_dt, initial_Q, n_substeps=n_substeps
    )
    return outlet_Q, Q_all, final_fuse_state


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
    glacier_frac: Optional[Array]
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
        glacier_frac: Optional[Array] = None,
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
        # Static per-HRU glacier fraction (None => no glacier). Aligned to the
        # same HRU ordering as hru_areas / forcing.
        self.glacier_frac = glacier_frac
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
        
        # Update network Manning's n + apply calibrated lake operating rules.
        network = apply_lake_rules(
            self.network._replace(manning_n=params.manning_n),
            getattr(params, "lake_rules", None))

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
            glacier_frac=self.glacier_frac,
        )

        return outlet_Q, runoff

    def simulate_full(
        self,
        forcing_series: Tuple[Array, Array, Array],
        params: CoupledParams,
        initial_state: Optional[CoupledState] = None,
        start_doy: int = 1,
    ) -> Tuple[Array, Array, Array]:
        """Run coupled simulation, returning discharge at every reach.

        Like :meth:`simulate` but exposes per-reach discharge for multi-gauge
        calibration (the loss reads simulated flow at each gauge's reach).

        Returns:
            Tuple ``(outlet_Q, Q_all, runoff)`` where ``Q_all`` is
            ``[n_timesteps, n_reaches]``.
        """
        initial_fuse_state = None
        initial_Q = None
        if initial_state is not None:
            initial_fuse_state = initial_state.fuse_state
            initial_Q = initial_state.router_Q

        network = apply_lake_rules(
            self.network._replace(manning_n=params.manning_n),
            getattr(params, "lake_rules", None))
        outlet_Q, Q_all, final_state = coupled_simulate_full(
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
            glacier_frac=self.glacier_frac,
        )
        return outlet_Q, Q_all, final_state

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
