"""
FUSE Model Implementation

Main model class that orchestrates the physics computations for a single
timestep and full simulation. Supports both single-HRU and batched operations.

The model is implemented as pure functions that can be JIT-compiled and
differentiated through.
"""

from typing import Tuple, Optional

import jax
import jax.numpy as jnp
from jax import Array, lax
import equinox as eqx

from .config import (
    ModelConfig,
    UpperLayerArch,
    LowerLayerArch,
    BaseflowType,
    PercolationType,
    SurfaceRunoffType,
    EvaporationType,
    InterflowType,
    PRMS_CONFIG,
)
from .state import State, Flux, Parameters, Forcing
from . import physics


# =============================================================================
# SINGLE TIMESTEP COMPUTATION
# =============================================================================

def fuse_step(
    state: State,
    forcing: Forcing,
    params: Parameters,
    config: ModelConfig,
    dt: float = 1.0,
    day_of_year: int = 1,
) -> Tuple[State, Flux]:
    """Compute one timestep of the FUSE model.
    
    This is the core model computation that advances state by one timestep.
    All physics are computed using smooth differentiable approximations.
    
    Args:
        state: Current model state
        forcing: Meteorological forcing for this timestep
        params: Model parameters
        config: Model configuration
        dt: Timestep in days (default 1.0)
        day_of_year: Day of year for seasonal snow melt (1-365)
        
    Returns:
        Tuple of (new_state, fluxes)
    """
    # Initialize flux output
    shape = jnp.broadcast_shapes(
        jnp.asarray(state.S1).shape,
        jnp.asarray(forcing.precip).shape
    )
    
    # =========================================================================
    # SNOW MODULE
    # =========================================================================
    if config.enable_snow:
        rain, melt, SWE_new = physics.compute_snow(
            forcing.precip,
            forcing.temp,
            state.SWE,
            params.T_rain,
            params.T_melt,
            params.melt_rate,
            day_of_year,
            params.MFMAX,
            params.MFMIN,
        )
        throughfall = rain + melt
    else:
        rain = forcing.precip
        melt = jnp.zeros(shape)
        throughfall = forcing.precip
        SWE_new = state.SWE
    
    # =========================================================================
    # SATURATED AREA & SURFACE RUNOFF (Eq 9, 11)
    # =========================================================================
    if config.surface_runoff == SurfaceRunoffType.UZ_LINEAR:
        Ac = physics.compute_satarea_linear(state.S1_T, params.S1_T_max, params.Ac_max)
    elif config.surface_runoff == SurfaceRunoffType.UZ_PARETO:
        Ac = physics.compute_satarea_pareto(state.S1, params.S1_max, params.b, params.Ac_max)
    else:  # LZ_GAMMA (TOPMODEL)
        Ac = physics.compute_satarea_topmodel(state.S2, params.S2_max, params.chi, params.Ac_max)
    
    qsx = physics.compute_surface_runoff(throughfall, Ac)
    infiltration = physics.smooth_max(throughfall - qsx, 0.0, 0.001)
    
    # =========================================================================
    # EVAPORATION (Eq 3)
    # =========================================================================
    if config.evaporation == EvaporationType.SEQUENTIAL:
        e1, e2 = physics.compute_evaporation_sequential(
            forcing.pet, state.S1, state.S2, params.S1_max, params.S2_max
        )
    else:  # ROOT_WEIGHT
        e1, e2 = physics.compute_evaporation_root_weighted(
            forcing.pet, state.S1, state.S2, params.S1_max, params.S2_max, params.r1
        )
    
    # Lower layer evap only if architecture supports it
    if not config.has_lower_evap:
        e2 = jnp.zeros_like(e2)
    
    # =========================================================================
    # PERCOLATION (Eq 4)
    # =========================================================================
    if config.percolation == PercolationType.TOTAL_STORAGE:
        q12 = physics.compute_percolation_total_storage(
            state.S1, params.S1_max, params.ku, params.c
        )
    elif config.percolation == PercolationType.FREE_STORAGE:
        q12 = physics.compute_percolation_free_storage(
            state.S1_F, params.S1_F_max, params.ku
        )
    else:  # LOWER_DEMAND
        q12 = physics.compute_percolation_lower_demand(
            state.S1_F, state.S2, params.S2_max, params.ku, params.alpha, params.psi
        )
    
    # =========================================================================
    # INTERFLOW (Eq 5)
    # =========================================================================
    if config.has_interflow:
        qif = physics.compute_interflow(state.S1_F, params.ki)
    else:
        qif = jnp.zeros_like(state.S1)
    
    # =========================================================================
    # BASEFLOW (Eq 6)
    # =========================================================================
    if config.baseflow == BaseflowType.LINEAR:
        qb = physics.compute_baseflow_linear(state.S2, params.v)
        qb_A = qb
        qb_B = jnp.zeros_like(qb)
    elif config.baseflow == BaseflowType.PARALLEL_LINEAR:
        qb_A, qb_B, qb = physics.compute_baseflow_parallel_linear(
            state.S2_FA, state.S2_FB, params.v_A, params.v_B
        )
    elif config.baseflow == BaseflowType.NONLINEAR:
        qb = physics.compute_baseflow_nonlinear(
            state.S2, params.S2_max, params.ks, params.n
        )
        qb_A = qb
        qb_B = jnp.zeros_like(qb)
    else:  # TOPMODEL
        qb = physics.compute_baseflow_topmodel(
            state.S2, params.S2_max, params.ks, params.m
        )
        qb_A = qb
        qb_B = jnp.zeros_like(qb)
    
    # =========================================================================
    # STATE UPDATE (Forward Euler)
    # =========================================================================
    # Upper layer
    if config.upper_arch == UpperLayerArch.SINGLE_STATE:
        dS1 = infiltration - e1 - q12 - qif
        S1_new = physics.smooth_max(state.S1 + dS1 * dt, 0.0, 0.001)
        S1_new = physics.smooth_min(S1_new, params.S1_max, 0.01)
        S1_T_new = S1_new * params.f_tens
        S1_F_new = S1_new * (1.0 - params.f_tens)
        S1_TA_new = S1_T_new * params.f_rchr
        S1_TB_new = S1_T_new * (1.0 - params.f_rchr)
        
    elif config.upper_arch == UpperLayerArch.TENSION_FREE:
        # Overflow from tension to free
        overflow_T = physics.logistic_overflow(
            state.S1_T + infiltration - e1,
            params.S1_T_max,
            params.smooth_frac * params.S1_T_max
        ) * (infiltration - e1)
        
        dS1_T = infiltration - e1 - overflow_T
        dS1_F = overflow_T - q12 - qif
        
        S1_T_new = physics.smooth_clamp(state.S1_T + dS1_T * dt, 0.0, params.S1_T_max)
        S1_F_new = physics.smooth_clamp(state.S1_F + dS1_F * dt, 0.0, params.S1_F_max)
        S1_new = S1_T_new + S1_F_new
        S1_TA_new = S1_T_new * params.f_rchr
        S1_TB_new = S1_T_new * (1.0 - params.f_rchr)
        
    else:  # TENSION2_FREE
        # Cascade through tension stores
        overflow_TA = physics.logistic_overflow(
            state.S1_TA + infiltration - e1 * params.f_rchr,
            params.S1_TA_max,
            params.smooth_frac * params.S1_TA_max
        ) * (infiltration - e1 * params.f_rchr)
        
        overflow_TB = physics.logistic_overflow(
            state.S1_TB + overflow_TA - e1 * (1.0 - params.f_rchr),
            params.S1_TB_max,
            params.smooth_frac * params.S1_TB_max
        ) * (overflow_TA - e1 * (1.0 - params.f_rchr))
        
        dS1_TA = infiltration - e1 * params.f_rchr - overflow_TA
        dS1_TB = overflow_TA - e1 * (1.0 - params.f_rchr) - overflow_TB
        dS1_F = overflow_TB - q12 - qif
        
        S1_TA_new = physics.smooth_clamp(state.S1_TA + dS1_TA * dt, 0.0, params.S1_TA_max)
        S1_TB_new = physics.smooth_clamp(state.S1_TB + dS1_TB * dt, 0.0, params.S1_TB_max)
        S1_F_new = physics.smooth_clamp(state.S1_F + dS1_F * dt, 0.0, params.S1_F_max)
        S1_T_new = S1_TA_new + S1_TB_new
        S1_new = S1_T_new + S1_F_new
    
    # Lower layer
    if config.lower_arch == LowerLayerArch.SINGLE_NOEVAP:
        dS2 = q12 - qb
        S2_new = physics.smooth_max(state.S2 + dS2 * dt, 0.0, 0.001)
        S2_T_new = S2_new
        S2_FA_new = S2_new
        S2_FB_new = jnp.zeros_like(S2_new)
        
    elif config.lower_arch == LowerLayerArch.SINGLE_EVAP:
        dS2 = q12 - e2 - qb
        S2_new = physics.smooth_clamp(state.S2 + dS2 * dt, 0.0, params.S2_max)
        S2_T_new = S2_new
        S2_FA_new = S2_new
        S2_FB_new = jnp.zeros_like(S2_new)
        
    else:  # TENSION_2RESERV
        # Split percolation between tension and free
        to_tension = params.kappa * q12
        to_free = (1.0 - params.kappa) * q12
        
        # Overflow from tension
        overflow_T = physics.logistic_overflow(
            state.S2_T + to_tension - e2,
            params.S2_T_max,
            params.smooth_frac * params.S2_T_max
        ) * (to_tension - e2)
        
        dS2_T = to_tension - e2 - overflow_T
        # Split free-water recharge in proportion to the two reservoirs'
        # capacities (S2_FA_max = f_base * S2_F_max, S2_FB_max = (1 - f_base) *
        # S2_F_max), matching FUSE/SAC-SMA tens2pll. A fixed 50/50 split would
        # ignore f_base and drive the smaller tank to saturate (and leak via
        # the storage clamp below).
        free_inflow = to_free + overflow_T
        dS2_FA = params.f_base * free_inflow - qb_A
        dS2_FB = (1.0 - params.f_base) * free_inflow - qb_B
        
        S2_T_new = physics.smooth_clamp(state.S2_T + dS2_T * dt, 0.0, params.S2_T_max)
        S2_FA_new = physics.smooth_clamp(state.S2_FA + dS2_FA * dt, 0.0, params.S2_FA_max)
        S2_FB_new = physics.smooth_clamp(state.S2_FB + dS2_FB * dt, 0.0, params.S2_FB_max)
        S2_new = S2_T_new + S2_FA_new + S2_FB_new
    
    # =========================================================================
    # BUILD OUTPUT
    # =========================================================================
    # Ensure output state has same dtype as input state
    dtype = state.S1.dtype
    new_state = State(
        S1=S1_new.astype(dtype),
        S1_T=S1_T_new.astype(dtype),
        S1_TA=S1_TA_new.astype(dtype),
        S1_TB=S1_TB_new.astype(dtype),
        S1_F=S1_F_new.astype(dtype),
        S2=S2_new.astype(dtype),
        S2_T=S2_T_new.astype(dtype),
        S2_FA=S2_FA_new.astype(dtype),
        S2_FB=S2_FB_new.astype(dtype),
        SWE=SWE_new.astype(dtype),
    )
    
    q_total = qsx + qif + qb
    
    flux = Flux(
        rain=rain,
        melt=melt,
        throughfall=throughfall,
        e1=e1,
        e2=e2,
        qsx=qsx,
        qif=qif,
        qb=qb,
        q12=q12,
        q_total=q_total,
        Ac=Ac,
    )
    
    return new_state, flux


# =============================================================================
# FULL SIMULATION
# =============================================================================

def fuse_simulate(
    forcing_series: Tuple[Array, Array, Array],
    initial_state: State,
    params: Parameters,
    config: ModelConfig,
    dt: float = 1.0,
    start_doy: int = 1,
) -> Tuple[Array, Array]:
    """Run FUSE simulation over a time series.
    
    Uses JAX's scan for efficient sequential computation with
    automatic differentiation support.
    
    Args:
        forcing_series: Tuple of (precip, pet, temp) arrays, each [n_timesteps, n_hrus]
        initial_state: Initial model state
        params: Model parameters
        config: Model configuration
        dt: Timestep in days
        start_doy: Starting day of year
        
    Returns:
        Tuple of (runoff, states) where:
            runoff: [n_timesteps, n_hrus] total runoff in mm/day
            states: Final model state
    """
    precip, pet, temp = forcing_series
    n_timesteps = precip.shape[0]
    
    def scan_fn(carry, inputs):
        state, doy = carry
        p, pe, t = inputs
        
        forcing = Forcing(precip=p, pet=pe, temp=t)
        new_state, flux = fuse_step(state, forcing, params, config, dt, doy)

        # Advance day of year with proper cycling (1-365 or 1-366 for leap years)
        # Wrap at 366 to handle both cases; seasonal calculations use /365.0
        new_doy = (doy % 366) + 1

        return (new_state, new_doy), flux.q_total

    # Run simulation
    (final_state, _), runoff = lax.scan(
        scan_fn,
        (initial_state, start_doy),
        (precip, pet, temp),
    )
    
    return runoff, final_state


# =============================================================================
# MODEL CLASS
# =============================================================================

class FUSEModel(eqx.Module):
    """FUSE model wrapper for convenient simulation and calibration.
    
    This class wraps the pure functions above into an object-oriented
    interface while maintaining full JAX compatibility.
    
    Attributes:
        config: Model configuration
        n_hrus: Number of HRUs
    """
    config: ModelConfig = eqx.field(static=True)
    n_hrus: int = eqx.field(static=True)
    
    def __init__(
        self,
        config: ModelConfig = None,
        n_hrus: int = 1,
    ):
        """Initialize FUSE model.
        
        Args:
            config: Model configuration (defaults to PRMS)
            n_hrus: Number of HRUs
        """
        self.config = config if config is not None else PRMS_CONFIG
        self.n_hrus = n_hrus
    
    def default_state(self) -> State:
        """Get default initial state."""
        return State.default(self.n_hrus)
    
    def default_params(self) -> Parameters:
        """Get default parameters."""
        return Parameters.default(self.n_hrus)
    
    def step(
        self,
        state: State,
        forcing: Forcing,
        params: Parameters,
        dt: float = 1.0,
        day_of_year: int = 1,
    ) -> Tuple[State, Flux]:
        """Run single timestep."""
        return fuse_step(state, forcing, params, self.config, dt, day_of_year)
    
    def simulate(
        self,
        forcing_series: Tuple[Array, Array, Array],
        params: Parameters,
        initial_state: State = None,
        dt: float = 1.0,
        start_doy: int = 1,
    ) -> Tuple[Array, State]:
        """Run full simulation.
        
        Args:
            forcing_series: Tuple of (precip, pet, temp) arrays
            params: Model parameters
            initial_state: Initial state (defaults to default_state())
            dt: Timestep in days
            start_doy: Starting day of year
            
        Returns:
            Tuple of (runoff array, final state)
        """
        if initial_state is None:
            initial_state = self.default_state()
        
        return fuse_simulate(
            forcing_series, initial_state, params, self.config, dt, start_doy
        )


def create_fuse_model(
    config_name: str = "prms",
    n_hrus: int = 1,
) -> FUSEModel:
    """Create a FUSE model from configuration name.
    
    Args:
        config_name: One of 'prms', 'sacramento', 'topmodel', 'vic'
        n_hrus: Number of HRUs
        
    Returns:
        Configured FUSEModel instance
    """
    from .config import get_config
    config = get_config(config_name)
    return FUSEModel(config=config, n_hrus=n_hrus)
