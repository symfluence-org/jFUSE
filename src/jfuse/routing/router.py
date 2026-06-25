"""
River Routing Algorithms

Implements differentiable river routing methods for channel flow simulation.
The primary method is Muskingum-Cunge, which provides a good balance between
physical realism and computational efficiency.

All routing functions are implemented as pure functions that are JIT-compilable
and support automatic differentiation.

References:
    Cunge, J.A. (1969). On the Subject of a Flood Propagation Computation Method
    (Muskingum Method). Journal of Hydraulic Research, 7(2), 205-230.
    
    Ponce, V.M. (1989). Engineering Hydrology: Principles and Practices.
    Prentice Hall.
"""

from typing import Tuple, NamedTuple, Optional
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array, lax
import equinox as eqx

from .network import NetworkArrays


# =============================================================================
# HYDRAULIC FUNCTIONS
# =============================================================================

def safe_pow(base: Array, exp: Array, eps: float = 1e-6) -> Array:
    """AD-safe power function."""
    safe_base = jnp.maximum(base, eps)
    return jnp.exp(exp * jnp.log(safe_base))


def compute_channel_geometry(
    Q: Array,
    width_coef: Array,
    width_exp: Array,
    depth_coef: Array,
    depth_exp: Array,
    min_Q: float = 0.01,
) -> Tuple[Array, Array, Array, Array]:
    """Compute channel geometry from discharge using power-law relationships.
    
    W = width_coef * Q^width_exp
    D = depth_coef * Q^depth_exp
    
    Args:
        Q: Discharge (m³/s)
        width_coef: Width coefficient
        width_exp: Width exponent (typically ~0.5)
        depth_coef: Depth coefficient
        depth_exp: Depth exponent (typically ~0.3)
        min_Q: Minimum discharge for computation
        
    Returns:
        Tuple of (width, depth, area, hydraulic_radius)
    """
    Q_safe = jnp.maximum(Q, min_Q)
    
    width = width_coef * safe_pow(Q_safe, width_exp)
    depth = depth_coef * safe_pow(Q_safe, depth_exp)
    
    area = width * depth
    wetted_perimeter = width + 2.0 * depth  # Rectangular approximation
    hydraulic_radius = area / wetted_perimeter
    
    return width, depth, area, hydraulic_radius


def compute_velocity_manning(
    Q: Array,
    slope: Array,
    manning_n: Array,
    width_coef: Array,
    width_exp: Array,
    depth_coef: Array,
    depth_exp: Array,
) -> Array:
    """Compute velocity using Manning's equation.
    
    V = (1/n) * R^(2/3) * S^(1/2)
    
    Args:
        Q: Discharge (m³/s)
        slope: Channel slope (m/m)
        manning_n: Manning's roughness coefficient
        width_coef, width_exp, depth_coef, depth_exp: Geometry parameters
        
    Returns:
        Flow velocity (m/s)
    """
    _, _, area, R = compute_channel_geometry(
        Q, width_coef, width_exp, depth_coef, depth_exp
    )
    
    # Manning equation
    safe_n = jnp.maximum(manning_n, 0.001)
    safe_slope = jnp.maximum(slope, 1e-6)
    
    velocity = (1.0 / safe_n) * safe_pow(R, 2.0/3.0) * jnp.sqrt(safe_slope)
    
    return velocity


def compute_celerity(
    Q: Array,
    slope: Array,
    manning_n: Array,
    width_coef: Array,
    width_exp: Array,
    depth_coef: Array,
    depth_exp: Array,
) -> Array:
    """Compute wave celerity for kinematic wave approximation.
    
    For Manning's equation with power-law geometry:
    c = (5/3) * V (simplified kinematic wave celerity)
    
    More generally: c = dQ/dA
    """
    V = compute_velocity_manning(
        Q, slope, manning_n, width_coef, width_exp, depth_coef, depth_exp
    )
    
    # Kinematic wave celerity is approximately 5/3 * V for wide channels
    # This can be refined with geometry
    celerity = (5.0 / 3.0) * V
    
    return celerity


# =============================================================================
# MUSKINGUM-CUNGE ROUTING
# =============================================================================

class MuskingumParams(NamedTuple):
    """Muskingum routing parameters for a single reach.
    
    Attributes:
        K: Storage constant (travel time, seconds)
        X: Weighting parameter [0, 0.5]
        C0, C1, C2: Muskingum coefficients
    """
    K: Array
    X: Array
    C0: Array
    C1: Array
    C2: Array


def compute_muskingum_params(
    Q: Array,
    length: Array,
    slope: Array,
    manning_n: Array,
    width_coef: Array,
    width_exp: Array,
    depth_coef: Array,
    depth_exp: Array,
    dt: float,
    x_lower: float = 0.0,
    x_upper: float = 0.5,
) -> MuskingumParams:
    """Compute Muskingum-Cunge parameters from reach properties.
    
    K = Δx / c  (travel time)
    X = 0.5 * (1 - Q / (B * c * S * Δx))  (Cunge approximation)
    
    Args:
        Q: Reference discharge (m³/s)
        length: Reach length (m)
        slope: Bed slope (m/m)
        manning_n: Manning's n
        width_coef, width_exp, depth_coef, depth_exp: Geometry parameters
        dt: Timestep (seconds)
        x_lower, x_upper: Bounds for X parameter
        
    Returns:
        MuskingumParams with computed coefficients
    """
    # Compute celerity
    celerity = compute_celerity(
        Q, slope, manning_n, width_coef, width_exp, depth_coef, depth_exp
    )
    celerity = jnp.maximum(celerity, 0.01)  # Minimum celerity
    
    # Compute width for X calculation
    width, _, _, _ = compute_channel_geometry(
        Q, width_coef, width_exp, depth_coef, depth_exp
    )
    
    # Storage constant K (travel time)
    K = length / celerity
    K = jnp.maximum(K, dt * 0.1)  # Minimum K
    
    # Muskingum X (Cunge approximation)
    safe_slope = jnp.maximum(slope, 1e-6)
    X = 0.5 * (1.0 - Q / (width * celerity * safe_slope * length + 1e-6))
    X = jnp.clip(X, x_lower, x_upper)
    
    # Muskingum coefficients
    denom = 2.0 * K * (1.0 - X) + dt
    C0 = (dt - 2.0 * K * X) / denom
    C1 = (dt + 2.0 * K * X) / denom
    C2 = (2.0 * K * (1.0 - X) - dt) / denom
    
    # Ensure non-negative coefficients for stability
    C0 = jnp.maximum(C0, 0.0)
    C1 = jnp.maximum(C1, 0.0)
    C2 = jnp.maximum(C2, 0.0)
    
    # Renormalize to sum to 1 (guard against all-clamped-to-zero -> NaN)
    total = jnp.maximum(C0 + C1 + C2, 1e-8)
    C0 = C0 / total
    C1 = C1 / total
    C2 = C2 / total
    
    return MuskingumParams(K=K, X=X, C0=C0, C1=C1, C2=C2)


def muskingum_route_reach(
    I_prev: Array,
    I_curr: Array,
    Q_prev: Array,
    params: MuskingumParams,
) -> Array:
    """Route flow through a single reach using Muskingum method.
    
    Q(t+Δt) = C0*I(t+Δt) + C1*I(t) + C2*Q(t)
    
    Args:
        I_prev: Inflow at previous timestep (m³/s)
        I_curr: Inflow at current timestep (m³/s)
        Q_prev: Outflow at previous timestep (m³/s)
        params: Muskingum parameters
        
    Returns:
        Outflow at current timestep (m³/s)
    """
    Q_curr = params.C0 * I_curr + params.C1 * I_prev + params.C2 * Q_prev
    return jnp.maximum(Q_curr, 0.0)


# =============================================================================
# LAKE / RESERVOIR ROUTING
# =============================================================================

def lake_outflow(
    S: Array,
    s_max: Array,
    q_ref: Array,
    q_min: Array,
    exp: Array,
    spill_coef: Array,
) -> Array:
    """Differentiable storage-discharge outflow for a lake or reservoir node.

    Outflow as a function of active storage ``S``::

        Q(S) = q_min + (q_ref - q_min) * clip(S / s_max, 0, 1) ** exp
                     + spill_coef * max(S - s_max, 0)

    * **Natural lakes**: ``q_min = 0`` gives a weir-like power-law rating.
    * **Regulated reservoirs**: ``q_min`` is the minimum (e.g. environmental)
      release and ``q_ref`` the reference managed discharge. ``q_ref``,
      ``q_min``, ``exp`` and ``spill_coef`` are the *operating-rule* parameters
      — all AD-active, so they calibrate by gradient against observed downstream
      flow when no reservoir-operation records are available.

    The spillway term sheds storage above capacity. Everything is smooth/clamped
    so gradients flow w.r.t. both storage and the rule parameters.

    Args:
        S: Active storage (m³).
        s_max: Storage capacity (m³).
        q_ref: Reference outflow at full storage (m³/s).
        q_min: Minimum release (m³/s).
        exp: Rating exponent (dimensionless).
        spill_coef: Spillway release rate above capacity (1/s).

    Returns:
        Outflow (m³/s).
    """
    safe_smax = jnp.maximum(s_max, 1.0)
    frac = jnp.clip(S / safe_smax, 0.0, 1.0)
    q_rating = q_min + (q_ref - q_min) * safe_pow(frac, exp)
    q_spill = spill_coef * jnp.maximum(S - s_max, 0.0)
    return jnp.maximum(q_rating + q_spill, 0.0)


# =============================================================================
# NETWORK ROUTING
# =============================================================================

class RouterState(NamedTuple):
    """State for network routing.

    Attributes:
        Q: Current discharge at each reach [n_reaches]
        Q_prev: Previous timestep discharge [n_reaches]
        I_prev: Previous timestep inflow [n_reaches]
        S_lake: Lake/reservoir storage at each reach [n_reaches]; unused
            (zeros) on reaches that are not lakes.
    """
    Q: Array
    Q_prev: Array
    I_prev: Array
    S_lake: Array = None


def route_network_step(
    state: RouterState,
    lateral_inflow: Array,
    network: NetworkArrays,
    dt: float,
) -> Tuple[RouterState, Array]:
    """Route one timestep through the river network.
    
    Uses topological ordering to process reaches from upstream to downstream,
    accumulating flows at junctions.
    
    Args:
        state: Current router state
        lateral_inflow: Lateral inflow to each reach (m³/s) [n_reaches]
        network: Network topology and parameters
        dt: Timestep (seconds)
        
    Returns:
        Tuple of (new_state, outlet_discharge)
    """
    n = network.n_reaches

    # Reference discharge for parameter computation (use previous Q)
    Q_ref = jnp.maximum(state.Q, 0.1)

    # Lake support is optional: networks built without lake attributes route as
    # pure channels and carry storage forward unchanged.
    has_lakes = getattr(network, "is_lake", None) is not None
    S_lake_prev = state.S_lake if state.S_lake is not None else jnp.zeros(n)

    # Level-parallel routing: reaches that share a topological level have no
    # mutual dependency, so route them together (vectorized) and only the
    # *levels* are sequential. This cuts the autodiff sequential depth from
    # n_reaches to n_levels (the network's longest path) — the key to making
    # reverse-mode differentiation of a national network tractable.
    if getattr(network, "reach_level", None) is not None and network.max_level is not None:
        # Muskingum coefficients depend only on the (fixed) reference discharge
        # and geometry, so compute them once for all reaches.
        mp = compute_muskingum_params(
            Q_ref, network.lengths, network.slopes, network.manning_n,
            network.width_coef, network.width_exp, network.depth_coef,
            network.depth_exp, dt,
        )
        ds_clamped = jnp.maximum(network.downstream_idx, 0)
        not_outlet = network.downstream_idx >= 0

        def level_step(carry, level):
            inflow_acc, Q_out, S_out, I_final = carry
            I_curr = inflow_acc  # total inflow per reach so far (final at this level)
            Q_chan = muskingum_route_reach(state.I_prev, I_curr, state.Q_prev, mp)
            if has_lakes:
                Q_lake_raw = lake_outflow(
                    S_lake_prev, network.lake_s_max, network.lake_q_ref,
                    network.lake_q_min, network.lake_exp, network.lake_spill_coef)
                Q_lake = jnp.clip(Q_lake_raw, 0.0, S_lake_prev / dt + I_curr)
                S_new = jnp.maximum(S_lake_prev + (I_curr - Q_lake) * dt, 0.0)
                Q_r = jnp.where(network.is_lake, Q_lake, Q_chan)
                S_r = jnp.where(network.is_lake, S_new, S_lake_prev)
            else:
                Q_r, S_r = Q_chan, S_lake_prev
            at_level = network.reach_level == level
            Q_out = jnp.where(at_level, Q_r, Q_out)
            S_out = jnp.where(at_level, S_r, S_out)
            I_final = jnp.where(at_level, I_curr, I_final)
            # Push this level's finalized outflow to each reach's downstream.
            contrib = jnp.where(at_level & not_outlet, Q_r, 0.0)
            inflow_acc = inflow_acc.at[ds_clamped].add(contrib)
            return (inflow_acc, Q_out, S_out, I_final), None

        (_, Q_final, S_lake_all, I_all), _ = lax.scan(
            level_step,
            (lateral_inflow, jnp.zeros(n), S_lake_prev, jnp.zeros(n)),
            jnp.arange(network.max_level + 1),
        )
        new_state = RouterState(Q=Q_final, Q_prev=state.Q, I_prev=I_all, S_lake=S_lake_all)
        outlet_Q = jnp.sum(jnp.where(network.is_outlet, Q_final, 0.0))
        return new_state, outlet_Q

    # --- Sequential fallback (networks without precomputed levels) ---
    def process_reach(carry, reach_idx):
        Q_accumulated = carry

        # Get upstream inflow (sum of upstream reach outflows)
        upstream_Q = jnp.sum(
            jnp.where(network.upstream_mask[reach_idx], Q_accumulated, 0.0)
        )

        # Total inflow = upstream + lateral
        I_curr = upstream_Q + lateral_inflow[reach_idx]
        I_prev = state.I_prev[reach_idx]

        # --- Channel routing (Muskingum-Cunge) ---
        params = compute_muskingum_params(
            Q_ref[reach_idx],
            network.lengths[reach_idx],
            network.slopes[reach_idx],
            network.manning_n[reach_idx],
            network.width_coef[reach_idx],
            network.width_exp[reach_idx],
            network.depth_coef[reach_idx],
            network.depth_exp[reach_idx],
            dt,
        )
        Q_chan = muskingum_route_reach(
            I_prev, I_curr, state.Q_prev[reach_idx], params
        )

        if has_lakes:
            # --- Lake / reservoir routing (storage-discharge node) ---
            S_i = S_lake_prev[reach_idx]
            Q_lake_raw = lake_outflow(
                S_i,
                network.lake_s_max[reach_idx],
                network.lake_q_ref[reach_idx],
                network.lake_q_min[reach_idx],
                network.lake_exp[reach_idx],
                network.lake_spill_coef[reach_idx],
            )
            # Cap release at the water physically available this step so storage
            # stays non-negative and mass is conserved.
            Q_lake = jnp.clip(Q_lake_raw, 0.0, S_i / dt + I_curr)
            S_lake_new_i = jnp.maximum(S_i + (I_curr - Q_lake) * dt, 0.0)

            is_lake_i = network.is_lake[reach_idx]
            Q_out = jnp.where(is_lake_i, Q_lake, Q_chan)
            S_lake_out_i = jnp.where(is_lake_i, S_lake_new_i, S_i)
        else:
            Q_out = Q_chan
            S_lake_out_i = S_lake_prev[reach_idx]

        # Update accumulated Q
        Q_accumulated = Q_accumulated.at[reach_idx].set(Q_out)

        # Store inflow + lake storage for next timestep
        return Q_accumulated, (Q_out, I_curr, S_lake_out_i)

    # Run scan over reaches in topological order
    reach_indices = jnp.arange(n)
    Q_final, (Q_all, I_all, S_lake_all) = lax.scan(
        process_reach,
        jnp.zeros(n),
        reach_indices,
    )

    # Build new state
    new_state = RouterState(
        Q=Q_final,
        Q_prev=state.Q,
        I_prev=I_all,
        S_lake=S_lake_all,
    )
    
    # Outlet discharge (sum of outlet reaches)
    outlet_Q = jnp.sum(jnp.where(network.is_outlet, Q_final, 0.0))
    
    return new_state, outlet_Q


def route_network(
    lateral_inflows: Array,
    network: NetworkArrays,
    dt: float = 3600.0,
    initial_Q: Optional[Array] = None,
    n_substeps: int = 1,
) -> Array:
    """Route a time series of lateral inflows through the network.

    Args:
        lateral_inflows: Lateral inflows [n_timesteps, n_reaches] in m³/s
        network: Network topology and parameters
        dt: Timestep in seconds (default 1 hour)
        initial_Q: Initial discharge [n_reaches] (default 0.1 m³/s)
        n_substeps: Number of Muskingum sub-steps per input timestep (static).
            Each input row is routed with ``n_substeps`` steps of ``dt /
            n_substeps``, holding the lateral inflow constant across the
            sub-steps, and the outlet is sampled at the end of each input
            timestep. Smaller sub-steps keep the Muskingum coefficients in
            their valid (non-clamped) range, improving routing stability when
            ``dt`` is large relative to reach travel times.

    Returns:
        Outlet discharge time series [n_timesteps]
    """
    n_timesteps, n_reaches = lateral_inflows.shape

    # Initial state
    if initial_Q is None:
        initial_Q = jnp.full(n_reaches, 0.1)

    initial_state = RouterState(
        Q=initial_Q,
        Q_prev=initial_Q,
        I_prev=jnp.zeros(n_reaches),
        S_lake=jnp.zeros(n_reaches),
    )

    # Sub-step by holding each row's inflow constant over n_substeps steps of
    # dt/n_substeps, then sampling the outlet at the end of each input timestep.
    if n_substeps > 1:
        step_dt = dt / n_substeps
        inflows = jnp.repeat(lateral_inflows, n_substeps, axis=0)
    else:
        step_dt = dt
        inflows = lateral_inflows

    def scan_fn(state, lateral):
        new_state, outlet_Q = route_network_step(state, lateral, network, step_dt)
        return new_state, outlet_Q

    # Checkpoint the per-timestep routing body (which itself scans over reaches)
    # so the backward pass recomputes it from the carry rather than storing every
    # reach's intermediates across all timesteps.
    _, outlet_series = lax.scan(jax.checkpoint(scan_fn), initial_state, inflows)

    if n_substeps > 1:
        # Keep the last sub-step of each input timestep.
        outlet_series = outlet_series[n_substeps - 1::n_substeps]

    return outlet_series


def route_network_full(
    lateral_inflows: Array,
    network: NetworkArrays,
    dt: float = 3600.0,
    initial_Q: Optional[Array] = None,
    n_substeps: int = 1,
) -> Tuple[Array, Array]:
    """Like :func:`route_network` but also returns discharge at *every* reach.

    Required for multi-gauge calibration, where the objective compares simulated
    flow at each gauge's reach against observations.

    Returns:
        Tuple ``(outlet_series, Q_all)`` where ``outlet_series`` is
        ``[n_timesteps]`` (summed outlet discharge) and ``Q_all`` is
        ``[n_timesteps, n_reaches]`` discharge at each reach, both sampled at the
        end of each input timestep.
    """
    n_timesteps, n_reaches = lateral_inflows.shape

    if initial_Q is None:
        initial_Q = jnp.full(n_reaches, 0.1)

    initial_state = RouterState(
        Q=initial_Q,
        Q_prev=initial_Q,
        I_prev=jnp.zeros(n_reaches),
        S_lake=jnp.zeros(n_reaches),
    )

    if n_substeps > 1:
        step_dt = dt / n_substeps
        inflows = jnp.repeat(lateral_inflows, n_substeps, axis=0)
    else:
        step_dt = dt
        inflows = lateral_inflows

    def scan_fn(state, lateral):
        new_state, outlet_Q = route_network_step(state, lateral, network, step_dt)
        return new_state, (outlet_Q, new_state.Q)

    # Checkpoint the per-timestep body (see route_network) to bound backward memory.
    _, (outlet_series, Q_all) = lax.scan(jax.checkpoint(scan_fn), initial_state, inflows)

    if n_substeps > 1:
        outlet_series = outlet_series[n_substeps - 1::n_substeps]
        Q_all = Q_all[n_substeps - 1::n_substeps]

    return outlet_series, Q_all


# =============================================================================
# ROUTER CLASS
# =============================================================================

class MuskingumCungeRouter(eqx.Module):
    """Muskingum-Cunge river router.
    
    Wrapper class providing a convenient interface for network routing
    with configurable parameters.
    
    Attributes:
        network: Network topology arrays
        dt: Routing timestep (seconds)
    """
    network: NetworkArrays
    dt: float = eqx.field(static=True)
    
    def __init__(
        self,
        network: NetworkArrays,
        dt: float = 3600.0,
    ):
        """Initialize router.
        
        Args:
            network: Network topology arrays
            dt: Routing timestep in seconds (default 1 hour)
        """
        self.network = network
        self.dt = dt
    
    def route(
        self,
        lateral_inflows: Array,
        initial_Q: Optional[Array] = None,
    ) -> Array:
        """Route lateral inflows through the network.
        
        Args:
            lateral_inflows: [n_timesteps, n_reaches] in m³/s
            initial_Q: Initial discharge [n_reaches]
            
        Returns:
            Outlet discharge [n_timesteps]
        """
        return route_network(lateral_inflows, self.network, self.dt, initial_Q)
    
    def route_with_states(
        self,
        lateral_inflows: Array,
        initial_Q: Optional[Array] = None,
    ) -> Tuple[Array, Array]:
        """Route with full state output.
        
        Args:
            lateral_inflows: [n_timesteps, n_reaches] in m³/s
            initial_Q: Initial discharge [n_reaches]
            
        Returns:
            Tuple of (outlet_discharge, all_reach_discharge)
            where all_reach_discharge is [n_timesteps, n_reaches]
        """
        n_timesteps, n_reaches = lateral_inflows.shape
        
        if initial_Q is None:
            initial_Q = jnp.full(n_reaches, 0.1)
        
        initial_state = RouterState(
            Q=initial_Q,
            Q_prev=initial_Q,
            I_prev=jnp.zeros(n_reaches),
            S_lake=jnp.zeros(n_reaches),
        )

        def scan_fn(state, lateral):
            new_state, outlet_Q = route_network_step(
                state, lateral, self.network, self.dt
            )
            return new_state, (outlet_Q, new_state.Q)
        
        _, (outlet_series, Q_all) = lax.scan(
            scan_fn, initial_state, lateral_inflows
        )
        
        return outlet_series, Q_all
    
    @property
    def n_reaches(self) -> int:
        """Number of reaches in the network."""
        return self.network.n_reaches
    
    @property
    def manning_n(self) -> Array:
        """Manning's n values for all reaches."""
        return self.network.manning_n
