"""
FUSE Physics Computations

Differentiable implementations of all flux equations from Clark et al. (2008).
All functions use smooth approximations for discontinuities to ensure
gradient flow through automatic differentiation.

Key design principles:
- All functions are pure and JIT-compatible
- Uses smooth approximations (sigmoid, softmax) for thresholds
- No control flow that would break differentiation

References:
    Clark, M. P., et al. (2008). Framework for Understanding Structural Errors
    (FUSE). Water Resources Research, 44, W00B02.

    Kavetski, D., & Kuczera, G. (2007). Model smoothing strategies to remove
    microscale discontinuities and spurious secondary optima in objective
    functions in hydrological calibration. Water Resources Research, 43, W03411.
"""

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from typing import Tuple, Optional

# =============================================================================
# SMOOTH UTILITY FUNCTIONS
# =============================================================================


def safe_pow(base: Array, exponent: Array, eps: float = 1e-5) -> Array:
    """AD-safe power function using exp(y * log(x)).

    This formulation ensures gradients flow through both base and exponent,
    unlike direct pow() which can have issues near zero.

    Uses smooth clamping to ensure base > eps.
    """
    k = 1e-6  # Smoothing width
    diff = base - eps
    smooth_abs = jnp.sqrt(diff * diff + 4 * k * k)
    safe_base = 0.5 * (base + eps + smooth_abs)
    return jnp.exp(exponent * jnp.log(safe_base))


def smooth_sigmoid(x: Array, k: float = 1.0) -> Array:
    """Smooth sigmoid function for step approximation.

    Uses tanh identity: sigmoid(x) = 0.5 * (1 + tanh(x/2))
    This is numerically stable and completely branch-free.
    """
    z = x / k
    return 0.5 * (1.0 + jnp.tanh(z * 0.5))


def smooth_max(a: ArrayLike, b: ArrayLike, k: float = 0.01) -> Array:
    """Smooth differentiable approximation to max(a, b).

    Uses: softmax(a,b) = 0.5*(a+b) + 0.5*sqrt((a-b)^2 + 4*k^2)
    """
    a, b = jnp.asarray(a), jnp.asarray(b)
    diff = a - b
    smooth_abs = jnp.sqrt(diff * diff + 4 * k * k)
    return 0.5 * (a + b + smooth_abs)


def smooth_min(a: ArrayLike, b: ArrayLike, k: float = 0.01) -> Array:
    """Smooth differentiable approximation to min(a, b).

    Uses: softmin(a,b) = 0.5*(a+b) - 0.5*sqrt((a-b)^2 + 4*k^2)
    """
    a, b = jnp.asarray(a), jnp.asarray(b)
    diff = a - b
    smooth_abs = jnp.sqrt(diff * diff + 4 * k * k)
    return 0.5 * (a + b - smooth_abs)


def smooth_clamp(x: ArrayLike, min_val: ArrayLike, max_val: ArrayLike, k: float = 0.01) -> Array:
    """Smooth clamp to [min_val, max_val]."""
    return smooth_min(smooth_max(x, min_val, k), max_val, k)


def logistic_overflow(S: Array, S_max: Array, w: Array) -> Array:
    """Logistic smoothing function for bucket overflow.

    Implements Clark et al. (2008) equation (12h):
    F(S, Smax, w) = 1 / (1 + exp(-(S - Smax - w*e) / w))

    This provides a differentiable approximation to the step function
    that triggers overflow when storage exceeds capacity.

    Args:
        S: Current storage
        S_max: Maximum storage capacity
        w: Smoothing width (typically smooth_frac * S_max)

    Returns:
        Overflow fraction [0, 1]
    """
    e_mult = 5.0  # Ensures S < S_max always
    eps_w = 1e-6
    k = 1e-8
    diff = w - eps_w
    smooth_abs = jnp.sqrt(diff * diff + 4 * k * k)
    safe_w = 0.5 * (w + eps_w + smooth_abs)

    x = (S - S_max - safe_w * e_mult) / safe_w
    return 0.5 * (1.0 + jnp.tanh(x * 0.5))


# =============================================================================
# SNOW MODULE
# =============================================================================


def compute_snow(
    precip: Array,
    temp: Array,
    SWE: Array,
    T_rain: Array,
    T_melt: Array,
    melt_rate: Array,
    day_of_year: int = 1,
    MFMAX: Optional[Array] = None,
    MFMIN: Optional[Array] = None,
) -> Tuple[Array, Array, Array]:
    """Snow accumulation and melt using temperature-index method.

    Simple temperature-index (degree-day) snow model.

    Args:
        precip: Total precipitation (mm/day)
        temp: Air temperature (°C)
        SWE: Current snow water equivalent (mm)
        T_rain: Rain/snow threshold temperature (°C)
        T_melt: Snowmelt threshold temperature (°C)
        melt_rate: Degree-day melt factor (mm/°C/day)
        day_of_year: Day of year for seasonal melt factor (1-365)
        MFMAX: Maximum seasonal melt factor (optional)
        MFMIN: Minimum seasonal melt factor (optional)

    Returns:
        Tuple of (rain, melt, SWE_new)
    """
    # Seasonal melt factor if provided
    if MFMAX is not None and MFMIN is not None:
        phase = 2 * jnp.pi * (day_of_year - 81) / 365.0
        MF_avg = (MFMAX + MFMIN) / 2.0
        MF_amp = (MFMAX - MFMIN) / 2.0
        effective_melt_rate = MF_avg + MF_amp * jnp.sin(phase)
    else:
        effective_melt_rate = melt_rate

    # Smooth rain/snow partitioning using sigmoid
    # snow_frac = 1 when temp << T_rain, 0 when temp >> T_rain
    transition_width = 2.0  # °C - width of rain/snow transition
    snow_frac = smooth_sigmoid(T_rain - temp, transition_width)

    snow = precip * snow_frac
    rain = precip * (1.0 - snow_frac)

    # Potential melt using degree-day method
    # melt_frac = 0 when temp < T_melt, increases linearly above
    melt_potential = smooth_max(temp - T_melt, 0.0, 0.1) * effective_melt_rate

    # Actual melt limited by available SWE
    melt = smooth_min(melt_potential, SWE + snow, 0.01)

    # Update SWE
    SWE_new = smooth_max(SWE + snow - melt, 0.0, 0.001)

    return rain, melt, SWE_new


# =============================================================================
# GLACIER MODULE
# =============================================================================


def compute_glacier(
    temp: Array,
    SWE: Array,
    rain: Array,
    snowmelt: Array,
    S_glac: Array,
    ICE: Array,
    DDF_ice: Array,
    T_ice: Array,
    K_glac: Array,
    dt: float = 1.0,
    swe_bare_thresh: float = 5.0,
    exposure_width: float = 1.0,
) -> Tuple[Array, Array, Array, Array]:
    """Temperature-index glacier melt + fast glacier reservoir (per unit
    glacierized area).

    Physical picture (HBV-glacier / GSM-SOCONT style, fixed geometry):

    * Seasonal snow on the glacier melts first — that is already handled by the
      snow module via ``SWE``. Once the snowpack is exhausted the underlying ice
      is *exposed* and melts at the ice degree-day factor ``DDF_ice``, which is
      larger than the snow factor (lower ice albedo).
    * All liquid water generated on the glacier — rain + snowmelt + ice melt —
      drains through a single fast linear reservoir ``S_glac`` rather than the
      soil column, reproducing the rapid, well-buffered response of glacier-fed
      rivers (the buffering that keeps such rivers flowing through dry summers).

    The ice store ``ICE`` is tracked so the water balance closes and optional
    multi-decade thinning can be modelled; initialised large it is effectively
    inexhaustible over a calibration horizon (fixed-geometry assumption).

    All water fluxes are depths per unit area (mm/day); ``S_glac``/``ICE`` are
    storages (mm w.e.). The caller area-weights ``q_glacier`` by the HRU's
    glacier fraction. Everything is smooth/AD-safe so ``DDF_ice``, ``T_ice`` and
    ``K_glac`` calibrate by gradient like any other parameter.

    Args:
        temp: Air temperature (°C).
        SWE: Snow water equivalent after the snow update (mm) — controls ice
            exposure.
        rain: Rainfall reaching the surface this step (mm/day).
        snowmelt: Snowmelt this step (mm/day).
        S_glac: Current glacier-reservoir storage (mm).
        ICE: Current ice store (mm w.e.).
        DDF_ice: Ice degree-day melt factor (mm/°C/day).
        T_ice: Ice-melt threshold temperature (°C).
        K_glac: Glacier-reservoir release coefficient (1/day).
        dt: Timestep (days).
        swe_bare_thresh: SWE (mm) below which ice is treated as exposed.
        exposure_width: Smoothing width (mm) of the snow-cover/bare-ice
            transition.

    Returns:
        Tuple ``(q_glacier, S_glac_new, ICE_new, ice_melt)`` — glacier-reservoir
        release (mm/day), updated reservoir storage, updated ice store and the
        ice melt this step (mm/day).
    """
    # Fraction of the glacier that is snow-free (ice exposed): ~1 when SWE is
    # near zero, ~0 once a seasonal snowpack has built up.
    exposure = smooth_sigmoid(swe_bare_thresh - SWE, exposure_width)

    # Degree-day ice melt, only on exposed ice and limited by available ice.
    melt_potential = DDF_ice * smooth_max(temp - T_ice, 0.0, 0.1) * exposure
    ice_melt = smooth_min(melt_potential, smooth_max(ICE, 0.0, 0.01), 0.01)
    ICE_new = smooth_max(ICE - ice_melt * dt, 0.0, 0.01)

    # All liquid water on the glacier feeds the fast reservoir; release is a
    # linear function of storage.
    glacier_input = rain + snowmelt + ice_melt
    q_glacier = K_glac * smooth_max(S_glac, 0.0, 0.01)
    S_glac_new = smooth_max(S_glac + (glacier_input - q_glacier) * dt, 0.0, 0.01)

    return q_glacier, S_glac_new, ICE_new, ice_melt


# =============================================================================
# EVAPORATION
# =============================================================================


def compute_evaporation_sequential(
    pet: Array,
    S1: Array,
    S2: Array,
    S1_max: Array,
    S2_max: Array,
) -> Tuple[Array, Array]:
    """Differentiable sequential evaporation (FUSE-faithful).

    Upper layer satisfies PET demand first, limited by available storage.
    Remaining PET is extracted from the lower layer with moisture stress.

    This is a smooth, AD-safe implementation of Clark et al. (2008)
    equations (3a) and (3b).

    Args:
        pet: Potential evapotranspiration (mm/day)
        S1: Upper layer storage (mm)
        S2: Lower layer storage (mm)
        S1_max: Maximum upper layer storage (mm)
        S2_max: Maximum lower layer storage (mm)

    Returns:
        Tuple of (e1, e2) evaporation from upper and lower layers
    """

    # ------------------------------------------------------------------
    # 1) Upper layer: full priority, limited only by storage
    # ------------------------------------------------------------------
    e1 = smooth_min(pet, S1, 0.01)

    # ------------------------------------------------------------------
    # 2) Remaining PET after upper layer extraction
    # ------------------------------------------------------------------
    pet_remain = smooth_max(pet - e1, 0.0, 0.001)

    # ------------------------------------------------------------------
    # 3) Lower layer: moisture-stressed extraction
    # ------------------------------------------------------------------
    S2_frac = S2 / smooth_max(S2_max, 1.0, 0.01)
    e2_demand = pet_remain * S2_frac
    e2 = smooth_min(e2_demand, S2, 0.01)

    return e1, e2


def compute_evaporation_root_weighted(
    pet: Array,
    S1: Array,
    S2: Array,
    S1_max: Array,
    S2_max: Array,
    r1: Array,
) -> Tuple[Array, Array]:
    """Root-weighted evaporation between layers.

    Implements equations (3c) and (3d) from Clark et al. (2008).

    e1 = r1 * PET * (S1/S1_max)
    e2 = (1-r1) * PET * (S2/S2_max)

    Args:
        pet: Potential evapotranspiration (mm/day)
        S1: Upper layer storage (mm)
        S2: Lower layer storage (mm)
        S1_max: Maximum upper layer storage (mm)
        S2_max: Maximum lower layer storage (mm)
        r1: Root fraction in upper layer [-]

    Returns:
        Tuple of (e1, e2) - actual evaporation from each layer
    """
    # Partitioned demand
    pet_upper = r1 * pet
    pet_lower = (1.0 - r1) * pet

    # Actual evaporation (soil moisture limited)
    e1_frac = S1 / smooth_max(S1_max, 1.0, 0.01)
    e1 = smooth_min(pet_upper * e1_frac, S1, 0.01)

    e2_frac = S2 / smooth_max(S2_max, 1.0, 0.01)
    e2 = smooth_min(pet_lower * e2_frac, S2, 0.01)

    return e1, e2


# =============================================================================
# PERCOLATION
# =============================================================================


def compute_percolation_total_storage(
    S1: Array,
    S1_max: Array,
    ku: Array,
    c: Array,
) -> Array:
    """Percolation based on total upper zone storage (VIC style).

    Implements equation (4a): q12 = ku * (S1/S1_max)^c

    Args:
        S1: Upper layer storage (mm)
        S1_max: Maximum upper layer storage (mm)
        ku: Drainage rate (1/day)
        c: Shape parameter

    Returns:
        Percolation flux (mm/day)
    """
    S1_frac = S1 / smooth_max(S1_max, 1.0, 0.01)
    return ku * safe_pow(S1_frac, c)


def compute_percolation_free_storage(
    S1_F: Array,
    S1_F_max: Array,
    ku: Array,
) -> Array:
    """Percolation from free storage above field capacity (PRMS).

    Implements equation (4b): q12 = ku * S1_F

    Args:
        S1_F: Free storage (mm)
        S1_F_max: Maximum free storage (mm)
        ku: Drainage rate (1/day)

    Returns:
        Percolation flux (mm/day)
    """
    return ku * S1_F


def compute_percolation_lower_demand(
    S1_F: Array,
    S2: Array,
    S2_max: Array,
    ku: Array,
    alpha: Array,
    psi: Array,
) -> Array:
    """Percolation driven by lower zone demand (Sacramento).

    Implements equation (4c): q12 = ku * S1_F * (1 - (S2/S2_max)^alpha) * psi

    Args:
        S1_F: Free storage (mm)
        S2: Lower layer storage (mm)
        S2_max: Maximum lower layer storage (mm)
        ku: Drainage rate (1/day)
        alpha: Shape parameter
        psi: Demand coefficient

    Returns:
        Percolation flux (mm/day)
    """
    S2_frac = S2 / smooth_max(S2_max, 1.0, 0.01)
    demand = 1.0 - safe_pow(S2_frac, alpha)
    return ku * S1_F * demand * psi


# =============================================================================
# INTERFLOW
# =============================================================================


def compute_interflow(
    S1_F: Array,
    ki: Array,
) -> Array:
    """Linear interflow from free storage.

    Implements equation (5b): qif = ki * S1_F

    Args:
        S1_F: Free storage (mm)
        ki: Interflow rate (1/day)

    Returns:
        Interflow flux (mm/day)
    """
    return ki * smooth_max(S1_F, 0.0, 0.01)


# =============================================================================
# BASEFLOW
# =============================================================================


def compute_baseflow_linear(S2: Array, v: Array) -> Array:
    """Single linear reservoir baseflow (PRMS).

    Implements equation (6a): qb = v * S2
    """
    return v * smooth_max(S2, 0.0, 0.01)


def compute_baseflow_parallel_linear(
    S2_FA: Array,
    S2_FB: Array,
    v_A: Array,
    v_B: Array,
) -> Tuple[Array, Array, Array]:
    """Two parallel linear reservoirs (Sacramento).

    Implements equation (6b):
        qb_A = v_A * S2_FA
        qb_B = v_B * S2_FB
        qb = qb_A + qb_B

    Returns:
        Tuple of (qb_A, qb_B, qb_total)
    """
    qb_A = v_A * smooth_max(S2_FA, 0.0, 0.01)
    qb_B = v_B * smooth_max(S2_FB, 0.0, 0.01)
    return qb_A, qb_B, qb_A + qb_B


def compute_baseflow_nonlinear(
    S2: Array,
    S2_max: Array,
    ks: Array,
    n: Array,
) -> Array:
    """Nonlinear power-law baseflow (ARNO/VIC).

    Implements equation (6c): qb = ks * S2_max * (S2/S2_max)^(1+n)
    """
    S2_frac = S2 / smooth_max(S2_max, 1.0, 0.01)
    return ks * S2_max * safe_pow(S2_frac, 1.0 + n)


def compute_baseflow_topmodel(
    S2: Array,
    S2_max: Array,
    ks: Array,
    m: Array,
) -> Array:
    """TOPMODEL power-law transmissivity.

    Implements equation (6d): qb = ks * exp(-S2/m)
    """
    safe_m = smooth_max(m, 1.0, 0.1)
    return ks * jnp.exp(-S2 / safe_m)


# =============================================================================
# SATURATED AREA / SURFACE RUNOFF
# =============================================================================


def compute_satarea_linear(
    S1_T: Array,
    S1_T_max: Array,
    Ac_max: Array,
) -> Array:
    """Linear saturated area function (PRMS).

    Implements equation (9a): Ac = Ac_max * (S1_T / S1_T_max)
    """
    frac = S1_T / smooth_max(S1_T_max, 1.0, 0.01)
    return Ac_max * smooth_clamp(frac, 0.0, 1.0)


def compute_satarea_pareto(
    S1: Array,
    S1_max: Array,
    b: Array,
    Ac_max: Array,
) -> Array:
    """Pareto distribution / VIC 'b' curve.

    Implements equation (9b): Ac = Ac_max * (1 - (1 - S1/S1_max)^b)
    """
    S1_frac = smooth_clamp(S1 / smooth_max(S1_max, 1.0, 0.01), 0.0, 0.999)
    return Ac_max * (1.0 - safe_pow(1.0 - S1_frac, b))


def compute_satarea_topmodel(
    S2: Array,
    S2_max: Array,
    chi: Array,
    Ac_max: Array,
) -> Array:
    """TOPMODEL saturated area from topographic index.

    Simplified implementation using sigmoid approximation.

    Implements equation (9c).
    """
    S2_frac = S2 / smooth_max(S2_max, 1.0, 0.01)
    x = chi * (S2_frac - 0.5)
    Ac = 0.5 * (1.0 + jnp.tanh(x * 0.5))
    return smooth_clamp(Ac * Ac_max, 0.0, Ac_max)


def compute_surface_runoff(
    throughfall: Array,
    Ac: Array,
) -> Array:
    """Surface runoff from saturated area.

    Implements equation (11): qsx = Ac * throughfall
    """
    return smooth_max(Ac * throughfall, 0.0, 0.001)
