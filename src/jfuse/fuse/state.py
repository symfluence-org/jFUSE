"""
FUSE State Variables, Fluxes, and Parameters

Defines the data structures for model states, fluxes, and parameters using
JAX-compatible pytree structures. Uses equinox for clean dataclass-like
syntax with full JAX compatibility.

References:
    Clark, M. P., et al. (2008). Framework for Understanding Structural Errors
    (FUSE). Water Resources Research, 44, W00B02.
"""

from typing import Dict, Tuple
import jax.numpy as jnp
from jax import Array
import equinox as eqx

# =============================================================================
# PARAMETER BOUNDS AND NAMES
# =============================================================================

# Parameter names in order (matches C++ implementation)
PARAM_NAMES = (
    "S1_max",  # Maximum upper layer storage (mm)
    "S2_max",  # Maximum lower layer storage (mm)
    "f_tens",  # Fraction of storage as tension
    "f_rchr",  # Fraction tension in primary zone
    "f_base",  # Fraction free storage in primary reservoir
    "r1",  # Root fraction in upper layer
    "ku",  # Upper layer drainage rate (1/day)
    "c",  # VIC/ARNO infiltration shape parameter
    "alpha",  # Sacramento percolation shape
    "psi",  # Sacramento lower zone demand coefficient
    "kappa",  # Fraction of percolation to tension storage
    "ki",  # Interflow rate (1/day)
    "ks",  # Baseflow rate (1/day)
    "n",  # TOPMODEL decay parameter
    "v",  # Linear baseflow rate (1/day)
    "v_A",  # Primary reservoir rate (1/day)
    "v_B",  # Secondary reservoir rate (1/day)
    "Ac_max",  # Maximum saturated area fraction
    "b",  # VIC 'b' parameter
    "lam",  # Pareto shape for storage capacity (renamed from lambda)
    "chi",  # TOPMODEL shape parameter
    "mu_t",  # Time scale for percolation (days)
    "T_rain",  # Rain/snow threshold temperature (°C)
    "T_melt",  # Snowmelt threshold temperature (°C)
    "melt_rate",  # Degree-day melt factor (mm/°C/day)
    "lapse_rate",  # Temperature lapse rate (°C/100m)
    "opg",  # Orographic precipitation gradient (%/100m)
    "MFMAX",  # Maximum seasonal melt factor (mm/°C/day)
    "MFMIN",  # Minimum seasonal melt factor (mm/°C/day)
    "smooth_frac",  # Smoothing fraction for overflow
    # --- Glacier module (appended; indices preserved for legacy arrays) ---
    "DDF_ice",  # Ice degree-day melt factor (mm/°C/day)
    "T_ice",  # Ice-melt threshold temperature (°C)
    "K_glac",  # Glacier-reservoir release coefficient (1/day)
)

# Parameter bounds: (lower, upper)
PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "S1_max": (50.0, 5000.0),
    "S2_max": (100.0, 20000.0),
    "f_tens": (0.05, 0.95),
    "f_rchr": (0.05, 0.95),
    "f_base": (0.05, 0.95),
    "r1": (0.05, 0.95),
    "ku": (0.001, 1.0),
    "c": (0.1, 10.0),
    "alpha": (0.1, 10.0),
    "psi": (0.001, 5.0),
    "kappa": (0.05, 0.95),
    "ki": (0.001, 1.0),
    "ks": (0.0001, 0.1),
    "n": (0.1, 10.0),
    "v": (0.001, 0.5),
    "v_A": (0.001, 0.5),
    "v_B": (0.0001, 0.1),
    "Ac_max": (0.01, 1.0),
    "b": (0.01, 3.0),
    "lam": (0.01, 5.0),
    "chi": (1.0, 20.0),
    "mu_t": (0.01, 100.0),
    "T_rain": (-3.0, 5.0),
    "T_melt": (-3.0, 5.0),
    "melt_rate": (0.5, 10.0),
    "lapse_rate": (0.3, 1.0),
    "opg": (-10.0, 50.0),
    "MFMAX": (1.0, 8.0),
    "MFMIN": (0.1, 2.0),
    "smooth_frac": (0.001, 0.1),
    # Glacier: ice DDF usually exceeds the snow factor (lower ice albedo).
    # Literature ice degree-day factors are ~5-8 mm/°C/day; the upper bound is
    # capped at 8 so calibration can't mask warm-biased (under-lapsed) glacier
    # forcing by inflating melt. K_glac fast (days).
    "DDF_ice": (3.0, 8.0),
    "T_ice": (-2.0, 2.0),
    "K_glac": (0.01, 1.0),
}

NUM_PARAMETERS = len(PARAM_NAMES)

# Default initial ice store (mm w.e.). ~1e7 mm is effectively inexhaustible over
# a calibration horizon (a 3000 mm/yr ablation glacier melts ~24,000 mm in 8
# years), so the glacier behaves fixed-geometry unless ICE is initialised lower.
DEFAULT_ICE = 1.0e7


def get_param_bounds_arrays() -> Tuple[Array, Array]:
    """Get parameter bounds as JAX arrays.

    Returns:
        Tuple of (lower_bounds, upper_bounds) arrays of shape (NUM_PARAMETERS,)
    """
    lower = jnp.array([PARAM_BOUNDS[name][0] for name in PARAM_NAMES])
    upper = jnp.array([PARAM_BOUNDS[name][1] for name in PARAM_NAMES])
    return lower, upper


# =============================================================================
# STATE VARIABLES
# =============================================================================


class State(eqx.Module):
    """Model state variables for a single HRU.

    State variable naming follows Clark et al. (2008) Table 1.
    All storages are in mm.

    Attributes:
        S1: Total upper layer storage
        S1_T: Upper layer tension storage (below field capacity)
        S1_TA: Primary tension storage (upper)
        S1_TB: Secondary tension storage (upper)
        S1_F: Free storage (above field capacity)
        S2: Total lower layer storage
        S2_T: Lower layer tension storage
        S2_FA: Primary baseflow reservoir
        S2_FB: Secondary baseflow reservoir
        SWE: Snow water equivalent
        ICE: Glacier ice store (mm w.e.); large => fixed-geometry glacier
        S_glac: Fast glacier-reservoir storage (mm)
    """

    S1: Array
    S1_T: Array
    S1_TA: Array
    S1_TB: Array
    S1_F: Array
    S2: Array
    S2_T: Array
    S2_FA: Array
    S2_FB: Array
    SWE: Array
    # Glacier states default so pre-glacier ``State(...)`` construction (and
    # downstream callers) keep working; ``fuse_simulate`` broadcasts state
    # leaves to the per-HRU shape, so scalar defaults are fine.
    ICE: Array = eqx.field(default=DEFAULT_ICE)
    S_glac: Array = eqx.field(default=0.0)
    # Snow water equivalent on the glacier fraction, evolved at the (colder)
    # glacier-surface temperature — separate from the GRU-mean land column SWE.
    SWE_glac: Array = eqx.field(default=0.0)

    @classmethod
    def default(cls, n_hrus: int = 1) -> "State":
        """Create default initial state.

        Args:
            n_hrus: Number of HRUs (for batched operations)

        Returns:
            State with default values
        """
        shape = (n_hrus,) if n_hrus > 1 else ()
        # Use float32 for compatibility with neuralgcm (which requires JAX_ENABLE_X64=False)
        dtype = jnp.float32
        return cls(
            S1=jnp.full(shape, 100.0, dtype=dtype),
            S1_T=jnp.full(shape, 40.0, dtype=dtype),
            S1_TA=jnp.full(shape, 20.0, dtype=dtype),
            S1_TB=jnp.full(shape, 20.0, dtype=dtype),
            S1_F=jnp.full(shape, 60.0, dtype=dtype),
            S2=jnp.full(shape, 400.0, dtype=dtype),
            S2_T=jnp.full(shape, 160.0, dtype=dtype),
            S2_FA=jnp.full(shape, 120.0, dtype=dtype),
            S2_FB=jnp.full(shape, 120.0, dtype=dtype),
            SWE=jnp.full(shape, 0.0, dtype=dtype),
            ICE=jnp.full(shape, DEFAULT_ICE, dtype=dtype),
            S_glac=jnp.full(shape, 0.0, dtype=dtype),
            SWE_glac=jnp.full(shape, 0.0, dtype=dtype),
        )

    def to_array(self) -> Array:
        """Flatten state to array for numerical integration."""
        return jnp.stack(
            [
                self.S1,
                self.S1_T,
                self.S1_TA,
                self.S1_TB,
                self.S1_F,
                self.S2,
                self.S2_T,
                self.S2_FA,
                self.S2_FB,
                self.SWE,
                self.ICE,
                self.S_glac,
                self.SWE_glac,
            ],
            axis=-1,
        )

    @classmethod
    def from_array(cls, arr: Array) -> "State":
        """Reconstruct state from flattened array.

        Tolerates legacy 10-column arrays (pre-glacier) by defaulting the ice
        store and glacier reservoir.
        """
        has_glacier = arr.shape[-1] >= 12
        ICE = arr[..., 10] if has_glacier else jnp.full_like(arr[..., 9], DEFAULT_ICE)
        S_glac = arr[..., 11] if has_glacier else jnp.zeros_like(arr[..., 9])
        # Glacier-snow column (13th) is optional; legacy arrays default it to 0.
        SWE_glac = arr[..., 12] if arr.shape[-1] >= 13 else jnp.zeros_like(arr[..., 9])
        return cls(
            S1=arr[..., 0],
            S1_T=arr[..., 1],
            S1_TA=arr[..., 2],
            S1_TB=arr[..., 3],
            S1_F=arr[..., 4],
            S2=arr[..., 5],
            S2_T=arr[..., 6],
            S2_FA=arr[..., 7],
            S2_FB=arr[..., 8],
            SWE=arr[..., 9],
            ICE=ICE,
            S_glac=S_glac,
            SWE_glac=SWE_glac,
        )


# =============================================================================
# FLUX VARIABLES
# =============================================================================


class Flux(eqx.Module):
    """Model fluxes for a single timestep.

    Flux naming follows Clark et al. (2008) Table 2.
    All fluxes in mm/day.

    Attributes:
        rain: Rainfall (after snow partition)
        melt: Snowmelt
        throughfall: Rain + melt reaching soil
        e1: Evaporation from upper layer
        e2: Evaporation from lower layer
        qsx: Saturation excess surface runoff
        qif: Interflow
        qb: Total baseflow
        q12: Percolation from upper to lower
        q_total: Total runoff (qsx + qif + qb)
        Ac: Saturated contributing area fraction
    """

    rain: Array
    melt: Array
    throughfall: Array
    e1: Array
    e2: Array
    qsx: Array
    qif: Array
    qb: Array
    q12: Array
    q_total: Array
    Ac: Array

    @classmethod
    def zeros(cls, shape: Tuple = ()) -> "Flux":
        """Create zero-initialized fluxes."""
        return cls(
            rain=jnp.zeros(shape),
            melt=jnp.zeros(shape),
            throughfall=jnp.zeros(shape),
            e1=jnp.zeros(shape),
            e2=jnp.zeros(shape),
            qsx=jnp.zeros(shape),
            qif=jnp.zeros(shape),
            qb=jnp.zeros(shape),
            q12=jnp.zeros(shape),
            q_total=jnp.zeros(shape),
            Ac=jnp.zeros(shape),
        )


# =============================================================================
# FORCING DATA
# =============================================================================


class Forcing(eqx.Module):
    """Meteorological forcing for a single timestep.

    Attributes:
        precip: Total precipitation (mm/day)
        pet: Potential evapotranspiration (mm/day)
        temp: Air temperature (°C) - for snow module
    """

    precip: Array
    pet: Array
    temp: Array

    @classmethod
    def from_arrays(cls, precip: Array, pet: Array, temp: Array) -> "Forcing":
        """Create forcing from separate arrays."""
        return cls(precip=precip, pet=pet, temp=temp)


# =============================================================================
# PARAMETERS
# =============================================================================


class Parameters(eqx.Module):
    """Model parameters for FUSE.

    Parameter naming and bounds follow Clark et al. (2008) Table 3.

    This class stores both the adjustable parameters and derived quantities
    that are computed from them.
    """

    # Storage parameters
    S1_max: Array  # Maximum upper layer storage (mm)
    S2_max: Array  # Maximum lower layer storage (mm)
    f_tens: Array  # Fraction of storage as tension
    f_rchr: Array  # Fraction tension in primary zone
    f_base: Array  # Fraction free storage in primary reservoir

    # Evaporation parameters
    r1: Array  # Root fraction in upper layer

    # Percolation parameters
    ku: Array  # Upper layer drainage rate (1/day)
    c: Array  # VIC/ARNO infiltration shape
    alpha: Array  # Sacramento percolation shape
    psi: Array  # Sacramento lower zone demand coefficient
    kappa: Array  # Fraction of percolation to tension storage

    # Lateral flow parameters
    ki: Array  # Interflow rate (1/day)
    ks: Array  # Baseflow rate (1/day)

    # Baseflow parameters
    n: Array  # TOPMODEL decay parameter
    v: Array  # Linear baseflow rate (1/day)
    v_A: Array  # Primary reservoir rate (1/day)
    v_B: Array  # Secondary reservoir rate (1/day)

    # Saturated area parameters
    Ac_max: Array  # Maximum saturated area fraction
    b: Array  # VIC 'b' parameter
    lam: Array  # Pareto shape (renamed from lambda)
    chi: Array  # TOPMODEL shape parameter

    # Time scale parameters
    mu_t: Array  # Time scale for percolation (days)

    # Snow parameters
    T_rain: Array  # Rain/snow threshold (°C)
    T_melt: Array  # Snowmelt threshold (°C)
    melt_rate: Array  # Degree-day melt factor (mm/°C/day)
    lapse_rate: Array  # Temperature lapse rate (°C/100m)
    opg: Array  # Orographic precipitation gradient
    MFMAX: Array  # Maximum seasonal melt factor
    MFMIN: Array  # Minimum seasonal melt factor

    # Smoothing
    smooth_frac: Array  # Smoothing fraction for overflow

    # Glacier parameters (appended to keep legacy PARAM_NAMES indices stable)
    DDF_ice: Array  # Ice degree-day melt factor (mm/°C/day)
    T_ice: Array  # Ice-melt threshold temperature (°C)
    K_glac: Array  # Glacier-reservoir release coefficient (1/day)

    # Derived parameters (computed from above)
    S1_T_max: Array  # = f_tens * S1_max
    S1_F_max: Array  # = (1 - f_tens) * S1_max
    S1_TA_max: Array  # = f_rchr * S1_T_max
    S1_TB_max: Array  # = (1 - f_rchr) * S1_T_max
    S2_T_max: Array  # = f_tens * S2_max
    S2_F_max: Array  # = (1 - f_tens) * S2_max
    S2_FA_max: Array  # = f_base * S2_F_max
    S2_FB_max: Array  # = (1 - f_base) * S2_F_max
    m: Array  # TOPMODEL: S2_max / n

    @classmethod
    def default(cls, n_hrus: int = 1) -> "Parameters":
        """Create default parameters."""
        shape = (n_hrus,) if n_hrus > 1 else ()
        # Use float32 for compatibility with neuralgcm (which requires JAX_ENABLE_X64=False)
        dtype = jnp.float32

        # Adjustable parameters with reasonable defaults
        params = {
            "S1_max": 200.0,
            "S2_max": 800.0,
            "f_tens": 0.4,
            "f_rchr": 0.3,
            "f_base": 0.3,
            "r1": 0.5,
            "ku": 0.1,
            "c": 2.0,
            "alpha": 2.0,
            "psi": 0.5,
            "kappa": 0.5,
            "ki": 0.05,
            "ks": 0.01,
            "n": 3.0,
            "v": 0.1,
            "v_A": 0.1,
            "v_B": 0.01,
            "Ac_max": 0.5,
            "b": 0.5,
            "lam": 1.0,
            "chi": 5.0,
            "mu_t": 10.0,
            "T_rain": 1.0,
            "T_melt": 0.0,
            "melt_rate": 3.0,
            "lapse_rate": 0.65,
            "opg": 5.0,
            "MFMAX": 4.5,
            "MFMIN": 1.0,
            "smooth_frac": 0.01,
            # Glacier: ice melts ~1.6x faster than snow by default. K_glac=0.1
            # buffers the glacier reservoir to a few days' residence (faster
            # over-attenuates and made glacier-fed rivers too flashy).
            "DDF_ice": 7.0,
            "T_ice": 0.0,
            "K_glac": 0.1,
        }

        # Convert to arrays with explicit dtype
        arr_params = {k: jnp.full(shape, v, dtype=dtype) for k, v in params.items()}

        # Compute derived parameters
        S1_max = arr_params["S1_max"]
        S2_max = arr_params["S2_max"]
        f_tens = arr_params["f_tens"]
        f_rchr = arr_params["f_rchr"]
        f_base = arr_params["f_base"]
        n = arr_params["n"]

        S1_T_max = f_tens * S1_max
        S1_F_max = (1.0 - f_tens) * S1_max
        S1_TA_max = f_rchr * S1_T_max
        S1_TB_max = (1.0 - f_rchr) * S1_T_max
        S2_T_max = f_tens * S2_max
        S2_F_max = (1.0 - f_tens) * S2_max
        S2_FA_max = f_base * S2_F_max
        S2_FB_max = (1.0 - f_base) * S2_F_max
        m = S2_max / jnp.maximum(n, 0.1)

        return cls(
            **arr_params,
            S1_T_max=S1_T_max,
            S1_F_max=S1_F_max,
            S1_TA_max=S1_TA_max,
            S1_TB_max=S1_TB_max,
            S2_T_max=S2_T_max,
            S2_F_max=S2_F_max,
            S2_FA_max=S2_FA_max,
            S2_FB_max=S2_FB_max,
            m=m,
        )

    @staticmethod
    def _default_scalar(name: str) -> float:
        """Default value for a parameter, used to pad legacy (pre-glacier)
        parameter arrays. Glacier params get physical defaults; anything else
        falls back to the midpoint of its bounds."""
        glacier_defaults = {"DDF_ice": 7.0, "T_ice": 0.0, "K_glac": 0.1}
        if name in glacier_defaults:
            return glacier_defaults[name]
        lo, hi = PARAM_BOUNDS.get(name, (0.0, 1.0))
        return 0.5 * (lo + hi)

    @classmethod
    def from_array(cls, arr: Array, n_hrus: int = 1) -> "Parameters":
        """Create parameters from flat array.

        Args:
            arr: Array of shape (NUM_PARAMETERS,) or (n_hrus, NUM_PARAMETERS)
            n_hrus: Number of HRUs

        Returns:
            Parameters instance with derived quantities computed
        """
        # Handle both single and batched cases
        if arr.ndim == 1:
            arr = arr[None, :]  # Add HRU dimension

        # Pad legacy arrays (pre-glacier, < NUM_PARAMETERS columns) with the
        # default value for each missing trailing parameter so older calibration
        # vectors and saved para_def arrays keep loading unchanged.
        n_given = arr.shape[-1]
        if n_given < NUM_PARAMETERS:
            n_rows = arr.shape[0]
            pad_cols = [
                jnp.full((n_rows,), Parameters._default_scalar(name), dtype=arr.dtype)
                for name in PARAM_NAMES[n_given:]
            ]
            arr = jnp.concatenate([arr, jnp.stack(pad_cols, axis=-1)], axis=-1)

        # Extract adjustable parameters
        params = {name: arr[:, i] for i, name in enumerate(PARAM_NAMES)}

        # Squeeze if single HRU
        if n_hrus == 1:
            params = {k: v.squeeze(0) for k, v in params.items()}

        # Compute derived parameters
        S1_max = params["S1_max"]
        S2_max = params["S2_max"]
        f_tens = params["f_tens"]
        f_rchr = params["f_rchr"]
        f_base = params["f_base"]
        n = params["n"]

        S1_T_max = f_tens * S1_max
        S1_F_max = (1.0 - f_tens) * S1_max
        S1_TA_max = f_rchr * S1_T_max
        S1_TB_max = (1.0 - f_rchr) * S1_T_max
        S2_T_max = f_tens * S2_max
        S2_F_max = (1.0 - f_tens) * S2_max
        S2_FA_max = f_base * S2_F_max
        S2_FB_max = (1.0 - f_base) * S2_F_max
        m = S2_max / jnp.maximum(n, 0.1)

        return cls(
            **params,
            S1_T_max=S1_T_max,
            S1_F_max=S1_F_max,
            S1_TA_max=S1_TA_max,
            S1_TB_max=S1_TB_max,
            S2_T_max=S2_T_max,
            S2_F_max=S2_F_max,
            S2_FA_max=S2_FA_max,
            S2_FB_max=S2_FB_max,
            m=m,
        )

    def to_array(self) -> Array:
        """Convert adjustable parameters to flat array."""
        return jnp.stack([getattr(self, name) for name in PARAM_NAMES], axis=-1)

    def validate_bounds(self, warn: bool = True) -> bool:
        """Check if parameters are within valid bounds.

        Args:
            warn: If True, print warnings for out-of-bounds parameters

        Returns:
            True if all parameters are within bounds, False otherwise
        """
        import warnings

        all_valid = True

        for name in PARAM_NAMES:
            if name not in PARAM_BOUNDS:
                continue
            value = getattr(self, name)
            low, high = PARAM_BOUNDS[name]

            # Handle both scalar and array values
            val_min = float(jnp.min(value))
            val_max = float(jnp.max(value))

            if val_min < low or val_max > high:
                all_valid = False
                if warn:
                    warnings.warn(
                        f"Parameter '{name}' out of bounds: "
                        f"value range [{val_min:.4f}, {val_max:.4f}] "
                        f"not in [{low}, {high}]",
                        UserWarning,
                    )

        return all_valid

    @classmethod
    def from_array_validated(cls, arr: Array, n_hrus: int = 1, clip: bool = True) -> "Parameters":
        """Create parameters from flat array with optional bounds enforcement.

        Args:
            arr: Array of shape (NUM_PARAMETERS,) or (n_hrus, NUM_PARAMETERS)
            n_hrus: Number of HRUs
            clip: If True, clip values to valid bounds

        Returns:
            Parameters instance with derived quantities computed
        """
        params = cls.from_array(arr, n_hrus)

        if clip:
            # Clip each parameter to its bounds
            clipped_values = {}
            for name in PARAM_NAMES:
                value = getattr(params, name)
                if name in PARAM_BOUNDS:
                    low, high = PARAM_BOUNDS[name]
                    clipped_values[name] = jnp.clip(value, low, high)
                else:
                    clipped_values[name] = value

            # Recompute derived parameters
            S1_max = clipped_values["S1_max"]
            S2_max = clipped_values["S2_max"]
            f_tens = clipped_values["f_tens"]
            f_rchr = clipped_values["f_rchr"]
            f_base = clipped_values["f_base"]
            n = clipped_values["n"]

            S1_T_max = f_tens * S1_max
            S1_F_max = (1.0 - f_tens) * S1_max
            S1_TA_max = f_rchr * S1_T_max
            S1_TB_max = (1.0 - f_rchr) * S1_T_max
            S2_T_max = f_tens * S2_max
            S2_F_max = (1.0 - f_tens) * S2_max
            S2_FA_max = f_base * S2_F_max
            S2_FB_max = (1.0 - f_base) * S2_F_max
            m = S2_max / jnp.maximum(n, 0.1)

            return cls(
                **clipped_values,
                S1_T_max=S1_T_max,
                S1_F_max=S1_F_max,
                S1_TA_max=S1_TA_max,
                S1_TB_max=S1_TB_max,
                S2_T_max=S2_T_max,
                S2_F_max=S2_F_max,
                S2_FA_max=S2_FA_max,
                S2_FB_max=S2_FB_max,
                m=m,
            )

        return params
