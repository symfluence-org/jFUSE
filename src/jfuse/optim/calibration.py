"""Gradient-based calibration for jFUSE.

This module provides high-level calibration utilities using optax optimizers.
Supports multi-objective optimization, parameter bounds, and adaptive learning rates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Any, NamedTuple
import time

import jax
import jax.numpy as jnp
import optax
from jax import random

from ..fuse.state import Parameters, PARAM_BOUNDS


class CalibrationState(NamedTuple):
    """State maintained during calibration."""

    params: Parameters
    opt_state: Any
    step: int
    best_loss: float
    best_params: Parameters
    loss_history: jnp.ndarray
    grad_norm_history: jnp.ndarray


@dataclass
class CalibrationConfig:
    """Configuration for calibration runs.

    Attributes:
        max_iterations: Maximum number of optimization steps
        learning_rate: Initial learning rate for optimizer
        warmup_steps: Number of warmup steps for learning rate schedule
        min_learning_rate: Minimum learning rate after decay
        patience: Steps without improvement before early stopping
        min_delta: Minimum improvement to reset patience counter
        clip_gradients: Maximum gradient norm for clipping
        optimizer: Name of optax optimizer ('adam', 'adamw', 'sgd', 'rmsprop')
        weight_decay: Weight decay coefficient for adamw
        log_every: Log progress every N steps
        checkpoint_every: Save checkpoint every N steps
        seed: Random seed for reproducibility
    """

    max_iterations: int = 1000
    learning_rate: float = 0.01
    warmup_steps: int = 100
    min_learning_rate: float = 1e-5
    patience: int = 100
    min_delta: float = 1e-6
    clip_gradients: float = 1.0
    optimizer: str = "adam"
    weight_decay: float = 0.0
    log_every: int = 10
    checkpoint_every: int = 100
    seed: int = 42


def create_optimizer(config: CalibrationConfig) -> optax.GradientTransformation:
    """Create optax optimizer with learning rate schedule.

    Args:
        config: Calibration configuration

    Returns:
        Chained optax transformation
    """
    # Learning rate schedule: warmup then cosine decay
    # Note: decay_steps is the TOTAL length including warmup, not just decay portion
    # Ensure there's at least 1 step of actual decay
    total_steps = max(config.warmup_steps + 1, config.max_iterations)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=config.min_learning_rate,
        peak_value=config.learning_rate,
        warmup_steps=config.warmup_steps,
        decay_steps=total_steps,
        end_value=config.min_learning_rate,
    )

    # Select base optimizer
    if config.optimizer == "adam":
        base_opt = optax.adam(learning_rate=schedule)
    elif config.optimizer == "adamw":
        base_opt = optax.adamw(learning_rate=schedule, weight_decay=config.weight_decay)
    elif config.optimizer == "sgd":
        base_opt = optax.sgd(learning_rate=schedule, momentum=0.9)
    elif config.optimizer == "rmsprop":
        base_opt = optax.rmsprop(learning_rate=schedule)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")

    # Chain with gradient clipping
    return optax.chain(
        optax.clip_by_global_norm(config.clip_gradients),
        base_opt,
    )


def _get_field_names(obj) -> list:
    """Get field names from dataclass, NamedTuple, or equinox.Module."""
    import dataclasses

    if dataclasses.is_dataclass(obj):
        return [f.name for f in dataclasses.fields(obj)]
    elif hasattr(obj, "_fields"):
        return list(obj._fields)
    else:
        raise TypeError(f"Cannot get fields from {type(obj)}")


def _reconstruct_parameters(params: Parameters, base_values: dict) -> Parameters:
    """Reconstruct Parameters with recomputed derived values.

    Args:
        params: Original Parameters object (for type reference)
        base_values: Dictionary of base parameter values

    Returns:
        New Parameters with derived values recomputed
    """
    # Extract values needed for derived computation
    S1_max = base_values["S1_max"]
    S2_max = base_values["S2_max"]
    f_tens = base_values["f_tens"]
    f_rchr = base_values["f_rchr"]
    f_base = base_values["f_base"]
    n = base_values["n"]

    # Recompute derived parameters
    S1_T_max = f_tens * S1_max
    S1_F_max = (1.0 - f_tens) * S1_max
    S1_TA_max = f_rchr * S1_T_max
    S1_TB_max = (1.0 - f_rchr) * S1_T_max
    S2_T_max = f_tens * S2_max
    S2_F_max = (1.0 - f_tens) * S2_max
    S2_FA_max = f_base * S2_F_max
    S2_FB_max = (1.0 - f_base) * S2_F_max
    m = S2_max / jnp.maximum(n, 0.1)

    # Combine base and derived
    all_values = {
        **base_values,
        "S1_T_max": S1_T_max,
        "S1_F_max": S1_F_max,
        "S1_TA_max": S1_TA_max,
        "S1_TB_max": S1_TB_max,
        "S2_T_max": S2_T_max,
        "S2_F_max": S2_F_max,
        "S2_FA_max": S2_FA_max,
        "S2_FB_max": S2_FB_max,
        "m": m,
    }

    return params.__class__(**all_values)


# List of derived parameter names that should not be transformed
_DERIVED_PARAMS = {
    "S1_T_max",
    "S1_F_max",
    "S1_TA_max",
    "S1_TB_max",
    "S2_T_max",
    "S2_F_max",
    "S2_FA_max",
    "S2_FB_max",
    "m",
}


def transform_to_unbounded(params: Parameters) -> Parameters:
    """Transform bounded parameters to unbounded space using logit.

    This allows unconstrained optimization while respecting parameter bounds.
    Uses the transformation: x_unbounded = logit((x - low) / (high - low))

    Args:
        params: Parameters in bounded space

    Returns:
        Parameters in unbounded space
    """
    from jfuse.coupled import CoupledParams

    def logit_transform(x, low: float, high: float):
        # Normalize to [0, 1]
        normalized = (x - low) / (high - low)
        # Clip to avoid infinities
        normalized = jnp.clip(normalized, 1e-6, 1 - 1e-6)
        # Logit transform
        return jnp.log(normalized / (1 - normalized))

    # Handle CoupledParams specially - only transform the FUSE params
    if isinstance(params, CoupledParams):
        transformed_fuse = transform_to_unbounded(params.fuse_params)
        return CoupledParams(
            fuse_params=transformed_fuse,
            manning_n=params.manning_n,
            width_coef=params.width_coef,
            width_exp=params.width_exp,
            depth_coef=params.depth_coef,
            depth_exp=params.depth_exp,
        )

    # Get field names
    field_names = _get_field_names(params)

    # Transform only base parameters (not derived ones)
    base_values = {}
    for name in field_names:
        if name in _DERIVED_PARAMS:
            continue  # Skip derived parameters
        value = getattr(params, name)
        if name in PARAM_BOUNDS:
            low, high = PARAM_BOUNDS[name]
            base_values[name] = logit_transform(value, low, high)
        else:
            base_values[name] = value

    # Reconstruct with recomputed derived parameters
    return _reconstruct_parameters(params, base_values)


def transform_to_bounded(params: Parameters) -> Parameters:
    """Transform unbounded parameters back to bounded space using sigmoid.

    Inverse of transform_to_unbounded.
    Uses the transformation: x_bounded = low + (high - low) * sigmoid(x_unbounded)

    Args:
        params: Parameters in unbounded space

    Returns:
        Parameters in bounded space
    """
    from jfuse.coupled import CoupledParams

    def sigmoid_transform(x, low: float, high: float):
        sigmoid = 1 / (1 + jnp.exp(-x))
        return low + (high - low) * sigmoid

    # Handle CoupledParams specially - only transform the FUSE params
    if isinstance(params, CoupledParams):
        transformed_fuse = transform_to_bounded(params.fuse_params)
        return CoupledParams(
            fuse_params=transformed_fuse,
            manning_n=params.manning_n,
            width_coef=params.width_coef,
            width_exp=params.width_exp,
            depth_coef=params.depth_coef,
            depth_exp=params.depth_exp,
        )

    # Get field names
    field_names = _get_field_names(params)

    # Transform only base parameters (not derived ones)
    base_values = {}
    for name in field_names:
        if name in _DERIVED_PARAMS:
            continue  # Skip derived parameters
        value = getattr(params, name)
        if name in PARAM_BOUNDS:
            low, high = PARAM_BOUNDS[name]
            base_values[name] = sigmoid_transform(value, low, high)
        else:
            base_values[name] = value

    # Reconstruct with recomputed derived parameters
    return _reconstruct_parameters(params, base_values)


def clip_to_bounds(params: Parameters) -> Parameters:
    """Clip parameters to their valid bounds.

    Args:
        params: Parameters potentially outside bounds

    Returns:
        Parameters clipped to valid bounds
    """
    from jfuse.coupled import CoupledParams

    # Handle CoupledParams specially
    if isinstance(params, CoupledParams):
        clipped_fuse = clip_to_bounds(params.fuse_params)
        return CoupledParams(
            fuse_params=clipped_fuse,
            manning_n=params.manning_n,
            width_coef=params.width_coef,
            width_exp=params.width_exp,
            depth_coef=params.depth_coef,
            depth_exp=params.depth_exp,
        )

    # Get field names
    field_names = _get_field_names(params)

    # Clip only base parameters (not derived ones)
    base_values = {}
    for name in field_names:
        if name in _DERIVED_PARAMS:
            continue  # Skip derived parameters
        value = getattr(params, name)
        if name in PARAM_BOUNDS:
            low, high = PARAM_BOUNDS[name]
            base_values[name] = jnp.clip(value, low, high)
        else:
            base_values[name] = value

    # Reconstruct with recomputed derived parameters
    return _reconstruct_parameters(params, base_values)


def compute_grad_norm(grads: Parameters) -> float:
    """Compute L2 norm of parameter gradients.

    Args:
        grads: Gradient structure matching Parameters

    Returns:
        Scalar L2 norm
    """
    grad_leaves = jax.tree_util.tree_leaves(grads)
    squared_sum = sum(jnp.sum(g**2) for g in grad_leaves)
    return jnp.sqrt(squared_sum)


class Calibrator:
    """High-level calibration interface for jFUSE models.

    Supports both FUSE-only and coupled FUSE+routing calibration with
    gradient-based optimization using optax.

    Example:
        >>> from jfuse import CoupledModel
        >>> from jfuse.optim import Calibrator, CalibrationConfig
        >>>
        >>> model = CoupledModel(...)
        >>> config = CalibrationConfig(max_iterations=500, learning_rate=0.01)
        >>> calibrator = Calibrator(model, config)
        >>>
        >>> result = calibrator.calibrate(
        ...     forcing=forcing_data,
        ...     observed=observed_discharge,
        ...     loss_fn='kge'
        ... )
        >>> print(f"Final KGE: {1 - result['final_loss']:.3f}")
    """

    def __init__(
        self,
        model: Any,
        config: Optional[CalibrationConfig] = None,
    ):
        """Initialize calibrator.

        Args:
            model: FUSEModel or CoupledModel instance
            config: Calibration configuration (uses defaults if None)
        """
        self.model = model
        self.config = config or CalibrationConfig()
        self.optimizer = create_optimizer(self.config)
        # Optional pre-built loss function. When set (e.g. by
        # calibrate_multi_site), calibrate() uses it instead of building one
        # from a single forcing/observed pair.
        self._loss_fn = None

    def _create_loss_fn(
        self,
        forcing: Tuple,
        observed: jnp.ndarray,
        loss_type: str = "kge",
        warmup_steps: int = 365,
        weights: Optional[Dict[str, float]] = None,
    ) -> Callable[[Parameters], float]:
        """Create loss function for calibration.

        Args:
            forcing: Tuple of forcing arrays (precip, pet, temp)
            observed: Observed discharge array
            loss_type: Loss type - single ('kge', 'nse', 'rmse', 'mse', 'mae')
                       or comma-separated for multi-objective ('kge,nse')
            warmup_steps: Number of timesteps to skip for loss calculation
            weights: Weights for multi-objective loss (equal weights if None)

        Returns:
            Loss function taking parameters and returning scalar loss
        """
        from ..coupled import nse_loss, kge_loss, mse_loss, rmse_loss, mae_loss, CoupledModel

        # Map loss names to functions
        loss_functions = {
            "kge": kge_loss,
            "nse": nse_loss,
            "mse": mse_loss,
            "rmse": rmse_loss,
            "mae": mae_loss,
        }

        # Parse loss_type - could be single or comma-separated
        loss_types = [lt.strip().lower() for lt in loss_type.split(",")]

        # Validate loss types
        for lt in loss_types:
            if lt not in loss_functions:
                raise ValueError(
                    f"Unknown loss type: {lt}. Available: {list(loss_functions.keys())}"
                )

        # Set up weights for multi-objective
        if len(loss_types) > 1:
            if weights is None:
                # Equal weights
                weights = {lt: 1.0 / len(loss_types) for lt in loss_types}
            else:
                # Normalize weights
                total_w = sum(weights.get(lt, 0.0) for lt in loss_types)
                if total_w > 0:
                    weights = {lt: weights.get(lt, 0.0) / total_w for lt in loss_types}
                else:
                    weights = {lt: 1.0 / len(loss_types) for lt in loss_types}

        # Ensure observations are 1D (outlet only)
        obs_1d = observed
        if observed.ndim > 1:
            # Take first column (outlet) or squeeze if single column
            if observed.shape[1] == 1:
                obs_1d = observed.squeeze(-1)
            else:
                obs_1d = observed[:, 0]  # Assume first column is outlet

        def loss_fn(params: Parameters) -> float:
            # Run simulation - detect model type properly
            if isinstance(self.model, CoupledModel):
                # Coupled model returns (outlet_Q_m3s, runoff_mm_day)
                # NOTE: outlet_Q is in m³/s, but obs are in mm/day
                # Use aggregated runoff for calibration (same units as obs)
                outlet_Q, runoff = self.model.simulate(forcing, params)

                # Aggregate runoff to outlet (area-weighted mean)
                if runoff.ndim > 1:
                    sim = jnp.mean(runoff, axis=1)  # mm/day
                else:
                    sim = runoff
            else:
                # FUSEModel returns (runoff, final_state)
                state = self.model.default_state()
                runoff, _ = self.model.simulate(forcing, params, state)

                # Aggregate if distributed
                if runoff.ndim > 1:
                    sim = jnp.mean(runoff, axis=1)  # Area-weighted average
                else:
                    sim = runoff

            # Apply warmup
            sim_eval = sim[warmup_steps:]
            obs_eval = obs_1d[warmup_steps:]

            # Single objective
            if len(loss_types) == 1:
                return loss_functions[loss_types[0]](sim_eval, obs_eval)

            # Multi-objective: weighted sum
            total = 0.0
            for lt in loss_types:
                loss_val = loss_functions[lt](sim_eval, obs_eval)

                # Normalize RMSE/MSE/MAE by observed std for scale invariance
                if lt in ["rmse", "mse", "mae"]:
                    obs_std = jnp.std(obs_eval)
                    if lt == "mse":
                        loss_val = loss_val / jnp.maximum(obs_std**2, 1e-6)
                    else:
                        loss_val = loss_val / jnp.maximum(obs_std, 1e-6)

                total += weights[lt] * loss_val

            return total

        return loss_fn

    def _make_step_fn(
        self,
        loss_fn: Callable[[Parameters], float],
        use_bounded_transform: bool = True,
    ) -> Callable:
        """Create JIT-compiled optimization step function.

        Args:
            loss_fn: Loss function taking parameters
            use_bounded_transform: Whether to use logit/sigmoid transforms

        Returns:
            JIT-compiled step function
        """

        @jax.jit
        def step_bounded(params_unbounded, opt_state):
            """Step with bounded parameter transform."""

            def loss_wrapper(p_unbounded):
                p_bounded = transform_to_bounded(p_unbounded)
                return loss_fn(p_bounded)

            loss, grads = jax.value_and_grad(loss_wrapper)(params_unbounded)
            updates, new_opt_state = self.optimizer.update(grads, opt_state, params_unbounded)
            new_params = optax.apply_updates(params_unbounded, updates)
            grad_norm = compute_grad_norm(grads)

            return new_params, new_opt_state, loss, grad_norm

        @jax.jit
        def step_clipped(params, opt_state):
            """Step with gradient clipping and bound enforcement."""
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            new_params = clip_to_bounds(new_params)
            grad_norm = compute_grad_norm(grads)

            return new_params, new_opt_state, loss, grad_norm

        return step_bounded if use_bounded_transform else step_clipped

    def calibrate(
        self,
        forcing: Tuple,
        observed: jnp.ndarray,
        initial_params: Optional[Parameters] = None,
        loss_fn: str = "kge",
        warmup_steps: int = 365,
        use_bounded_transform: bool = True,
        callback: Optional[Callable[[int, float, Parameters], None]] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Run gradient-based calibration.

        Args:
            forcing: Tuple of forcing arrays (precip, pet, temp)
            observed: Observed discharge array
            initial_params: Starting parameters (uses defaults if None)
            loss_fn: Loss type ('kge', 'nse', 'rmse', 'multi')
            warmup_steps: Number of timesteps to skip for loss
            use_bounded_transform: Use logit/sigmoid for bounds
            callback: Optional function called each step
            verbose: Print progress messages

        Returns:
            Dictionary with calibration results:
                - final_params: Calibrated parameters
                - final_loss: Final loss value
                - best_params: Best parameters found
                - best_loss: Best loss value
                - loss_history: Array of loss values
                - grad_norm_history: Array of gradient norms
                - n_iterations: Number of iterations run
                - converged: Whether early stopping triggered
                - elapsed_time: Total calibration time in seconds
        """
        # Initialize parameters
        if initial_params is None:
            initial_params = self.model.default_params()

        # Create loss function (or use a pre-built one, e.g. from
        # calibrate_multi_site). A pre-built loss is consumed once so later
        # single-site calibrate() calls on the same Calibrator are unaffected.
        if self._loss_fn is not None:
            loss_function = self._loss_fn
            self._loss_fn = None
        else:
            loss_function = self._create_loss_fn(forcing, observed, loss_fn, warmup_steps)

        # Create step function
        step_fn = self._make_step_fn(loss_function, use_bounded_transform)

        # Initialize optimizer state
        if use_bounded_transform:
            params = transform_to_unbounded(initial_params)
        else:
            params = initial_params

        opt_state = self.optimizer.init(params)

        # Tracking variables
        best_loss = float("inf")
        best_params = initial_params
        patience_counter = 0
        loss_history = []
        grad_norm_history = []

        start_time = time.time()

        if verbose:
            print(f"Starting calibration with {self.config.max_iterations} max iterations")
            print(f"Optimizer: {self.config.optimizer}, LR: {self.config.learning_rate}")

        # Optimization loop
        for i in range(self.config.max_iterations):
            params, opt_state, loss, grad_norm = step_fn(params, opt_state)

            loss_val = float(loss)
            grad_norm_val = float(grad_norm)
            loss_history.append(loss_val)
            grad_norm_history.append(grad_norm_val)

            # Track best
            if loss_val < best_loss - self.config.min_delta:
                best_loss = loss_val
                if use_bounded_transform:
                    best_params = transform_to_bounded(params)
                else:
                    best_params = params
                patience_counter = 0
            else:
                patience_counter += 1

            # Callback
            if callback is not None:
                current_params = transform_to_bounded(params) if use_bounded_transform else params
                callback(i, loss_val, current_params)

            # Logging
            if verbose and (i % self.config.log_every == 0 or i == self.config.max_iterations - 1):
                metric_val = 1 - loss_val if loss_fn == "kge" else loss_val
                print(
                    f"Step {i:4d}: Loss = {loss_val:.6f}, "
                    f"{'KGE' if loss_fn == 'kge' else 'Metric'} = {metric_val:.4f}, "
                    f"|grad| = {grad_norm_val:.6f}"
                )

            # Early stopping
            if patience_counter >= self.config.patience:
                if verbose:
                    print(
                        f"Early stopping at iteration {i} (no improvement for {self.config.patience} steps)"
                    )
                break

            # Check for convergence
            if grad_norm_val < 1e-8:
                if verbose:
                    print(f"Converged at iteration {i} (gradient norm < 1e-8)")
                break

        elapsed_time = time.time() - start_time

        # Final parameters
        if use_bounded_transform:
            final_params = transform_to_bounded(params)
        else:
            final_params = params

        if verbose:
            print(f"\nCalibration complete in {elapsed_time:.1f}s")
            print(f"Final loss: {loss_history[-1]:.6f}")
            print(f"Best loss: {best_loss:.6f}")

        return {
            "final_params": final_params,
            "final_loss": loss_history[-1],
            "best_params": best_params,
            "best_loss": best_loss,
            "loss_history": jnp.array(loss_history),
            "grad_norm_history": jnp.array(grad_norm_history),
            "n_iterations": len(loss_history),
            "converged": patience_counter >= self.config.patience or grad_norm_val < 1e-8,
            "elapsed_time": elapsed_time,
        }

    def calibrate_multi_site(
        self,
        forcing_list: List[Tuple],
        observed_list: List[jnp.ndarray],
        site_weights: Optional[List[float]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Calibrate using multiple observation sites.

        Args:
            forcing_list: List of forcing tuples for each site
            observed_list: List of observed arrays for each site
            site_weights: Weights for each site (uniform if None)
            **kwargs: Additional arguments passed to calibrate()

        Returns:
            Calibration results dictionary
        """
        n_sites = len(forcing_list)
        if site_weights is None:
            site_weights = [1.0 / n_sites] * n_sites

        # _create_loss_fn only accepts loss-related arguments; the remaining
        # kwargs (use_bounded_transform, callback, verbose, ...) are forwarded
        # to calibrate() below.
        loss_type = kwargs.get("loss_fn", "kge")
        warmup_steps = kwargs.get("warmup_steps", 365)

        site_losses = [
            self._create_loss_fn(forcing, observed, loss_type, warmup_steps)
            for forcing, observed in zip(forcing_list, observed_list)
        ]

        def multi_site_loss(params: Parameters) -> float:
            total_loss = 0.0
            for loss_fn, weight in zip(site_losses, site_weights):
                total_loss += weight * loss_fn(params)
            return total_loss

        # calibrate() picks up this pre-built loss instead of building its own.
        self._loss_fn = multi_site_loss

        # Run calibration with the custom multi-site loss. forcing/observed are
        # placeholders (ignored because _loss_fn is set).
        return self.calibrate(
            forcing=forcing_list[0],
            observed=observed_list[0],
            **kwargs,
        )


def random_search(
    model: Any,
    forcing: Tuple,
    observed: jnp.ndarray,
    n_samples: int = 100,
    loss_fn: str = "kge",
    warmup_steps: int = 365,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[Parameters, float]:
    """Random search for good initial parameters.

    Useful for finding a good starting point for gradient-based optimization.

    Args:
        model: FUSEModel or CoupledModel
        forcing: Forcing data tuple
        observed: Observed discharge
        n_samples: Number of random samples to try
        loss_fn: Loss type
        warmup_steps: Warmup timesteps
        seed: Random seed
        verbose: Print progress

    Returns:
        Tuple of (best_params, best_loss)
    """
    key = random.PRNGKey(seed)

    best_params = None
    best_loss = float("inf")

    # Get default params as template
    default_params = model.default_params()

    # Create loss function
    calibrator = Calibrator(model)
    loss_function = calibrator._create_loss_fn(forcing, observed, loss_fn, warmup_steps)

    for i in range(n_samples):
        key, subkey = random.split(key)

        # Generate random parameters within bounds
        param_values = {}
        for name in default_params.__dataclass_fields__:
            key, param_key = random.split(key)
            if name in PARAM_BOUNDS:
                low, high = PARAM_BOUNDS[name]
                param_values[name] = random.uniform(param_key, minval=low, maxval=high)
            else:
                param_values[name] = getattr(default_params, name)

        params = default_params.__class__(**param_values)

        try:
            loss = float(loss_function(params))

            if loss < best_loss:
                best_loss = loss
                best_params = params

                if verbose:
                    print(f"Sample {i+1}/{n_samples}: New best loss = {loss:.6f}")
        except Exception as e:
            if verbose:
                print(f"Sample {i+1}/{n_samples}: Failed - {e}")
            continue

    if verbose:
        print(f"\nRandom search complete. Best loss: {best_loss:.6f}")

    return best_params, best_loss
