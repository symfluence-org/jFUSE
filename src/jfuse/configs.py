# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2024-2026 jFUSE Contributors

"""Predefined gradient configs and Fortran-FUSE decision mapping.

Centralizes the gradient-optimized model configurations, the mapping from
Fortran FUSE decision names to jFUSE enums, the name->config registry, and the
decision-to-config builder. These were previously duplicated (and at risk of
drifting out of sync) between :mod:`jfuse.runner` and
:mod:`jfuse.calibration.worker`.
"""

from typing import Any, Dict

from .fuse.config import (
    PRMS_CONFIG,
    SACRAMENTO_CONFIG,
    TOPMODEL_CONFIG,
    VIC_CONFIG,
    BaseflowType,
    EvaporationType,
    InterflowType,
    LowerLayerArch,
    ModelConfig,
    PercolationType,
    RainfallErrorType,
    RoutingType,
    SnowType,
    SurfaceRunoffType,
    UpperLayerArch,
)

# Custom config optimized for gradient-based calibration (ADAM/LBFGS).
# Uses NONLINEAR baseflow to enable gradients for ks, S2_max, and n.
# Uses UZ_PARETO surface runoff to enable gradients for S1_max.
# Working parameters (14): S1_max, S2_max, ku, ki, ks, n, Ac_max, b, f_rchr,
#                          T_rain, T_melt, MFMAX, MFMIN, smooth_frac
PRMS_GRADIENT_CONFIG = ModelConfig(
    upper_arch=UpperLayerArch.TENSION2_FREE,     # PRMS-style 3-state upper layer
    lower_arch=LowerLayerArch.SINGLE_NOEVAP,     # Single lower reservoir
    baseflow=BaseflowType.NONLINEAR,             # qb = ks*S2_max*(S2/S2_max)^(1+n)
    percolation=PercolationType.FREE_STORAGE,    # From free storage
    surface_runoff=SurfaceRunoffType.UZ_PARETO,  # Pareto - activates S1_max gradient
    evaporation=EvaporationType.SEQUENTIAL,
    interflow=InterflowType.LINEAR,              # Uses ki
    snow=SnowType.TEMP_INDEX,
    routing=RoutingType.NONE,
    rainfall_error=RainfallErrorType.ADDITIVE,
)

# Maximum gradient config - Sacramento-based architecture for most parameters.
MAX_GRADIENT_CONFIG = ModelConfig(
    upper_arch=UpperLayerArch.TENSION2_FREE,      # f_tens, f_rchr, smooth_frac
    lower_arch=LowerLayerArch.TENSION_2RESERV,    # f_base, kappa, lower evap
    baseflow=BaseflowType.PARALLEL_LINEAR,        # v_A, v_B (forced by TENSION_2RESERV)
    percolation=PercolationType.LOWER_DEMAND,     # ku, alpha, psi
    surface_runoff=SurfaceRunoffType.UZ_PARETO,   # Ac_max, b, S1_max
    evaporation=EvaporationType.ROOT_WEIGHT,      # r1
    interflow=InterflowType.LINEAR,               # ki
    snow=SnowType.TEMP_INDEX,                     # T_rain, T_melt, MFMAX, MFMIN
    routing=RoutingType.NONE,
    rainfall_error=RainfallErrorType.ADDITIVE,
)

# Mapping from Fortran FUSE decision names to jFUSE enums.
FUSE_DECISION_MAP = {
    'ARCH1': {
        'onestate_1': UpperLayerArch.SINGLE_STATE,
        'tension1_1': UpperLayerArch.TENSION_FREE,
        'tension2_1': UpperLayerArch.TENSION2_FREE,
    },
    'ARCH2': {
        'fixedsiz_2': LowerLayerArch.SINGLE_NOEVAP,
        'unlimfrc_2': LowerLayerArch.SINGLE_NOEVAP,
        'unlimpow_2': LowerLayerArch.SINGLE_EVAP,
        'tens2pll_2': LowerLayerArch.TENSION_2RESERV,
    },
    'ARCH2_BASEFLOW': {
        'fixedsiz_2': BaseflowType.LINEAR,
        'unlimfrc_2': BaseflowType.LINEAR,
        'unlimpow_2': BaseflowType.NONLINEAR,
        'tens2pll_2': BaseflowType.PARALLEL_LINEAR,
    },
    'QSURF': {
        'arno_x_vic': SurfaceRunoffType.UZ_PARETO,
        'prms_varnt': SurfaceRunoffType.UZ_LINEAR,
        'tmdl_param': SurfaceRunoffType.LZ_GAMMA,
    },
    'QPERC': {
        'perc_f2sat': PercolationType.FREE_STORAGE,
        'perc_w2sat': PercolationType.TOTAL_STORAGE,
        'perc_lower': PercolationType.LOWER_DEMAND,
    },
    'ESOIL': {
        'sequential': EvaporationType.SEQUENTIAL,
        'rootweight': EvaporationType.ROOT_WEIGHT,
    },
    'QINTF': {
        'intflwnone': InterflowType.NONE,
        'intflwsome': InterflowType.LINEAR,
    },
    'Q_TDH': {
        'rout_gamma': RoutingType.GAMMA,
        'no_routing': RoutingType.NONE,
    },
    'SNOWM': {
        'temp_index': SnowType.TEMP_INDEX,
        'no_snowmod': SnowType.NONE,
    },
    'RFERR': {
        'additive_e': RainfallErrorType.ADDITIVE,
        'multiplc_e': RainfallErrorType.MULTIPLICATIVE,
    },
}

# Map config names to predefined configs.
JFUSE_CONFIGS = {
    'prms': PRMS_CONFIG,
    'prms_gradient': PRMS_GRADIENT_CONFIG,       # Optimized for gradient-based calibration
    'max_gradient': MAX_GRADIENT_CONFIG,         # Maximum parameter coverage
    'sacramento': SACRAMENTO_CONFIG,
    'topmodel': TOPMODEL_CONFIG,
    'vic': VIC_CONFIG,
}


def build_config_from_decisions(decisions: Dict[str, Any]) -> ModelConfig:
    """Build a jFUSE ``ModelConfig`` from Fortran FUSE decision options.

    Args:
        decisions: Mapping of FUSE decision keys (e.g. ``ARCH1``, ``QSURF``) to
            option names. Values may be plain strings or single-element lists.

    Returns:
        A ``ModelConfig`` assembled from :data:`FUSE_DECISION_MAP`, falling back
        to sensible defaults for any missing or unrecognized decision.
    """
    resolved = {}
    for key, value in decisions.items():
        if isinstance(value, list):
            resolved[key] = value[0] if value else None
        else:
            resolved[key] = value

    arch2_val = resolved.get('ARCH2', 'fixedsiz_2')

    config_kwargs = {
        'upper_arch': FUSE_DECISION_MAP['ARCH1'].get(
            resolved.get('ARCH1', 'tension1_1'), UpperLayerArch.TENSION_FREE),
        'lower_arch': FUSE_DECISION_MAP['ARCH2'].get(
            arch2_val, LowerLayerArch.SINGLE_NOEVAP),
        'baseflow': FUSE_DECISION_MAP['ARCH2_BASEFLOW'].get(
            arch2_val, BaseflowType.LINEAR),
        'percolation': FUSE_DECISION_MAP['QPERC'].get(
            resolved.get('QPERC', 'perc_f2sat'), PercolationType.FREE_STORAGE),
        'surface_runoff': FUSE_DECISION_MAP['QSURF'].get(
            resolved.get('QSURF', 'arno_x_vic'), SurfaceRunoffType.UZ_PARETO),
        'evaporation': FUSE_DECISION_MAP['ESOIL'].get(
            resolved.get('ESOIL', 'rootweight'), EvaporationType.ROOT_WEIGHT),
        'interflow': FUSE_DECISION_MAP['QINTF'].get(
            resolved.get('QINTF', 'intflwnone'), InterflowType.NONE),
        'snow': FUSE_DECISION_MAP['SNOWM'].get(
            resolved.get('SNOWM', 'temp_index'), SnowType.TEMP_INDEX),
        'routing': FUSE_DECISION_MAP['Q_TDH'].get(
            resolved.get('Q_TDH', 'rout_gamma'), RoutingType.GAMMA),
        'rainfall_error': FUSE_DECISION_MAP['RFERR'].get(
            resolved.get('RFERR', 'multiplc_e'), RainfallErrorType.MULTIPLICATIVE),
    }

    return ModelConfig(**config_kwargs)
