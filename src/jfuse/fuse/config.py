"""
FUSE Model Configuration

Defines the modeling decisions available in FUSE based on Clark et al. (2008).
Each decision point represents a different physical process formulation that
can be combined to create hybrid model structures.

This module supports reading FUSE decision files directly, allowing access to
all 79+ model structure combinations possible in FUSE.

References:
    Clark, M. P., et al. (2008). Framework for Understanding Structural Errors
    (FUSE): A modular framework to diagnose differences between hydrological
    models. Water Resources Research, 44, W00B02.
"""

from enum import IntEnum
from typing import NamedTuple, Dict
from pathlib import Path
import jax.numpy as jnp


class UpperLayerArch(IntEnum):
    """Upper layer (unsaturated zone) architecture.

    Controls the state variable structure for the upper soil layer.
    See Clark et al. (2008) equations (1a), (1b), (1c).

    FUSE decision file keys: ARCH1
        onestate_1  -> SINGLE_STATE
        tension1_1  -> TENSION_FREE
        tension2_1  -> TENSION2_FREE
    """

    SINGLE_STATE = 0  # Single state S1 (TOPMODEL, ARNO/VIC style) - Eq 1a
    TENSION_FREE = 1  # Separate tension (S1_T) and free (S1_F) storage (Sacramento) - Eq 1b
    TENSION2_FREE = 2  # Two tension stores + free storage (PRMS) - Eq 1c


class LowerLayerArch(IntEnum):
    """Lower layer (saturated zone) architecture.

    Controls the state variable structure and baseflow mechanism.
    See Clark et al. (2008) equations (2a), (2b), (2c).

    FUSE decision file keys: ARCH2
        unlimfrc_2  -> SINGLE_NOEVAP (unlimited, no primary/secondary)
        unlimpow_2  -> SINGLE_NOEVAP (unlimited, power recession)
        fixedsiz_2  -> SINGLE_EVAP (fixed size lower layer)
        tens2pll_2  -> TENSION_2RESERV (tension + two parallel)
    """

    SINGLE_NOEVAP = 0  # Single reservoir, infinite, no evaporation (TOPMODEL/PRMS) - Eq 2a
    SINGLE_EVAP = 1  # Single reservoir with evaporation, fixed capacity (ARNO/VIC) - Eq 2b
    TENSION_2RESERV = 2  # Tension + two parallel reservoirs (Sacramento) - Eq 2c


class BaseflowType(IntEnum):
    """Baseflow parameterization.

    Different functional forms for computing baseflow from lower layer storage.
    See Clark et al. (2008) equations (6a)-(6d).

    FUSE decision file keys: implicitly from ARCH2
        unlimfrc_2  -> LINEAR
        unlimpow_2  -> TOPMODEL
        fixedsiz_2  -> NONLINEAR
        tens2pll_2  -> PARALLEL_LINEAR
    """

    LINEAR = 0  # Single linear reservoir: qb = v*S2 (PRMS) - Eq 6a
    PARALLEL_LINEAR = 1  # Two parallel linear reservoirs (Sacramento) - Eq 6b
    NONLINEAR = 2  # Nonlinear power function (ARNO/VIC) - Eq 6c
    TOPMODEL = 3  # TOPMODEL power law transmissivity - Eq 6d


class PercolationType(IntEnum):
    """Percolation parameterization.

    Controls vertical water movement from upper to lower layer.
    See Clark et al. (2008) equations (4a)-(4c).

    FUSE decision file keys: QPERC
        perc_f2sat  -> TOTAL_STORAGE (fraction to saturation)
        perc_lower  -> LOWER_DEMAND (lower layer control)
        perc_w2sat  -> FREE_STORAGE (water to saturation)
    """

    TOTAL_STORAGE = 0  # Based on total upper zone storage (VIC style) - Eq 4a
    FREE_STORAGE = 1  # Based on free storage above field capacity (PRMS) - Eq 4b
    LOWER_DEMAND = 2  # Driven by lower zone demand (Sacramento) - Eq 4c


class SurfaceRunoffType(IntEnum):
    """Surface runoff (saturated area) parameterization.

    Controls how saturated contributing area is computed.
    See Clark et al. (2008) equations (9a)-(9c).

    FUSE decision file keys: QSURF
        prms_varnt  -> UZ_LINEAR
        arno_x_vic  -> UZ_PARETO
        tmdl_param  -> LZ_GAMMA
    """

    UZ_LINEAR = 0  # Linear function of tension storage (PRMS) - Eq 9a
    UZ_PARETO = 1  # Pareto distribution / VIC 'b' curve (ARNO/VIC) - Eq 9b
    LZ_GAMMA = 2  # TOPMODEL topographic index distribution - Eq 9c


class EvaporationType(IntEnum):
    """Evaporation parameterization.

    Controls partitioning of evaporative demand between soil layers.
    See Clark et al. (2008) equations (3a)-(3d).

    FUSE decision file keys: ESOIL
        sequential  -> SEQUENTIAL
        rootweight  -> ROOT_WEIGHT
    """

    SEQUENTIAL = 0  # Sequential: upper layer first, then lower - Eq 3a,3b
    ROOT_WEIGHT = 1  # Root weighting between layers - Eq 3c,3d


class InterflowType(IntEnum):
    """Interflow parameterization.

    Controls lateral subsurface flow from upper layer.
    See Clark et al. (2008) equations (5a)-(5b).

    FUSE decision file keys: QINTF
        intflwnone  -> NONE
        intflwsome  -> LINEAR
    """

    NONE = 0  # No interflow - Eq 5a
    LINEAR = 1  # Linear function of free storage - Eq 5b


class SnowType(IntEnum):
    """Snow model type.

    FUSE decision file keys: SNOWM
        no_snowmod  -> NONE
        temp_index  -> TEMP_INDEX
    """

    NONE = 0  # No snow module
    TEMP_INDEX = 1  # Temperature index (degree-day) snow model


class RoutingType(IntEnum):
    """Time delay histogram / routing type.

    FUSE decision file keys: Q_TDH
        no_routing  -> NONE
        rout_gamma  -> GAMMA
    """

    NONE = 0  # No routing / time delay
    GAMMA = 1  # Gamma distribution routing


class RainfallErrorType(IntEnum):
    """Rainfall error model type.

    FUSE decision file keys: RFERR
        additive_e  -> ADDITIVE
        multiplica  -> MULTIPLICATIVE
    """

    ADDITIVE = 0  # Additive rainfall error
    MULTIPLICATIVE = 1  # Multiplicative rainfall error


class ModelConfig(NamedTuple):
    """Complete model configuration specifying all physics options.

    This structure encapsulates all modeling decisions and can be used
    to select different model structures at runtime.

    Attributes:
        upper_arch: Upper layer architecture
        lower_arch: Lower layer architecture
        baseflow: Baseflow parameterization
        percolation: Percolation parameterization
        surface_runoff: Surface runoff parameterization
        evaporation: Evaporation parameterization
        interflow: Interflow parameterization
        snow: Snow model type
        routing: Routing type
        rainfall_error: Rainfall error model
    """

    upper_arch: UpperLayerArch = UpperLayerArch.SINGLE_STATE
    lower_arch: LowerLayerArch = LowerLayerArch.SINGLE_NOEVAP
    baseflow: BaseflowType = BaseflowType.LINEAR
    percolation: PercolationType = PercolationType.TOTAL_STORAGE
    surface_runoff: SurfaceRunoffType = SurfaceRunoffType.UZ_LINEAR
    evaporation: EvaporationType = EvaporationType.SEQUENTIAL
    interflow: InterflowType = InterflowType.NONE
    snow: SnowType = SnowType.TEMP_INDEX
    routing: RoutingType = RoutingType.NONE
    rainfall_error: RainfallErrorType = RainfallErrorType.ADDITIVE
    enable_glacier: bool = False

    @property
    def enable_snow(self) -> bool:
        """Whether snow module is enabled."""
        return self.snow != SnowType.NONE

    @property
    def num_upper_states(self) -> int:
        """Number of state variables in upper layer."""
        if self.upper_arch == UpperLayerArch.SINGLE_STATE:
            return 1
        elif self.upper_arch == UpperLayerArch.TENSION_FREE:
            return 2
        else:  # TENSION2_FREE
            return 3

    @property
    def num_lower_states(self) -> int:
        """Number of state variables in lower layer."""
        if self.lower_arch in (LowerLayerArch.SINGLE_NOEVAP, LowerLayerArch.SINGLE_EVAP):
            return 1
        else:  # TENSION_2RESERV
            return 3

    @property
    def num_states(self) -> int:
        """Total number of state variables."""
        n = self.num_upper_states + self.num_lower_states
        if self.enable_snow:
            n += 1
        return n

    @property
    def has_lower_evap(self) -> bool:
        """Whether lower layer has evaporation."""
        return self.lower_arch in (LowerLayerArch.SINGLE_EVAP, LowerLayerArch.TENSION_2RESERV)

    @property
    def has_interflow(self) -> bool:
        """Whether interflow is enabled."""
        return self.interflow == InterflowType.LINEAR

    def to_indices(self) -> jnp.ndarray:
        """Convert config to integer indices for use in jit-compiled functions."""
        return jnp.array(
            [
                int(self.upper_arch),
                int(self.lower_arch),
                int(self.baseflow),
                int(self.percolation),
                int(self.surface_runoff),
                int(self.evaporation),
                int(self.interflow),
                int(self.snow),
                int(self.routing),
                int(self.rainfall_error),
            ],
            dtype=jnp.int32,
        )

    def describe(self) -> str:
        """Return human-readable description of configuration."""
        lines = [
            "FUSE Model Configuration:",
            f"  Upper layer architecture: {self.upper_arch.name}",
            f"  Lower layer architecture: {self.lower_arch.name}",
            f"  Baseflow type: {self.baseflow.name}",
            f"  Percolation type: {self.percolation.name}",
            f"  Surface runoff type: {self.surface_runoff.name}",
            f"  Evaporation type: {self.evaporation.name}",
            f"  Interflow type: {self.interflow.name}",
            f"  Snow model: {self.snow.name}",
            f"  Routing: {self.routing.name}",
            f"  Rainfall error: {self.rainfall_error.name}",
            f"  Number of states: {self.num_states}",
        ]
        return "\n".join(lines)


# =============================================================================
# DECISION FILE MAPPINGS
# =============================================================================

# Map FUSE decision file strings to enum values
ARCH1_MAP = {
    "onestate_1": UpperLayerArch.SINGLE_STATE,
    "tension1_1": UpperLayerArch.TENSION_FREE,
    "tension2_1": UpperLayerArch.TENSION2_FREE,
}

# ARCH2 determines both lower layer arch and baseflow type
ARCH2_MAP = {
    "unlimfrc_2": (LowerLayerArch.SINGLE_NOEVAP, BaseflowType.LINEAR),
    "unlimpow_2": (LowerLayerArch.SINGLE_NOEVAP, BaseflowType.TOPMODEL),
    "fixedsiz_2": (LowerLayerArch.SINGLE_EVAP, BaseflowType.NONLINEAR),
    "tens2pll_2": (LowerLayerArch.TENSION_2RESERV, BaseflowType.PARALLEL_LINEAR),
}

QSURF_MAP = {
    "prms_varnt": SurfaceRunoffType.UZ_LINEAR,
    "arno_x_vic": SurfaceRunoffType.UZ_PARETO,
    "tmdl_param": SurfaceRunoffType.LZ_GAMMA,
}

QPERC_MAP = {
    "perc_f2sat": PercolationType.TOTAL_STORAGE,
    "perc_lower": PercolationType.LOWER_DEMAND,
    "perc_w2sat": PercolationType.FREE_STORAGE,
}

ESOIL_MAP = {
    "sequential": EvaporationType.SEQUENTIAL,
    "rootweight": EvaporationType.ROOT_WEIGHT,
}

QINTF_MAP = {
    "intflwnone": InterflowType.NONE,
    "intflwsome": InterflowType.LINEAR,
}

SNOWM_MAP = {
    "no_snowmod": SnowType.NONE,
    "temp_index": SnowType.TEMP_INDEX,
}

Q_TDH_MAP = {
    "no_routing": RoutingType.NONE,
    "rout_gamma": RoutingType.GAMMA,
}

RFERR_MAP = {
    "additive_e": RainfallErrorType.ADDITIVE,
    "multiplica": RainfallErrorType.MULTIPLICATIVE,
}


def parse_decisions_file(filepath: str) -> Dict[str, str]:
    """Parse a FUSE decisions file.

    Supports both:
    1. jFUSE format: KEY VALUE  (e.g., ARCH1 tension1_1)
    2. FUSE Fortran: VALUE KEY  (e.g., tension1_1 ARCH1)
    """
    decisions = {}
    path = Path(filepath)

    # List of known configuration keys
    KNOWN_KEYS = {"ARCH1", "ARCH2", "QSURF", "QPERC", "ESOIL", "QINTF", "Q_TDH", "SNOWM", "RFERR"}

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            # Skip empty lines, comments, and separator lines (dashes)
            if not line or line.startswith("!") or line.startswith("-"):
                continue

            # Remove inline comments
            if "!" in line:
                line = line[: line.index("!")].strip()

            # Parse decision
            parts = line.split()
            if len(parts) >= 2:
                token0 = parts[0].upper()
                token1 = parts[1].upper()

                # Check logic: Is the first token a Key? (jFUSE style)
                if token0 in KNOWN_KEYS:
                    key = token0
                    val = parts[1].lower()
                # Is the second token a Key? (Fortran FUSE style)
                elif token1 in KNOWN_KEYS:
                    key = token1
                    val = parts[0].lower()
                else:
                    # Fallback or skip
                    continue

                decisions[key] = val

    return decisions


def config_from_decisions(decisions: Dict[str, str]) -> ModelConfig:
    """Create a ModelConfig from parsed FUSE decisions.

    Args:
        decisions: Dictionary from parse_decisions_file()

    Returns:
        ModelConfig with the specified decisions
    """
    # Get upper layer architecture
    arch1_val = decisions.get("ARCH1", "onestate_1")
    upper_arch = ARCH1_MAP.get(arch1_val, UpperLayerArch.SINGLE_STATE)

    # Get lower layer architecture and baseflow type
    arch2_val = decisions.get("ARCH2", "unlimfrc_2")
    lower_arch, baseflow = ARCH2_MAP.get(
        arch2_val, (LowerLayerArch.SINGLE_NOEVAP, BaseflowType.LINEAR)
    )

    # Get other decisions
    qsurf_val = decisions.get("QSURF", "arno_x_vic")
    surface_runoff = QSURF_MAP.get(qsurf_val, SurfaceRunoffType.UZ_PARETO)

    qperc_val = decisions.get("QPERC", "perc_f2sat")
    percolation = QPERC_MAP.get(qperc_val, PercolationType.TOTAL_STORAGE)

    esoil_val = decisions.get("ESOIL", "sequential")
    evaporation = ESOIL_MAP.get(esoil_val, EvaporationType.SEQUENTIAL)

    qintf_val = decisions.get("QINTF", "intflwnone")
    interflow = QINTF_MAP.get(qintf_val, InterflowType.NONE)

    snowm_val = decisions.get("SNOWM", "temp_index")
    snow = SNOWM_MAP.get(snowm_val, SnowType.TEMP_INDEX)

    qtdh_val = decisions.get("Q_TDH", "no_routing")
    routing = Q_TDH_MAP.get(qtdh_val, RoutingType.NONE)

    rferr_val = decisions.get("RFERR", "additive_e")
    rainfall_error = RFERR_MAP.get(rferr_val, RainfallErrorType.ADDITIVE)

    return ModelConfig(
        upper_arch=upper_arch,
        lower_arch=lower_arch,
        baseflow=baseflow,
        percolation=percolation,
        surface_runoff=surface_runoff,
        evaporation=evaporation,
        interflow=interflow,
        snow=snow,
        routing=routing,
        rainfall_error=rainfall_error,
    )


def load_decisions_file(filepath: str) -> ModelConfig:
    """Load a FUSE decisions file and return a ModelConfig.

    This is the main entry point for reading FUSE decision files.

    Args:
        filepath: Path to the FUSE decisions file

    Returns:
        ModelConfig configured according to the decisions file

    Example:
        >>> config = load_decisions_file("fuse_zDecisions.txt")
        >>> model = FUSEModel(config)
    """
    decisions = parse_decisions_file(filepath)
    return config_from_decisions(decisions)


def write_decisions_file(config: ModelConfig, filepath: str):
    """Write a ModelConfig to a FUSE decisions file.

    Args:
        config: Model configuration
        filepath: Output path for the decisions file
    """
    # Reverse mappings
    arch1_rev = {v: k for k, v in ARCH1_MAP.items()}
    arch2_rev = {v: k for k, v in ARCH2_MAP.items()}
    qsurf_rev = {v: k for k, v in QSURF_MAP.items()}
    qperc_rev = {v: k for k, v in QPERC_MAP.items()}
    esoil_rev = {v: k for k, v in ESOIL_MAP.items()}
    qintf_rev = {v: k for k, v in QINTF_MAP.items()}
    snowm_rev = {v: k for k, v in SNOWM_MAP.items()}
    qtdh_rev = {v: k for k, v in Q_TDH_MAP.items()}
    rferr_rev = {v: k for k, v in RFERR_MAP.items()}

    with open(filepath, "w") as f:
        f.write("! FUSE model decisions\n")
        f.write("! Generated by jFUSE\n")
        f.write("! " + "-" * 40 + "\n")
        f.write(f"RFERR    {rferr_rev.get(config.rainfall_error, 'additive_e')}\n")
        f.write(f"ARCH1    {arch1_rev.get(config.upper_arch, 'onestate_1')}\n")
        f.write(f"ARCH2    {arch2_rev.get((config.lower_arch, config.baseflow), 'unlimfrc_2')}\n")
        f.write(f"QSURF    {qsurf_rev.get(config.surface_runoff, 'arno_x_vic')}\n")
        f.write(f"QPERC    {qperc_rev.get(config.percolation, 'perc_f2sat')}\n")
        f.write(f"ESOIL    {esoil_rev.get(config.evaporation, 'sequential')}\n")
        f.write(f"QINTF    {qintf_rev.get(config.interflow, 'intflwnone')}\n")
        f.write(f"Q_TDH    {qtdh_rev.get(config.routing, 'no_routing')}\n")
        f.write(f"SNOWM    {snowm_rev.get(config.snow, 'temp_index')}\n")


def enumerate_all_configs() -> Dict[str, ModelConfig]:
    """Generate all valid FUSE model structure combinations.

    Returns:
        Dictionary mapping structure IDs to ModelConfigs

    Note:
        This generates 3 * 4 * 3 * 3 * 2 * 2 = 432 combinations,
        though not all are physically meaningful.
    """
    configs = {}

    for arch1 in UpperLayerArch:
        for arch2_key, (lower_arch, baseflow) in ARCH2_MAP.items():
            for qsurf in SurfaceRunoffType:
                for qperc in PercolationType:
                    for esoil in EvaporationType:
                        for qintf in InterflowType:
                            # Create config ID
                            config_id = f"{arch1.name}_{lower_arch.name}_{baseflow.name}_{qsurf.name}_{qperc.name}_{esoil.name}_{qintf.name}"

                            config = ModelConfig(
                                upper_arch=arch1,
                                lower_arch=lower_arch,
                                baseflow=baseflow,
                                percolation=qperc,
                                surface_runoff=qsurf,
                                evaporation=esoil,
                                interflow=qintf,
                                snow=SnowType.TEMP_INDEX,
                                routing=RoutingType.NONE,
                                rainfall_error=RainfallErrorType.ADDITIVE,
                            )
                            configs[config_id] = config

    return configs


# =============================================================================
# PREDEFINED MODEL CONFIGURATIONS (Parent models from Clark et al. 2008)
# =============================================================================

PRMS_CONFIG = ModelConfig(
    upper_arch=UpperLayerArch.TENSION2_FREE,
    lower_arch=LowerLayerArch.SINGLE_NOEVAP,
    baseflow=BaseflowType.LINEAR,
    percolation=PercolationType.FREE_STORAGE,
    surface_runoff=SurfaceRunoffType.UZ_LINEAR,
    evaporation=EvaporationType.SEQUENTIAL,
    interflow=InterflowType.LINEAR,
    snow=SnowType.TEMP_INDEX,
)

SACRAMENTO_CONFIG = ModelConfig(
    upper_arch=UpperLayerArch.TENSION_FREE,
    lower_arch=LowerLayerArch.TENSION_2RESERV,
    baseflow=BaseflowType.PARALLEL_LINEAR,
    percolation=PercolationType.LOWER_DEMAND,
    surface_runoff=SurfaceRunoffType.UZ_LINEAR,
    evaporation=EvaporationType.SEQUENTIAL,
    interflow=InterflowType.NONE,
    snow=SnowType.TEMP_INDEX,
)

TOPMODEL_CONFIG = ModelConfig(
    upper_arch=UpperLayerArch.SINGLE_STATE,
    lower_arch=LowerLayerArch.SINGLE_NOEVAP,
    baseflow=BaseflowType.TOPMODEL,
    percolation=PercolationType.TOTAL_STORAGE,
    surface_runoff=SurfaceRunoffType.LZ_GAMMA,
    evaporation=EvaporationType.SEQUENTIAL,
    interflow=InterflowType.NONE,
    snow=SnowType.TEMP_INDEX,
)

VIC_CONFIG = ModelConfig(
    upper_arch=UpperLayerArch.SINGLE_STATE,
    lower_arch=LowerLayerArch.SINGLE_EVAP,
    baseflow=BaseflowType.NONLINEAR,
    percolation=PercolationType.TOTAL_STORAGE,
    surface_runoff=SurfaceRunoffType.UZ_PARETO,
    evaporation=EvaporationType.ROOT_WEIGHT,
    interflow=InterflowType.NONE,
    snow=SnowType.TEMP_INDEX,
)

# Dictionary mapping names to configs
NAMED_CONFIGS = {
    "prms": PRMS_CONFIG,
    "sacramento": SACRAMENTO_CONFIG,
    "topmodel": TOPMODEL_CONFIG,
    "vic": VIC_CONFIG,
}


def get_config(name: str) -> ModelConfig:
    """Get a predefined model configuration by name.

    Args:
        name: One of 'prms', 'sacramento', 'topmodel', 'vic'

    Returns:
        ModelConfig for the specified model

    Raises:
        KeyError: If name is not recognized
    """
    return NAMED_CONFIGS[name.lower()]


class FUSEConfig:
    """Convenience class for accessing FUSE model configurations.

    Supports both predefined configurations and loading from FUSE decision files.

    Example:
        >>> # Use predefined config
        >>> config = FUSEConfig.prms()
        >>> model = FUSEModel(config)

        >>> # Load from FUSE decision file
        >>> config = FUSEConfig.from_file("fuse_zDecisions.txt")
        >>> model = FUSEModel(config)

        >>> # Create custom config
        >>> config = FUSEConfig.custom(
        ...     upper_arch=UpperLayerArch.TENSION_FREE,
        ...     baseflow=BaseflowType.TOPMODEL,
        ... )
    """

    @staticmethod
    def prms() -> ModelConfig:
        """Get PRMS-like model configuration."""
        return PRMS_CONFIG

    @staticmethod
    def sacramento() -> ModelConfig:
        """Get Sacramento-like model configuration."""
        return SACRAMENTO_CONFIG

    @staticmethod
    def topmodel() -> ModelConfig:
        """Get TOPMODEL-like model configuration."""
        return TOPMODEL_CONFIG

    @staticmethod
    def vic() -> ModelConfig:
        """Get VIC-like model configuration."""
        return VIC_CONFIG

    @staticmethod
    def from_file(filepath: str) -> ModelConfig:
        """Load configuration from a FUSE decisions file.

        Args:
            filepath: Path to the FUSE decisions file (e.g., fuse_zDecisions.txt)

        Returns:
            ModelConfig based on the decisions file
        """
        return load_decisions_file(filepath)

    @staticmethod
    def from_decisions(decisions: Dict[str, str]) -> ModelConfig:
        """Create configuration from a dictionary of decisions.

        Args:
            decisions: Dictionary with keys like 'ARCH1', 'ARCH2', 'QSURF', etc.

        Returns:
            ModelConfig based on the decisions

        Example:
            >>> config = FUSEConfig.from_decisions({
            ...     'ARCH1': 'tension1_1',
            ...     'ARCH2': 'tens2pll_2',
            ...     'QSURF': 'prms_varnt',
            ...     'QPERC': 'perc_lower',
            ... })
        """
        return config_from_decisions(decisions)

    @staticmethod
    def custom(
        upper_arch: UpperLayerArch = UpperLayerArch.SINGLE_STATE,
        lower_arch: LowerLayerArch = LowerLayerArch.SINGLE_NOEVAP,
        baseflow: BaseflowType = BaseflowType.LINEAR,
        percolation: PercolationType = PercolationType.TOTAL_STORAGE,
        surface_runoff: SurfaceRunoffType = SurfaceRunoffType.UZ_LINEAR,
        evaporation: EvaporationType = EvaporationType.SEQUENTIAL,
        interflow: InterflowType = InterflowType.NONE,
        snow: SnowType = SnowType.TEMP_INDEX,
        routing: RoutingType = RoutingType.NONE,
        rainfall_error: RainfallErrorType = RainfallErrorType.ADDITIVE,
    ) -> ModelConfig:
        """Create a custom model configuration.

        Args:
            upper_arch: Upper layer architecture
            lower_arch: Lower layer architecture
            baseflow: Baseflow parameterization
            percolation: Percolation parameterization
            surface_runoff: Surface runoff parameterization
            evaporation: Evaporation parameterization
            interflow: Interflow parameterization
            snow: Snow model type
            routing: Routing type
            rainfall_error: Rainfall error model

        Returns:
            Custom ModelConfig
        """
        return ModelConfig(
            upper_arch=upper_arch,
            lower_arch=lower_arch,
            baseflow=baseflow,
            percolation=percolation,
            surface_runoff=surface_runoff,
            evaporation=evaporation,
            interflow=interflow,
            snow=snow,
            routing=routing,
            rainfall_error=rainfall_error,
        )

    @staticmethod
    def all_structures() -> Dict[str, ModelConfig]:
        """Get all possible model structure combinations.

        Returns:
            Dictionary mapping structure IDs to ModelConfigs
        """
        return enumerate_all_configs()

    @staticmethod
    def save(config: ModelConfig, filepath: str):
        """Save configuration to a FUSE decisions file.

        Args:
            config: Model configuration
            filepath: Output path for the decisions file
        """
        write_decisions_file(config, filepath)
