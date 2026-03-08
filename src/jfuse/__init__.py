"""
jFUSE: JAX-based Differentiable Hydrological Modeling Framework

A pure JAX implementation of the FUSE (Framework for Understanding Structural Errors)
hydrological model with differentiable river routing for gradient-based calibration.

Features:
- Full automatic differentiation through both rainfall-runoff and routing
- JIT compilation for high performance
- GPU acceleration via JAX
- Multiple model architectures from Clark et al. (2008)
- Muskingum-Cunge and other routing methods
- Support for coupled gradient-based optimization

Example:
    >>> import jfuse
    >>> model = jfuse.CoupledModel.from_netcdf("data/forcing.nc", "data/network.nc")
    >>> params = model.default_params()
    >>> loss, grads = jfuse.value_and_grad_loss(model, params, observed)
"""

# JAX precision configuration
# IMPORTANT: Keep x64 disabled for compatibility with neuralgcm coupling
# neuralgcm requires float32 for pretrained checkpoints to work correctly
import jax
jax.config.update("jax_enable_x64", False)

__version__ = "0.1.0"
__author__ = "Darri Eythorsson"

# Core types and configuration
from jfuse.fuse.config import (
    ModelConfig,
    UpperLayerArch,
    LowerLayerArch,
    BaseflowType,
    PercolationType,
    SurfaceRunoffType,
    EvaporationType,
    InterflowType,
    PRMS_CONFIG,
    SACRAMENTO_CONFIG,
    TOPMODEL_CONFIG,
    VIC_CONFIG,
)

# State and parameter structures
from jfuse.fuse.state import (
    State,
    Flux,
    Parameters,
    Forcing,
    PARAM_BOUNDS,
    PARAM_NAMES,
)

# FUSE model
from jfuse.fuse.model import (
    FUSEModel,
    create_fuse_model,
    fuse_step,
    fuse_simulate,
)

# Routing
from jfuse.routing.network import (
    RiverNetwork,
    Reach,
    create_network_from_topology,
)

from jfuse.routing.router import (
    MuskingumCungeRouter,
    route_network,
)

# Coupled model
from jfuse.coupled import (
    CoupledModel,
    coupled_simulate,
    coupled_loss,
    value_and_grad_loss,
    nse_loss,
    kge_loss,
    mse_loss,
    rmse_loss,
    mae_loss,
)

# I/O utilities
from jfuse.io.netcdf import (
    load_forcing,
    load_network,
    save_results,
)

# Optimization utilities
from jfuse.optim.calibration import (
    Calibrator,
    CalibrationConfig,
)

# Convenience function for quick model setup
def quick_setup(
    forcing_path: str,
    network_path: str,
    config: ModelConfig = None,
) -> CoupledModel:
    """
    Quick setup of a coupled FUSE + routing model from NetCDF files.
    
    Args:
        forcing_path: Path to forcing NetCDF file
        network_path: Path to network topology NetCDF file
        config: Model configuration (defaults to PRMS)
        
    Returns:
        CoupledModel ready for simulation and calibration
    """
    return CoupledModel.from_netcdf(forcing_path, network_path, config)


def register():
    """Register jFUSE components with SYMFLUENCE's model registry.

    Called automatically by SYMFLUENCE's plugin discovery system via the
    ``symfluence.plugins`` entry point, or manually with::

        import jfuse
        jfuse.register()
    """
    from symfluence.core.registry import model_manifest
    from symfluence.models.registry import ModelRegistry
    from symfluence.optimization.registry import OptimizerRegistry

    from jfuse.sfconfig import JFUSEConfigAdapter
    from jfuse.extractor import JFUSEResultExtractor
    from jfuse.runner import JFUSERunner
    from jfuse.preprocessor import JFUSEPreProcessor
    from jfuse.postprocessor import JFUSEPostprocessor, JFUSERoutedPostprocessor
    from jfuse.calibration.worker import JFUSEWorker
    from jfuse.calibration.parameter_manager import JFUSEParameterManager
    from jfuse.calibration.optimizer import JFUSEModelOptimizer

    # Register via unified manifest
    model_manifest(
        "JFUSE",
        config_adapter=JFUSEConfigAdapter,
        result_extractor=JFUSEResultExtractor,
    )

    # Register individual components
    ModelRegistry.register_runner('JFUSE', method_name='run_jfuse')(JFUSERunner)
    ModelRegistry.register_preprocessor('JFUSE')(JFUSEPreProcessor)
    ModelRegistry.register_postprocessor('JFUSE')(JFUSEPostprocessor)
    ModelRegistry.register_postprocessor('JFUSE_routed')(JFUSERoutedPostprocessor)

    # Register calibration components
    OptimizerRegistry.register_worker('JFUSE')(JFUSEWorker)
    OptimizerRegistry.register_parameter_manager('JFUSE')(JFUSEParameterManager)
    OptimizerRegistry.register_optimizer('JFUSE')(JFUSEModelOptimizer)


__all__ = [
    # Version
    "__version__",
    # Config
    "ModelConfig",
    "UpperLayerArch",
    "LowerLayerArch",
    "BaseflowType",
    "PercolationType",
    "SurfaceRunoffType",
    "EvaporationType",
    "InterflowType",
    "PRMS_CONFIG",
    "SACRAMENTO_CONFIG",
    "TOPMODEL_CONFIG",
    "VIC_CONFIG",
    # State
    "State",
    "Flux",
    "Parameters",
    "Forcing",
    "PARAM_BOUNDS",
    "PARAM_NAMES",
    # FUSE
    "FUSEModel",
    "create_fuse_model",
    "fuse_step",
    "fuse_simulate",
    # Routing
    "RiverNetwork",
    "Reach",
    "create_network_from_topology",
    "MuskingumCungeRouter",
    "route_network",
    # Coupled
    "CoupledModel",
    "coupled_simulate",
    "coupled_loss",
    "value_and_grad_loss",
    # I/O
    "load_forcing",
    "load_network",
    "save_results",
    # Optimization
    "Calibrator",
    "CalibrationConfig",
    # Loss functions
    "nse_loss",
    "kge_loss",
    "mse_loss",
    "rmse_loss",
    "mae_loss",
    # Quick setup
    "quick_setup",
    # SYMFLUENCE integration
    "register",
]
