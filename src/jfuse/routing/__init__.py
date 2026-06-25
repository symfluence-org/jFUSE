"""
River Routing subpackage.

Implements differentiable river routing algorithms for channel flow simulation,
with the network represented as a Directed Acyclic Graph (DAG) for efficient
topological processing.
"""

from .network import (
    Reach,
    RiverNetwork,
    NetworkArrays,
    create_network_from_topology,
)

from .router import (
    MuskingumCungeRouter,
    MuskingumParams,
    RouterState,
    route_network,
    route_network_full,
    route_network_step,
    lake_outflow,
    compute_muskingum_params,
    muskingum_route_reach,
    compute_channel_geometry,
    compute_velocity_manning,
    compute_celerity,
)

__all__ = [
    # Network
    "Reach",
    "RiverNetwork",
    "NetworkArrays",
    "create_network_from_topology",
    # Router
    "MuskingumCungeRouter",
    "MuskingumParams",
    "RouterState",
    "route_network",
    "route_network_full",
    "route_network_step",
    "lake_outflow",
    "compute_muskingum_params",
    "muskingum_route_reach",
    "compute_channel_geometry",
    "compute_velocity_manning",
    "compute_celerity",
]
