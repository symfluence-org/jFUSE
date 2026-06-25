"""
River Network Representation

Defines the river network topology as a Directed Acyclic Graph (DAG) with
efficient traversal using topological ordering. This enables parallelization
of independent reaches and proper sequencing of dependent reaches.

The network structure supports:
- Topological sorting for upstream-to-downstream routing
- Parallel processing of independent branches
- Efficient adjacency representation for JAX operations
"""

from typing import List, Dict, Optional, NamedTuple, Sequence
from dataclasses import dataclass
import jax.numpy as jnp
from jax import Array
import numpy as np


@dataclass
class Reach:
    """A single river reach with routing parameters.

    Attributes:
        id: Unique identifier
        length: Reach length in meters
        slope: Bed slope (m/m)
        manning_n: Manning's roughness coefficient
        width_coef: Channel width coefficient (W = coef * Q^exp)
        width_exp: Channel width exponent
        depth_coef: Channel depth coefficient
        depth_exp: Channel depth exponent
        upstream_ids: List of upstream reach IDs
        downstream_id: Downstream reach ID (-1 for outlet)
        hru_id: Associated HRU ID for lateral inflow
        area: Contributing catchment area (m²)
    """

    id: int
    length: float
    slope: float
    manning_n: float = 0.035
    width_coef: float = 7.2
    width_exp: float = 0.5
    depth_coef: float = 0.27
    depth_exp: float = 0.3
    upstream_ids: List[int] = None
    downstream_id: int = -1
    hru_id: int = -1
    area: float = 0.0
    # --- Lake / reservoir node (optional) ---
    # When ``is_lake`` is True the reach is routed as a storage node
    # (level-pool / regulated outflow) instead of Muskingum-Cunge channel
    # routing. Defaults describe "no lake" so existing reaches are unaffected.
    is_lake: bool = False
    lake_s_max: float = 0.0  # Active storage capacity (m³)
    lake_q_ref: float = 0.0  # Reference outflow at full storage (m³/s)
    lake_q_min: float = 0.0  # Minimum (e.g. environmental) release (m³/s)
    lake_exp: float = 2.0  # Storage-discharge rating exponent
    lake_spill_coef: float = 1.0  # Spillway release rate above capacity (1/s)

    def __post_init__(self):
        if self.upstream_ids is None:
            self.upstream_ids = []


class NetworkArrays(NamedTuple):
    """JAX-compatible array representation of the network.

    This structure enables efficient vectorized operations in JAX
    while maintaining the DAG topology.

    Attributes:
        n_reaches: Number of reaches
        reach_ids: Array of reach IDs in topological order
        lengths: Reach lengths [n_reaches]
        slopes: Bed slopes [n_reaches]
        manning_n: Manning's n values [n_reaches]
        width_coef: Width coefficients [n_reaches]
        width_exp: Width exponents [n_reaches]
        depth_coef: Depth coefficients [n_reaches]
        depth_exp: Depth exponents [n_reaches]
        areas: Contributing areas [n_reaches]
        hru_ids: Associated HRU IDs [n_reaches]
        upstream_mask: Boolean mask [n_reaches, n_reaches] where
                       upstream_mask[i,j] = True if reach j is upstream of reach i
        downstream_idx: Index of downstream reach [n_reaches], -1 for outlets
        is_headwater: Boolean mask for headwater reaches [n_reaches]
        is_outlet: Boolean mask for outlet reaches [n_reaches]
    """

    n_reaches: int
    reach_ids: Array
    lengths: Array
    slopes: Array
    manning_n: Array
    width_coef: Array
    width_exp: Array
    depth_coef: Array
    depth_exp: Array
    areas: Array
    hru_ids: Array
    upstream_mask: Array
    downstream_idx: Array
    is_headwater: Array
    is_outlet: Array
    # Topological level of each reach (headwater=0, level = max(upstream)+1) and
    # the max level. Enable level-parallel routing: reaches sharing a level have
    # no mutual dependency and route together, cutting the autodiff sequential
    # depth from n_reaches to n_levels. Default None => sequential fallback.
    reach_level: Optional[Array] = None
    max_level: Optional[int] = None
    # Lake / reservoir attributes [n_reaches]. Default None => pure channel
    # routing (no lakes), preserving behaviour for networks built without them.
    is_lake: Optional[Array] = None
    lake_s_max: Optional[Array] = None
    lake_q_ref: Optional[Array] = None
    lake_q_min: Optional[Array] = None
    lake_exp: Optional[Array] = None
    lake_spill_coef: Optional[Array] = None


class RiverNetwork:
    """River network topology with topological ordering.

    Manages the DAG structure of a river network and provides efficient
    traversal orders for routing computations.

    Example:
        >>> network = RiverNetwork()
        >>> network.add_reach(Reach(id=0, length=1000, slope=0.001))
        >>> network.add_reach(Reach(id=1, length=2000, slope=0.0005,
        ...                         upstream_ids=[0], downstream_id=-1))
        >>> network.build_topology()
        >>> order = network.topological_order  # [0, 1]
    """

    def __init__(self):
        self.reaches: Dict[int, Reach] = {}
        self._topo_order: List[int] = []
        self._is_built: bool = False

    def add_reach(self, reach: Reach) -> None:
        """Add a reach to the network."""
        self.reaches[reach.id] = reach
        self._is_built = False

    def add_reaches(self, reaches: Sequence[Reach]) -> None:
        """Add multiple reaches to the network."""
        for reach in reaches:
            self.add_reach(reach)

    @property
    def n_reaches(self) -> int:
        """Number of reaches in the network."""
        return len(self.reaches)

    @property
    def topological_order(self) -> List[int]:
        """Get reach IDs in topological order (upstream to downstream)."""
        if not self._is_built:
            raise RuntimeError("Network topology not built. Call build_topology() first.")
        return self._topo_order

    def build_topology(self) -> None:
        """Build the topological ordering of reaches.

        Uses Kahn's algorithm for topological sorting.
        """
        if self.n_reaches == 0:
            self._topo_order = []
            self._is_built = True
            return

        # Build adjacency and compute in-degrees
        in_degree = {rid: 0 for rid in self.reaches}

        for rid, reach in self.reaches.items():
            in_degree[rid] = len(reach.upstream_ids)

        # Initialize queue with headwaters (in-degree 0)
        queue = [rid for rid, deg in in_degree.items() if deg == 0]
        result = []

        while queue:
            # Pop from queue
            rid = queue.pop(0)
            result.append(rid)

            # Find downstream reaches and decrement their in-degree
            reach = self.reaches[rid]
            if reach.downstream_id >= 0 and reach.downstream_id in self.reaches:
                in_degree[reach.downstream_id] -= 1
                if in_degree[reach.downstream_id] == 0:
                    queue.append(reach.downstream_id)

        # Check for cycles
        if len(result) != self.n_reaches:
            # Fall back to ID-based ordering
            result = sorted(self.reaches.keys())

        self._topo_order = result
        self._is_built = True

    def to_arrays(self) -> NetworkArrays:
        """Convert network to JAX-compatible arrays.

        Returns:
            NetworkArrays namedtuple with all network data as arrays
        """
        if not self._is_built:
            self.build_topology()

        n = self.n_reaches
        order = self._topo_order

        # Map reach IDs to indices
        id_to_idx = {rid: i for i, rid in enumerate(order)}

        # Initialize arrays
        reach_ids = np.array(order, dtype=np.int32)
        lengths = np.zeros(n, dtype=np.float32)
        slopes = np.zeros(n, dtype=np.float32)
        manning_n = np.zeros(n, dtype=np.float32)
        width_coef = np.zeros(n, dtype=np.float32)
        width_exp = np.zeros(n, dtype=np.float32)
        depth_coef = np.zeros(n, dtype=np.float32)
        depth_exp = np.zeros(n, dtype=np.float32)
        areas = np.zeros(n, dtype=np.float32)
        hru_ids = np.zeros(n, dtype=np.int32)
        upstream_mask = np.zeros((n, n), dtype=bool)
        downstream_idx = np.full(n, -1, dtype=np.int32)
        is_headwater = np.zeros(n, dtype=bool)
        is_outlet = np.zeros(n, dtype=bool)
        is_lake = np.zeros(n, dtype=bool)
        lake_s_max = np.zeros(n, dtype=np.float32)
        lake_q_ref = np.zeros(n, dtype=np.float32)
        lake_q_min = np.zeros(n, dtype=np.float32)
        lake_exp = np.full(n, 2.0, dtype=np.float32)
        lake_spill_coef = np.full(n, 1.0, dtype=np.float32)

        # Fill arrays
        for i, rid in enumerate(order):
            reach = self.reaches[rid]
            lengths[i] = reach.length
            slopes[i] = reach.slope
            manning_n[i] = reach.manning_n
            width_coef[i] = reach.width_coef
            width_exp[i] = reach.width_exp
            depth_coef[i] = reach.depth_coef
            depth_exp[i] = reach.depth_exp
            areas[i] = reach.area
            hru_ids[i] = reach.hru_id
            is_lake[i] = reach.is_lake
            lake_s_max[i] = reach.lake_s_max
            lake_q_ref[i] = reach.lake_q_ref
            lake_q_min[i] = reach.lake_q_min
            lake_exp[i] = reach.lake_exp
            lake_spill_coef[i] = reach.lake_spill_coef

            # Upstream mask
            for up_id in reach.upstream_ids:
                if up_id in id_to_idx:
                    upstream_mask[i, id_to_idx[up_id]] = True

            # Downstream index
            if reach.downstream_id >= 0 and reach.downstream_id in id_to_idx:
                downstream_idx[i] = id_to_idx[reach.downstream_id]

            # Headwater/outlet flags
            is_headwater[i] = len(reach.upstream_ids) == 0
            is_outlet[i] = reach.downstream_id < 0

        # Topological level: order is headwater-first (Kahn), so each reach's
        # upstream are already assigned. level = max(upstream level) + 1.
        reach_level = np.zeros(n, dtype=np.int32)
        for i in range(n):
            ups = np.where(upstream_mask[i])[0]
            reach_level[i] = (int(reach_level[ups].max()) + 1) if ups.size else 0
        max_level = int(reach_level.max()) if n else 0

        return NetworkArrays(
            n_reaches=n,
            reach_ids=jnp.array(reach_ids),
            lengths=jnp.array(lengths),
            slopes=jnp.array(slopes),
            manning_n=jnp.array(manning_n),
            width_coef=jnp.array(width_coef),
            width_exp=jnp.array(width_exp),
            depth_coef=jnp.array(depth_coef),
            depth_exp=jnp.array(depth_exp),
            areas=jnp.array(areas),
            hru_ids=jnp.array(hru_ids),
            upstream_mask=jnp.array(upstream_mask),
            downstream_idx=jnp.array(downstream_idx),
            is_headwater=jnp.array(is_headwater),
            is_outlet=jnp.array(is_outlet),
            reach_level=jnp.array(reach_level),
            max_level=max_level,
            is_lake=jnp.array(is_lake),
            lake_s_max=jnp.array(lake_s_max),
            lake_q_ref=jnp.array(lake_q_ref),
            lake_q_min=jnp.array(lake_q_min),
            lake_exp=jnp.array(lake_exp),
            lake_spill_coef=jnp.array(lake_spill_coef),
        )

    def get_outlet_ids(self) -> List[int]:
        """Get IDs of outlet reaches."""
        return [rid for rid, reach in self.reaches.items() if reach.downstream_id < 0]

    def get_headwater_ids(self) -> List[int]:
        """Get IDs of headwater reaches."""
        return [rid for rid, reach in self.reaches.items() if len(reach.upstream_ids) == 0]


def create_network_from_topology(
    reach_ids: Sequence[int],
    downstream_ids: Sequence[int],
    lengths: Sequence[float],
    slopes: Sequence[float],
    manning_n: Optional[Sequence[float]] = None,
    areas: Optional[Sequence[float]] = None,
    hru_ids: Optional[Sequence[int]] = None,
) -> RiverNetwork:
    """Create a river network from topology arrays.

    This is a convenience function for creating networks from typical
    GIS-derived topology data.

    Args:
        reach_ids: Unique reach identifiers
        downstream_ids: Downstream reach ID for each reach (-1 for outlets)
        lengths: Reach lengths in meters
        slopes: Bed slopes (m/m)
        manning_n: Manning's n values (default 0.035)
        areas: Contributing areas in m² (default 0)
        hru_ids: Associated HRU IDs (default same as reach_ids)

    Returns:
        Configured RiverNetwork
    """
    n = len(reach_ids)

    # Defaults. mizuRoute topologies carry a separate HRU dimension
    # (n_HRU != n_seg), so per-HRU arrays passed here (areas, hru_ids) may not be
    # per-reach — fall back rather than index out of range.
    if manning_n is None or len(manning_n) != n:
        manning_n = [0.035] * n
    if areas is None or len(areas) != n:
        areas = [0.0] * n
    if hru_ids is None or len(hru_ids) != n:
        hru_ids = list(reach_ids)

    # Build upstream relationships
    upstream_map: Dict[int, List[int]] = {rid: [] for rid in reach_ids}
    for i, rid in enumerate(reach_ids):
        ds_id = downstream_ids[i]
        if ds_id >= 0 and ds_id in upstream_map:
            upstream_map[ds_id].append(rid)

    # Create network
    network = RiverNetwork()

    for i, rid in enumerate(reach_ids):
        reach = Reach(
            id=rid,
            length=lengths[i],
            slope=max(slopes[i], 1e-6),  # Ensure positive slope
            manning_n=manning_n[i],
            upstream_ids=upstream_map[rid],
            downstream_id=downstream_ids[i],
            hru_id=hru_ids[i],
            area=areas[i],
        )
        network.add_reach(reach)

    network.build_topology()
    return network
