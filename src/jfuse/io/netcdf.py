"""
NetCDF I/O Utilities

Functions for reading and writing jFUSE data from/to NetCDF files.
Supports the standard data formats used by FUSE and mizuRoute.
"""

from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import jax.numpy as jnp
from jax import Array

try:
    import xarray as xr

    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

try:
    import netCDF4 as nc4

    HAS_NETCDF4 = True
except ImportError:
    HAS_NETCDF4 = False

from ..routing import RiverNetwork, create_network_from_topology
from ..fuse import State


class ForcingData:
    """Container for forcing time series data.

    Attributes:
        precip: Precipitation [n_timesteps, n_hrus] (mm/day)
        pet: Potential ET [n_timesteps, n_hrus] (mm/day)
        temp: Temperature [n_timesteps, n_hrus] (°C)
        time: Time coordinate
        hru_ids: HRU identifiers
    """

    def __init__(
        self,
        precip: Array,
        pet: Array,
        temp: Array,
        time: Optional[Array] = None,
        hru_ids: Optional[Array] = None,
    ):
        self.precip = precip
        self.pet = pet
        self.temp = temp
        self.time = time
        self.hru_ids = hru_ids

    @property
    def n_timesteps(self) -> int:
        return self.precip.shape[0]

    @property
    def n_hrus(self) -> int:
        return self.precip.shape[1] if self.precip.ndim > 1 else 1

    def to_tuple(self) -> Tuple[Array, Array, Array]:
        """Convert to tuple for model input."""
        return (self.precip, self.pet, self.temp)


def load_forcing(
    filepath: str,
    precip_var: str = "precip",
    pet_var: str = "pet",
    temp_var: str = "temp",
    time_dim: str = "time",
    hru_dim: str = "hru",
) -> ForcingData:
    """Load forcing data from NetCDF file.

    Args:
        filepath: Path to NetCDF file
        precip_var: Variable name for precipitation
        pet_var: Variable name for potential ET
        temp_var: Variable name for temperature
        time_dim: Name of time dimension
        hru_dim: Name of HRU dimension

    Returns:
        ForcingData container
    """
    if not HAS_XARRAY and not HAS_NETCDF4:
        raise ImportError("Either xarray or netCDF4 is required for NetCDF I/O")

    # ds and the coordinate arrays hold different concrete types depending on
    # the backend (xarray vs netCDF4); treat them as dynamic.
    ds: Any
    time: Any
    hru_ids: Any

    if HAS_XARRAY:
        ds = xr.open_dataset(filepath)

        # Try to get variables with different naming conventions
        precip = _get_variable(ds, [precip_var, "ppt", "precipitation", "pr", "P"])
        pet = _get_variable(ds, [pet_var, "PET", "evap", "E"])
        temp = _get_variable(ds, [temp_var, "temperature", "tas", "T", "airtemp"])

        # Convert to JAX arrays
        precip = jnp.array(precip.values)
        pet = jnp.array(pet.values)
        temp = jnp.array(temp.values)

        # Get coordinates
        time = ds[time_dim].values if time_dim in ds.coords else None
        hru_ids = ds[hru_dim].values if hru_dim in ds.dims else None

        ds.close()

    else:  # Use netCDF4
        ds = nc4.Dataset(filepath, "r")

        precip = jnp.array(ds.variables[precip_var][:])
        pet = jnp.array(ds.variables[pet_var][:])
        temp = jnp.array(ds.variables[temp_var][:])

        time = ds.variables[time_dim][:] if time_dim in ds.variables else None
        hru_ids = ds.variables[hru_dim][:] if hru_dim in ds.variables else None

        ds.close()

    # Ensure proper shape [time, hru]
    if precip.ndim == 1:
        precip = precip[:, None]
        pet = pet[:, None]
        temp = temp[:, None]

    return ForcingData(precip, pet, temp, time, hru_ids)


def _get_variable(ds, names: List[str]):
    """Try to get a variable using multiple possible names."""
    for name in names:
        if name in ds:
            return ds[name]
    raise KeyError(f"Could not find variable. Tried: {names}")


def load_network(
    filepath: str,
    reach_id_var: Optional[str] = None,
    downstream_var: Optional[str] = None,
    length_var: Optional[str] = None,
    slope_var: Optional[str] = None,
    manning_var: Optional[str] = None,
    area_var: Optional[str] = None,
    hru_id_var: Optional[str] = None,
) -> Tuple[RiverNetwork, Array]:
    """Load river network from NetCDF file.

    Supports both jFUSE-style and mizuRoute-style variable naming conventions.
    If variable names are not specified, attempts to auto-detect from the file.

    Args:
        filepath: Path to network NetCDF file
        reach_id_var: Variable name for reach IDs (auto: reach_id, segId)
        downstream_var: Variable name for downstream reach IDs (auto: downstream_id, downSegId)
        length_var: Variable name for reach lengths (auto: length)
        slope_var: Variable name for bed slopes (auto: slope)
        manning_var: Variable name for Manning's n (auto: manning_n, n)
        area_var: Variable name for contributing areas (auto: area)
        hru_id_var: Variable name for HRU IDs (auto: hru_id, hruId)

    Returns:
        Tuple of (RiverNetwork, hru_areas)
    """
    if not HAS_XARRAY and not HAS_NETCDF4:
        raise ImportError("Either xarray or netCDF4 is required for NetCDF I/O")

    ds: Any
    if HAS_XARRAY:
        ds = xr.open_dataset(filepath)
        var_names = list(ds.data_vars)
    else:
        ds = nc4.Dataset(filepath, "r")
        var_names = list(ds.variables.keys())

    # Auto-detect variable names if not specified
    def find_var(options, required=True):
        """Find first matching variable name from options list."""
        for opt in options:
            if opt in var_names:
                return opt
        if required:
            raise KeyError(f"Could not find any of {options} in {var_names}")
        return None

    # Reach ID: reach_id (jFUSE), segId (mizuRoute)
    if reach_id_var is None:
        reach_id_var = find_var(["reach_id", "segId", "seg_id", "reachId", "COMID"])

    # Downstream ID: downstream_id (jFUSE), downSegId (mizuRoute)
    if downstream_var is None:
        downstream_var = find_var(
            ["downstream_id", "downSegId", "down_seg_id", "tosegment", "toSegment", "NextDownID"]
        )

    # Length
    if length_var is None:
        length_var = find_var(["length", "Length", "seg_length", "LENGTHKM"])

    # Slope
    if slope_var is None:
        slope_var = find_var(["slope", "Slope", "seg_slope", "So"])

    # Manning's n (optional)
    if manning_var is None:
        manning_var = find_var(["manning_n", "n", "Mann_n", "roughness"], required=False)

    # Area (optional but commonly available)
    if area_var is None:
        area_var = find_var(["area", "Area", "hruArea", "basin_area", "TotDASqKM"], required=False)

    # HRU ID (optional)
    if hru_id_var is None:
        hru_id_var = find_var(["hru_id", "hruId", "hru_to_seg", "hruToSegId"], required=False)

    # Read variables
    if HAS_XARRAY:
        reach_ids = ds[reach_id_var].values
        downstream_ids = ds[downstream_var].values
        lengths = ds[length_var].values
        slopes = ds[slope_var].values

        manning_n = ds[manning_var].values if manning_var and manning_var in ds else None
        areas = ds[area_var].values if area_var and area_var in ds else None
        hru_ids = ds[hru_id_var].values if hru_id_var and hru_id_var in ds else None

        ds.close()
    else:
        reach_ids = ds.variables[reach_id_var][:]
        downstream_ids = ds.variables[downstream_var][:]
        lengths = ds.variables[length_var][:]
        slopes = ds.variables[slope_var][:]

        manning_n = (
            ds.variables[manning_var][:] if manning_var and manning_var in ds.variables else None
        )
        areas = ds.variables[area_var][:] if area_var and area_var in ds.variables else None
        hru_ids = ds.variables[hru_id_var][:] if hru_id_var and hru_id_var in ds.variables else None

        ds.close()

    # Handle length units - mizuRoute uses km, we need m
    if "km" in length_var.lower() or np.nanmean(lengths) < 100:
        # Likely in km, convert to m
        lengths = lengths * 1000.0

    # Convert to lists for processing
    reach_ids_list = list(reach_ids)
    downstream_ids_list = list(downstream_ids)

    # Fix outlet detection - different systems use different conventions:
    # - mizuRoute: 0 often means "no downstream" (outlet)
    # - Some systems: -1 means outlet
    # - Some systems: outlet downstream_id points to non-existent reach
    # We standardize to -1 for outlets
    reach_id_set = set(reach_ids_list)
    for i in range(len(downstream_ids_list)):
        ds_id = downstream_ids_list[i]
        # Mark as outlet (-1) if:
        # 1. downstream_id = 0 (common convention)
        # 2. downstream_id < 0 (already outlet)
        # 3. downstream_id not in reach_ids (orphan/outlet)
        # 4. downstream_id = reach_id (self-referencing, rare)
        if ds_id == 0 or ds_id < 0 or ds_id not in reach_id_set or ds_id == reach_ids_list[i]:
            downstream_ids_list[i] = -1

    # Create network
    network = create_network_from_topology(
        reach_ids=reach_ids_list,
        downstream_ids=downstream_ids_list,
        lengths=list(lengths),
        slopes=list(slopes),
        manning_n=list(manning_n) if manning_n is not None else None,
        areas=list(areas) if areas is not None else None,
        hru_ids=list(hru_ids) if hru_ids is not None else None,
    )

    # Get HRU areas
    hru_areas = jnp.array(areas) if areas is not None else jnp.ones(len(reach_ids)) * 1e6

    return network, hru_areas


def load_observations(
    filepath: str,
    discharge_var: str = "discharge",
    time_dim: str = "time",
) -> Tuple[Array, Optional[Array]]:
    """Load observed discharge from NetCDF file.

    Args:
        filepath: Path to observations NetCDF file
        discharge_var: Variable name for discharge
        time_dim: Name of time dimension

    Returns:
        Tuple of (discharge array, time array)
    """
    # Backend-dependent concrete types; treat as dynamic.
    ds: Any
    time: Any
    if HAS_XARRAY:
        ds = xr.open_dataset(filepath)
        discharge = jnp.array(ds[discharge_var].values)
        time = ds[time_dim].values if time_dim in ds.coords else None
        ds.close()
    elif HAS_NETCDF4:
        ds = nc4.Dataset(filepath, "r")
        discharge = jnp.array(ds.variables[discharge_var][:])
        time = ds.variables[time_dim][:] if time_dim in ds.variables else None
        ds.close()
    else:
        raise ImportError("Either xarray or netCDF4 is required")

    return discharge, time


def save_results(
    filepath: str,
    outlet_discharge: Array,
    runoff: Optional[Array] = None,
    time: Optional[Array] = None,
    parameters: Optional[Dict[str, Array]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save simulation results to NetCDF file.

    Args:
        filepath: Output path
        outlet_discharge: Outlet discharge [n_timesteps]
        runoff: HRU runoff [n_timesteps, n_hrus] (optional)
        time: Time coordinate (optional)
        parameters: Dictionary of calibrated parameters (optional)
        metadata: Additional metadata attributes (optional)
    """
    if not HAS_XARRAY:
        raise ImportError("xarray is required for saving results")

    # Convert JAX arrays to numpy for xarray.
    outlet_np = np.asarray(outlet_discharge)

    # Build dataset
    data_vars: Dict[str, Any] = {
        "outlet_discharge": (["time"], outlet_np),
    }

    if runoff is not None:
        data_vars["runoff"] = (["time", "hru"], np.asarray(runoff))

    coords: Dict[str, Any] = {}
    if time is not None:
        coords["time"] = time
    else:
        coords["time"] = np.arange(len(outlet_np))

    if runoff is not None:
        coords["hru"] = np.arange(runoff.shape[1])

    ds = xr.Dataset(data_vars, coords=coords)

    # Add parameters as variables
    if parameters is not None:
        for name, values in parameters.items():
            values_np = np.asarray(values)
            if values_np.ndim == 0:
                ds.attrs[f"param_{name}"] = float(values_np)
            elif values_np.ndim == 1:
                ds[f"param_{name}"] = (["hru"], values_np)

    # Add metadata
    if metadata is not None:
        ds.attrs.update(metadata)

    ds.attrs["creator"] = "jFUSE"
    ds.attrs["conventions"] = "CF-1.8"

    ds.to_netcdf(filepath)
    ds.close()


def save_state(
    filepath: str,
    state: State,
    description: str = "Model state checkpoint",
) -> None:
    """Save model state to NetCDF file for restart.

    Args:
        filepath: Output path
        state: FUSE state to save
        description: Description attribute
    """
    if not HAS_XARRAY:
        raise ImportError("xarray is required for saving state")

    data_vars: Dict[str, Any] = {}
    for field in ["S1", "S1_T", "S1_TA", "S1_TB", "S1_F", "S2", "S2_T", "S2_FA", "S2_FB", "SWE"]:
        values = np.array(getattr(state, field))
        if values.ndim == 0:
            data_vars[field] = ([], values)
        else:
            data_vars[field] = (["hru"], values)

    ds = xr.Dataset(data_vars)
    ds.attrs["description"] = description
    ds.attrs["creator"] = "jFUSE"

    ds.to_netcdf(filepath)
    ds.close()


def load_state(filepath: str) -> State:
    """Load model state from NetCDF checkpoint.

    Args:
        filepath: Path to state file

    Returns:
        State object
    """
    if not HAS_XARRAY:
        raise ImportError("xarray is required for loading state")

    ds = xr.open_dataset(filepath)

    state = State(
        S1=jnp.array(ds["S1"].values),
        S1_T=jnp.array(ds["S1_T"].values),
        S1_TA=jnp.array(ds["S1_TA"].values),
        S1_TB=jnp.array(ds["S1_TB"].values),
        S1_F=jnp.array(ds["S1_F"].values),
        S2=jnp.array(ds["S2"].values),
        S2_T=jnp.array(ds["S2_T"].values),
        S2_FA=jnp.array(ds["S2_FA"].values),
        S2_FB=jnp.array(ds["S2_FB"].values),
        SWE=jnp.array(ds["SWE"].values),
    )

    ds.close()
    return state
