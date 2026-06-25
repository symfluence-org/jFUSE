"""I/O utilities for jFUSE.

This module provides functions for loading and saving data in various formats,
primarily NetCDF for hydrological forcing, network topology, and simulation results.
Also provides FUSE file manager parsing for compatibility with original FUSE.
"""

from .netcdf import (
    load_forcing,
    load_network,
    load_observations,
    save_results,
    save_state,
    load_state,
    ForcingData,
)

from .filemanager import (
    FileManagerConfig,
    ForcingInfo,
    parse_filemanager,
    parse_forcing_info,
    write_filemanager,
)

__all__ = [
    # NetCDF I/O
    "load_forcing",
    "load_network",
    "load_observations",
    "save_results",
    "save_state",
    "load_state",
    "ForcingData",
    # File manager
    "FileManagerConfig",
    "ForcingInfo",
    "parse_filemanager",
    "parse_forcing_info",
    "write_filemanager",
]
