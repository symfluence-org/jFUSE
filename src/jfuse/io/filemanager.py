"""
FUSE File Manager Parser

Parses FUSE file manager files that specify paths, settings, and run configuration.
Compatible with FUSE_FILEMANAGER_V1.5 format.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime
import re


@dataclass
class FileManagerConfig:
    """Configuration from FUSE file manager file.

    Attributes:
        settings_path: Path to settings files
        input_path: Path to input forcing files
        output_path: Path for output files
        suffix_forcing: Suffix for forcing files (appended to basin_id)
        suffix_elev_bands: Suffix for elevation bands file
        suffix_network: Suffix for network topology file
        forcing_info_file: Name of forcing info file
        constraints_file: Name of parameter constraints file
        numerix_file: Name of numerical solution file
        decisions_file: Name of model decisions file
        model_id: String identifier for the model run
        q_only: Only write Q to output (True) or all variables (False)
        date_start_sim: Start date of simulation
        date_end_sim: End date of simulation
        date_start_eval: Start date of evaluation period
        date_end_eval: End date of evaluation period
        numtim_sub: Number of timesteps per sub-period (-9999 for no sub-periods)
        metric: Objective function metric (NSE, KGE, etc.)
        transform: Streamflow transformation (log, boxcox, or power value)
        maxn: SCE max trials
        kstop: SCE shuffling loops
        pcento: SCE percentage threshold
    """

    # Paths
    settings_path: Path = field(default_factory=lambda: Path("."))
    input_path: Path = field(default_factory=lambda: Path("."))
    output_path: Path = field(default_factory=lambda: Path("."))

    # File suffixes
    suffix_forcing: str = "_input.nc"
    suffix_elev_bands: str = "_elev_bands.nc"
    suffix_network: str = "_network.nc"

    # Settings files
    forcing_info_file: str = "input_info.txt"
    constraints_file: str = "fuse_zConstraints.txt"
    numerix_file: str = "fuse_zNumerix.txt"
    decisions_file: str = "fuse_zDecisions.txt"

    # Output settings
    model_id: str = "jfuse_run"
    q_only: bool = False

    # Dates
    date_start_sim: Optional[datetime] = None
    date_end_sim: Optional[datetime] = None
    date_start_eval: Optional[datetime] = None
    date_end_eval: Optional[datetime] = None
    numtim_sub: int = -9999

    # Evaluation settings
    metric: str = "KGE"
    transform: str = "1"  # Power transform, "log", or "boxcox"

    # SCE parameters
    maxn: int = 1000
    kstop: int = 3
    pcento: float = 0.001

    @property
    def decisions_path(self) -> Path:
        """Full path to decisions file."""
        return self.settings_path / self.decisions_file

    @property
    def forcing_info_path(self) -> Path:
        """Full path to forcing info file."""
        return self.settings_path / self.forcing_info_file

    @property
    def constraints_path(self) -> Path:
        """Full path to constraints file."""
        return self.settings_path / self.constraints_file

    @property
    def numerix_path(self) -> Path:
        """Full path to numerix file."""
        return self.settings_path / self.numerix_file

    def forcing_file(self, basin_id: str) -> Path:
        """Get forcing file path for a basin."""
        return self.input_path / f"{basin_id}{self.suffix_forcing}"

    def elev_bands_file(self, basin_id: str) -> Path:
        """Get elevation bands file path for a basin."""
        return self.input_path / f"{basin_id}{self.suffix_elev_bands}"

    def network_file(self, basin_id: str) -> Path:
        """Get network topology file path for a basin."""
        return self.input_path / f"{basin_id}{self.suffix_network}"

    def output_file(self, suffix: str = "_output.nc") -> Path:
        """Get output file path."""
        return self.output_path / f"{self.model_id}{suffix}"


def parse_filemanager(filepath: str) -> FileManagerConfig:
    """Parse a FUSE file manager file.

    Args:
        filepath: Path to the file manager file (e.g., fm_catch.txt)

    Returns:
        FileManagerConfig with parsed settings

    Raises:
        ValueError: If file format is invalid
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"File manager not found: {filepath}")

    config = FileManagerConfig()

    with open(path, "r") as f:
        lines = f.readlines()

    # Check header
    if not lines or "FUSE_FILEMANAGER" not in lines[0]:
        raise ValueError(f"Invalid file manager format: {filepath}")

    # Parse values - extract quoted strings or bare values
    values = []
    for line in lines[1:]:
        line = line.strip()

        # Skip empty lines and pure comments
        if not line or line.startswith("!"):
            continue

        # Extract quoted value
        match = re.match(r"'([^']*)'", line)
        if match:
            values.append(match.group(1).strip())
        else:
            # Try unquoted value (before any comment)
            parts = line.split("!")
            if parts:
                val = parts[0].strip()
                if val:
                    values.append(val)

    # Map values to config fields
    # Expected order based on FUSE file manager format
    idx = 0

    def get_value(default=None):
        nonlocal idx
        if idx < len(values):
            val = values[idx]
            idx += 1
            return val
        return default

    # Paths
    config.settings_path = Path(get_value("."))
    config.input_path = Path(get_value("."))
    config.output_path = Path(get_value("."))

    # Suffixes
    config.suffix_forcing = get_value("_input.nc")
    config.suffix_elev_bands = get_value("_elev_bands.nc")

    # Settings files
    config.forcing_info_file = get_value("input_info.txt")
    config.constraints_file = get_value("fuse_zConstraints.txt")
    config.numerix_file = get_value("fuse_zNumerix.txt")
    config.decisions_file = get_value("fuse_zDecisions.txt")

    # Output settings
    config.model_id = get_value("jfuse_run")
    q_only_str = get_value("FALSE")
    config.q_only = q_only_str.upper() == "TRUE"

    # Dates
    def parse_date(date_str):
        if not date_str or date_str == "-9999":
            return None
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return None

    config.date_start_sim = parse_date(get_value())
    config.date_end_sim = parse_date(get_value())
    config.date_start_eval = parse_date(get_value())
    config.date_end_eval = parse_date(get_value())

    numtim = get_value("-9999")
    config.numtim_sub = int(numtim) if numtim else -9999

    # Evaluation settings
    config.metric = get_value("KGE")
    config.transform = get_value("1")

    # SCE parameters
    maxn = get_value("1000")
    config.maxn = int(maxn) if maxn else 1000

    kstop = get_value("3")
    config.kstop = int(kstop) if kstop else 3

    pcento = get_value("0.001")
    config.pcento = float(pcento) if pcento else 0.001

    return config


@dataclass
class ForcingInfo:
    """Information about forcing file variables.

    Attributes:
        precip_var: Name of precipitation variable
        pet_var: Name of PET variable (or variables to compute it)
        temp_var: Name of temperature variable
        time_var: Name of time variable
        hru_var: Name of HRU/space dimension
        obs_var: Name of observed streamflow variable (optional)
    """

    precip_var: str = "pptrate"
    pet_var: str = "pet"
    temp_var: str = "airtemp"
    time_var: str = "time"
    hru_var: str = "hru"
    obs_var: Optional[str] = "q_obs"

    # Units conversion factors (to mm/day and °C)
    precip_mult: float = 86400.0  # kg/m2/s to mm/day
    pet_mult: float = 86400.0
    temp_offset: float = -273.15  # K to °C (if needed)

    # Additional variables that might be needed
    extra_vars: Dict[str, str] = field(default_factory=dict)


def parse_forcing_info(filepath: str) -> ForcingInfo:
    """Parse a FUSE forcing info file.

    The forcing info file defines variable names and units in the forcing NetCDF.

    Args:
        filepath: Path to forcing info file

    Returns:
        ForcingInfo with variable mappings
    """
    path = Path(filepath)

    if not path.exists():
        # Return defaults if file doesn't exist
        return ForcingInfo()

    info = ForcingInfo()

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith("!"):
                continue

            # Parse key-value pairs
            # Format could be: 'variable_name' ! PRECIP_VAR
            # or: PRECIP_VAR = 'variable_name'

            match = re.match(r"'([^']+)'.*!(.*)", line)
            if match:
                value = match.group(1).strip()
                comment = match.group(2).strip().upper()

                if "PRECIP" in comment or "PPT" in comment:
                    info.precip_var = value
                elif "PET" in comment or "EVAP" in comment:
                    info.pet_var = value
                elif "TEMP" in comment or "TAIR" in comment:
                    info.temp_var = value
                elif "TIME" in comment:
                    info.time_var = value
                elif "HRU" in comment or "SPACE" in comment:
                    info.hru_var = value
                elif "OBS" in comment or "Q_OBS" in comment:
                    info.obs_var = value

    return info


def write_filemanager(config: FileManagerConfig, filepath: str):
    """Write a FUSE file manager file.

    Args:
        config: FileManagerConfig to write
        filepath: Output path
    """
    with open(filepath, "w") as f:
        f.write("FUSE_FILEMANAGER_V1.5\n")
        f.write("! *** paths\n")
        f.write(f"'{config.settings_path}/'     ! SETNGS_PATH\n")
        f.write(f"'{config.input_path}/'        ! INPUT_PATH\n")
        f.write(f"'{config.output_path}/'       ! OUTPUT_PATH\n")

        f.write("! *** suffixes for input files\n")
        f.write(f"'{config.suffix_forcing}'     ! suffix_forcing\n")
        f.write(f"'{config.suffix_elev_bands}'  ! suffix_elev_bands\n")

        f.write("! *** settings files\n")
        f.write(f"'{config.forcing_info_file}'  ! FORCING_INFO\n")
        f.write(f"'{config.constraints_file}'   ! CONSTRAINTS\n")
        f.write(f"'{config.numerix_file}'       ! MOD_NUMERIX\n")
        f.write(f"'{config.decisions_file}'     ! M_DECISIONS\n")

        f.write("! *** output files\n")
        f.write(f"'{config.model_id}'           ! FMODEL_ID\n")
        f.write(f"'{'TRUE' if config.q_only else 'FALSE'}'  ! Q_ONLY\n")

        f.write("! *** dates\n")
        f.write(
            f"'{config.date_start_sim.strftime('%Y-%m-%d') if config.date_start_sim else '-9999'}'  ! date_start_sim\n"
        )
        f.write(
            f"'{config.date_end_sim.strftime('%Y-%m-%d') if config.date_end_sim else '-9999'}'    ! date_end_sim\n"
        )
        f.write(
            f"'{config.date_start_eval.strftime('%Y-%m-%d') if config.date_start_eval else '-9999'}'  ! date_start_eval\n"
        )
        f.write(
            f"'{config.date_end_eval.strftime('%Y-%m-%d') if config.date_end_eval else '-9999'}'    ! date_end_eval\n"
        )
        f.write(f"'{config.numtim_sub}'         ! numtim_sub\n")

        f.write("! *** evaluation\n")
        f.write(f"'{config.metric}'             ! METRIC\n")
        f.write(f"'{config.transform}'          ! TRANSFO\n")

        f.write("! *** SCE parameters\n")
        f.write(f"'{config.maxn}'               ! MAXN\n")
        f.write(f"'{config.kstop}'              ! KSTOP\n")
        f.write(f"'{config.pcento}'             ! PCENTO\n")
