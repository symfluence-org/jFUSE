<p align="center">
  <img src="src/assets/jfuse_logo.png" alt="jFUSE logo" width="180">
</p>

# JAX-based differentiable hydrological modeling framework implementing FUSE

![CI](https://github.com/symfluence-org/jFUSE/actions/workflows/ci.yml/badge.svg)
[![PyPI version](https://img.shields.io/pypi/v/jfuse.svg)](https://pypi.org/project/jfuse/)
[![Python versions](https://img.shields.io/pypi/pyversions/jfuse.svg)](https://pypi.org/project/jfuse/)
[![License](https://img.shields.io/github/license/symfluence-org/jFUSE.svg)](LICENSE)

A fully differentiable JAX implementation of the Framework for Understanding Structural Errors (FUSE) hydrological model from Clark et al. (2008), with Muskingum-Cunge routing.

**Note: jFUSE is in active development**

## Features

- **Fully differentiable**: Automatic differentiation through the entire model using JAX
- **JIT-compiled**: Fast execution with XLA compilation
- **FUSE decision file compatible**: Read standard FUSE decision files to configure model structure
- **Muskingum-Cunge routing**: Network-based streamflow routing with adaptive parameters
- **Gradient-based calibration**: Built-in calibration with optax optimizers
- **GPU-ready**: Seamless scaling to GPU with JAX

### Requirements
- Python >= 3.11
- JAX >= 0.4.0
- equinox >= 0.11.0
- optax >= 0.1.7
- numpy
- xarray (for NetCDF I/O)

## Installation

```bash
# Clone or download the package
git clone https://github.com/symfluence-org/jFUSE.git
cd jfuse

# Install in development mode
pip install -e .

# Or install with all dependencies
pip install -e ".[dev]"

# Or install with PyPi
pip install jfuse          # CPU
pip install jfuse[gpu]     # CUDA

```
## Command Line Interface

jFUSE provides a CLI compatible with FUSE file manager format:

```bash
# Run simulation
jfuse run fm_catch.txt bow_at_banff

# Run gradient-based calibration
jfuse run fm_catch.txt bow_at_banff --mode=calib --method=gradient

# Show file manager configuration
jfuse info fm_catch.txt
```

## Download Example Data

We provide a ready-to-use example dataset for jFUSE (lumped and distributed setups).

Download the ZIP release asset and unzip it into a `data/` folder:

```bash
wget https://github.com/symfluence-org/jFUSE/releases/latest/download/jfuse-example-data.zip
unzip jfuse-example-data.zip
mv jfuse-example-data data
```

## Lumped configuration example

This example calibrates a lumped jFUSE model using ERA5 forcing.

```bash
jfuse run \
  data/domain_Bow_at_Banff_lumped_era5/settings/FUSE/fm_catch.txt \
  Bow_at_Banff_lumped_era5 \
  --mode=calib \
  --method=gradient \
  --loss=kge \
  --lr=0.01 \
  --epochs=500 \
  --plot
```

## Distributed configuration example

This example calibrates a distributed jFUSE model using a Muskingum–Cunge routing network.

```bash
jfuse run \
  data/domain_Bow_at_Banff_distributed/settings/FUSE/fm_catch.txt \
  Bow_at_Banff_distributed \
  --mode=calib \
  --method=gradient \
  --network=data/domain_Bow_at_Banff_distributed/settings/mizuRoute/topology.nc \
  --obs-file=data/domain_Bow_at_Banff_distributed/observations/streamflow/preprocessed/Bow_at_Banff_distributed_streamflow_processed.csv \
  --loss=kge,nse \
  --lr=0.01 \
  --epochs=1000 \
  --plot
```

## License

MIT License - see LICENSE file for details.

## References

- Clark, M. P., et al. (2008). Framework for Understanding Structural Errors (FUSE). Water Resources Research, 44, W00B02.
- Cunge, J. A. (1969). On the subject of a flood propagation computation method (Muskingum method). Journal of Hydraulic Research, 7(2), 205-230.
