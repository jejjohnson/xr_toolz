# xr_toolz

[![Tests](https://github.com/jejjohnson/xr_toolz/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/xr_toolz/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/xr_toolz/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/xr_toolz/actions/workflows/lint.yml)
[![Type Check](https://github.com/jejjohnson/xr_toolz/actions/workflows/typecheck.yml/badge.svg)](https://github.com/jejjohnson/xr_toolz/actions/workflows/typecheck.yml)
[![Deploy Docs](https://github.com/jejjohnson/xr_toolz/actions/workflows/pages.yml/badge.svg)](https://github.com/jejjohnson/xr_toolz/actions/workflows/pages.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

**Composable operator library for geoprocessing Earth System Data Cubes.**

`xr_toolz` provides a uniform `Operator` abstraction for preprocessing, inference, and evaluation of xarray datasets. Pipelines compose linearly via `Sequential` or as DAGs via the functional `Graph` API — the same operator works in both.

## Package layout

```
xr_toolz/
├── core/   # Operator, Sequential, Input, Node, Graph
├── geo/    # Generic xarray geoprocessing (validation, subset, regrid,
│           # detrend, masks, metrics, spectral, ...)
├── ocn/    # Oceanography physics (streamfunction, geostrophic velocity, ...)
├── atm/    # Atmospheric physics (potential temperature, wind, ...)
│   └── gas/ch4/  # Trace-gas physics (column averaging kernel, ...)
├── rs/     # Remote sensing (NDVI, radiance/reflectance, ...)
└── ice/    # Cryosphere (reserved; no content yet)
```

Rule: anything domain-agnostic lives in `geo`; only true physics lives in the other domain submodules.

## Quick start

```bash
# Prerequisites: uv (https://github.com/astral-sh/uv)
git clone https://github.com/jejjohnson/xr_toolz.git
cd xr_toolz
make install      # install all dependency groups
make test         # run tests
make docs-serve   # preview docs locally
```

## Status

Pre-alpha. See the full design document in [`docs/design/`](docs/design/) for motivation, architecture, boundaries, and roadmap.

## License

MIT. See [LICENSE](LICENSE).
