---
status: draft
version: 0.1.0
---

# Layer 0 — Primitive Examples

Pure function usage patterns. *(P2: progressive disclosure — simplest layer first)*

---

## Pipe Syntax with toolz

### Compose L0 functions into functional pipelines

```python
from toolz import compose_left
from functools import partial
from geo_toolz._src.validation.coords import validate_longitude, validate_latitude
from geo_toolz._src.subset.where import subset_bbox
from geo_toolz._src.detrend.climatology import remove_climatology

preprocess = compose_left(
    validate_longitude,
    validate_latitude,
    partial(subset_bbox, lon_bnds=(-30, 45), lat_bnds=(25, 65)),
    partial(remove_climatology, climatology=clim),
)

ds_clean = preprocess(ds_raw)
```

---

## Direct Function Calls

### xr.Dataset.pipe for one-off transforms

```python
from geo_toolz._src.detrend.climatology import calculate_climatology, remove_climatology

# Learn state from training data
clim = calculate_climatology(ds_train, freq="day", smoothing=60)

# Apply — P3: xarray in, xarray out
ds_anom = remove_climatology(ds_test, clim)
```
