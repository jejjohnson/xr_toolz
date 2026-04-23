---
status: draft
version: 0.1.0
---

# geo_toolz — Examples

Usage patterns organized by API layer. Demonstrates the design principles:
- **P1: Everything is an operator** — uniform `__call__` interface
- **P2: Progressive disclosure** — L0 functions → L1 operators → L2 graphs
- **P3: xarray in, xarray out** — coordinates and metadata preserved
- **P4: Bring your own model** — inference as a pipeline step

## Structure

```
examples/
├── README.md              # This file
├── primitives.md          # Layer 0 — pure functions, pipe syntax, toolz composition
├── components.md          # Layer 1 — Sequential, operator composition, Hydra, stateful ops
├── models.md              # Layer 2 — Graph API, inference, model comparison
└── integration.md         # Layer 3 — sklearn, xrpatcher, xarray_sklearn, ekalmX
```

## Reading Order

1. **[primitives.md](primitives.md)** — L0: pure functions and pipe syntax
2. **[components.md](components.md)** — L1: Sequential pipelines and operator patterns
3. **[models.md](models.md)** — L2: Graph API, ModelOp, model comparison
4. **[integration.md](integration.md)** — L3: sklearn, xrpatcher, xarray_sklearn, ecosystem
