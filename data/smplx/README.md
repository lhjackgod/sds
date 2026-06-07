# SMPL-X Assets

Place manually downloaded licensed source assets under `source/`:

```text
source/
  models/
    smplx/
      SMPLX_NEUTRAL.npz
  smplx_uv_template.obj
```

Then run `src/generate_smplx_template.py`. It writes the usable template OBJ,
UV NPZ, and normalized per-vertex labels under `generated/`.

SMPL-X model parameters and the official UV template are not redistributed by
this repository.
