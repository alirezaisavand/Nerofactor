# Saved sphere regularization evidence

File-only analysis of the interrupted corrected-sphere v2 run. See
[the scientific assessment](../../SPHERE_REGULARIZATION_AUDIT_V1.md) and
[external identities](external-artifacts.json). Originals remain under
`/home/isavand/paper/evidence/sphere-regularization-audit-v1` and the source run
listed in that manifest. Copies were SHA-256 compared with originals.

- `audit.json`: all20 saved validation observations, checkpoint hashes,24 small
  CPU reflectance checks and limitations.17 checks satisfy the declared two-grid
  bound;7 are unresolved. This is not native shader validation.
- `progression.csv/png`: material, effective diffuse illumination and reconstructed
  prior diagnostics. Local physical partials are **not** old neural gradients.
- `saved-map-progression.png`: existing RGB/material/component observations.
- `logged-losses.png`: actual archived aggregate training losses.
- `world-Q-diagnostics.png`: chart, unoriented-world and tensor diagnostics; no GT.
- `illumination-clipping.csv`: foreground fractions of clipped display-light
  channels. Clip fractions were read directly from saved `diffuse_light` arrays,
  not inferred from PNGs.
- `surface-pair-preflight.json`: CPU mesh inspection for the proposed loss domain.

No checkpoint or large numerical archive is stored here. The source run remains
incomplete at41427 optimizer returns, with a40000-step last completed checkpoint
and unverified cleanup. Do not relabel it successful.

The analysis code is `experiments/analyze_sphere_regularization.py` and
`experiments/visualize_sphere_q_audit.py`. They read NPZ with `allow_pickle=False`.
To regenerate in a **fresh** directory, without constructing the native model:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
timeout --signal=TERM --kill-after=10s 300s python3 -B \
  experiments/analyze_sphere_regularization.py \
  --batch /home/isavand/paper/evidence/sphere-core-v2-ec8cee6b6d234133b0c7e1168fff02e1 \
  --cameras /home/isavand/nero/data/nerf_synthetic/sphere/transforms_train.json \
  --output /ABSOLUTE/NEW/AUDIT
python3 -B experiments/visualize_sphere_q_audit.py \
  --audit /ABSOLUTE/NEW/AUDIT/audit.json \
  --output /ABSOLUTE/NEW/AUDIT/world-Q-diagnostics.png
```

The original audit was not rerun after adding the explicit zero-allocation input
guard; a separate saved-array check established strictly positive allocation in
all20 archives (minimum1.487e-32). No numerical calculation or saved result changed.
The exact originally executed uncommitted script bytes were not archived; current
code hashes identify the reviewable regeneration implementation, not a recovered
historical execution source identity.

The clipping CSV can be independently reproduced by reading each archive's
`hit_mask` and computing the per-channel fraction of `diffuse_light[hit_mask]>=1`,
plus the fractions with any/all channels clipped. No shader queries are needed.
