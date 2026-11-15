# E.11 — Bell oracle benchmark and native admission

**Preparation ready; real generation and native GPU admission pending.** No new
Bell images or training results exist yet. The [frozen generation definition](protocols/BELL_ORACLE_GENERATION_E11_V1.md)
implements the accepted E.10 additive-family proposal, independently of the native
corrected shader. The mesh is a benchmark-defined oracle, not verified real GT.

The new entry point is `generate_bell_benchmark_e11.py`. Its default path only
checks input hashes and decodes bounded numeric calibration data; it does not
import Torch/CUDA or render. `--execute` additionally requires an explicit
hash-bound authorization, the reviewed isolated image and a fresh `/outputs`
child. A geometry/convergence failure stops generation and preserves partial files.
No automatic retry, mesh correction, parameter tuning or threshold change exists.

Implemented checks cover safe calibration parsing; original A pinhole/normal
conventions; all128 silhouettes and64 independent CPU triangle intersections;
float64 Cartesian BRDF/MIS reference; smooth world-Q/material labels; fixed
coarse/fine convergence; known illumination and six unseen-light captures;
linear/sRGB/depth/mask consistency; input/source hashes, work caps and cleanup.
The offline verifier recomputes labels/encoding/convergence from saved arrays,
without rendering. Hash equality establishes identity, not physical correctness.

## First native comparison (specified, not launched)

| Arm | Source selection | Target / observation |
|---|---|---|
|A_original|original A equations, estimator and decoder; explicit known-light control|same additive Bell targets|
|A_estimator_v1|E.10 estimator corrections; original decoder|same additive Bell targets|
|A_corrected_core_v1|E.10 estimator/atan2/numerical corrections|same additive Bell targets|

All three use the same immutable oracle mesh, camera IDs,512 fitting rays,
512 diffuse/1024 specular samples, initial seed6033/network capacity, known
world environment, zero indirect/human light, priors off, original sRGB
Charbonnier, original Adam(eps1e−4,AMSGrad,betas.9/.999), clipping.7 and
1000-step warm-up/hard100000 cosine semantics. The existing configs specify2000
updates, validation/save at2000, but **none of those runs is authorized**.
No held-view or relighting result selects an initialization or checkpoint.
The distinct A_energy_budget_v1 physical family is not admitted to this panel.

Native construction/gradient admission comes first: C, exactly two successful
original Adam returns at steps0/1. Step0 has zero LR and may change Adam state
without weight displacement; step1 must change parameters. Existing
`CorrectedProbe` verifies original loss/anisotropic paths and finite gradients;
known-light/zero-bounce learned illumination groups are intentionally inactive.
The existing two-update observer forbids validation, checkpoint saving and NVS.
Do not add extra scientific queries to this smoke silently.

Static interface review: each known-light arm registers a behavior/source/config
signature; strict checkpoint loading rejects wrong variants/keys/shapes/dtypes.
`CorrectedRenderer.predict_materials` exports linear widths,metallic,albedo,
double-angle rotation and chart-local Q at mesh vertices. Such vertex Q must be
mapped through the **same surface chart/normals** before comparison to per-pixel
world-Q labels; interpolating chart angles is invalid. `relight` accepts an
explicit callback/identity and restores the lighting override on exit while
retaining native visibility. These interfaces have not been executed on this
new dataset. Checkpoint roundtrip/full export/relighting admission is not proved
by a two-update smoke that intentionally stops before those operations.

For the eventual comparison retain native validation PSNR/SSIM for continuity,
and report independent GT-mask/full-frame linear MSE/relative L2 and sRGB metrics,
held-view and unseen-light errors, diffuse/specular decomposition, world-Q
absolute/relative errors, sorted widths, eligible orientation errors, color and
metallic. Keep E.10 recovery bounds unchanged: relative Q≤.05,eligible angle≤5°,
max RGB albedo≤.02,metallic≤.03,held-out relative radiance L2≤.01. Original-A's
nominal Q does not certify its faulty decoded frame. Truth near-isotropic pixels
remain in tensor/axis/color/metallic metrics with explicit angular eligibility;
retain original pass flags and report their limitations. No aggregate claim or
new per-object success criterion is inferred from implementation tests.

## Commands and the approval boundary

Run new tests only:

```bash
cd /home/isavand/paper/Nerofactor-3d2-e2a
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider tests/benchmarks/test_bell_e11.py
bash -n benchmarks/run_bell_e11.sh
```

After explicit generation approval, use this in a foreground tmux terminal.
The authorization file must bind the final committed source, exact freeze hash,
fresh host batch and limits; it cannot be replaced by a command-line yes flag.
`E11_BATCH` and `E11_AUTH` below refer to the **approved** paths, not discovered
latest outputs. This command does not build or modify the image.

```bash
export E11_SOURCE=/home/isavand/paper/Nerofactor-3d2-e2a
export E11_CAMERAS=/home/isavand/paper/data/GlossySynthetic/bell
export E11_MESH=/home/isavand/paper/final_results/meshes-pretrained/syn/bell.ply
export E11_BATCH=/home/isavand/paper/evidence/APPROVED_FRESH_BELL_BATCH
export E11_AUTH=/home/isavand/paper/evidence/APPROVED_BELL_AUTHORIZATION.json
export E11_GPU=0
bash "$E11_SOURCE/benchmarks/run_bell_e11.sh"
```

Then compute the generation report hash and run the offline verifier into a new
file (do not overwrite any failed verification). The host has the CPU dependencies;
no model/GPU query occurs:

```bash
E11_PACKAGE="$E11_BATCH/benchmark"
E11_REPORT_SHA=$(sha256sum "$E11_PACKAGE/geometry/generation.json" | cut -d' ' -f1)
python3 -B -m benchmarks.verify_bell_e11 --root "$E11_PACKAGE" \
  --report-sha256 "$E11_REPORT_SHA" --output "$E11_BATCH/offline-verification-v1.json"
```

Before native execution, independently review the real reports and request a
separate approval binding the dataset hashes, final source identities, reviewed
image, fresh native run and exactly two original Adam returns. Only that review
may create a new affirmative oracle-admission.json; preserve the candidate.
Reuse [E.10's native command](protocols/PAPER_ANISOTROPIC_BENCHMARK_E10_V1.md#admission-and-the-smallest-native-launch)
with the newly admitted dataset/geometry paths (the command is in its "Admission
and the smallest native launch" section). Do not execute it with a candidate
manifest or automatically extend it to2000 steps.

## Remaining scientific limitations

Real calibration/mesh alignment, surface normal orientation and reference
convergence have not yet passed on this oracle. Narrow geometry visibility at
faceted silhouettes, grazing incidence and QMC error may cause a legitimate
mandatory failure. Two-grid agreement is not a rigorous error bound or an
independent replication; additive Kurt is not globally passive. The saved energy
measurements must accompany the interpretation. Native corrected construction,
backward performance and real parameter displacement are unverified on these
new inputs. No corrected-reconstruction, physical-recovery or novelty claim is
justified yet. The smallest next experiment is the **one frozen dataset
render**, followed only if admitted by the separately approved two-update smoke.
