# E.11E v1 — independent midpoint reference check

**Preparation only; one new explicit source/freeze/input/output-bound approval
is required for GPU execution.** Continue from
`398ed4cee5b9d12383379159715af12730b8b95c`. Preserve
[E.11D results](../BELL_HIGHLIGHT_PILOT_E11D_RESULTS_V1.md), its failed candidate
decisions, freeze, raw arrays and all original datasets/artifacts unchanged.
No full images/dataset, training, material model, optimizer or unrelated audit.

## Fixed scientific panel and input admission

Exactly **L2, camera58, the original84 probes and all eight E.11D material cases**:
truth, orientation±15°, original major-axis width×1.1/.9, original minor-axis
width×1.1/.9, and determinant-matched isotropic widths. Original color/metal,
normal, axes, physical equations, surface positions and outgoing views remain
fixed. Do not reorder perturbed axes or select/exclude points based on results.
Use the existing independent Cartesian additive Kurt kernel/counterfactual
functions; no original/corrected native material estimator or A2 shader.

Required source evidence:

- v3 generation SHA256:
  `5035b22b594848900426fd5dfaa8b96382b57a76cfde97864b7ef7591c46e6d7`.
- Completed E.11D native report SHA256:
  `4ad7c79fb63713c5710aa8407c3229fb8bd71341dc81d39e985cb24608a3f1f1`.
- Camera58 L2 MIS capture SHA256:
  `71d64abbaf312c2c2d24a337639e0458b9e3025678c7b3d92494a89dab889797`.

Verify all27 original E.11D capture identities and the consumed v3 mesh, labels,
rays and reference against their pinned inventories. The84 pixel IDs must match
the original spatial selection and source labels exactly. Only primitive numeric
NPZ reads with `allow_pickle=False`; no camera/checkpoint pickle loading. Use
explicit read-only input roots; no latest/best discovery. Verify consumed inputs
and frozen code again before declaring completion. Frozen G/A remain unchanged.

## Quadrature definition and conventions

Two complete grids, m=256 and512, each with m×m **midpoints**:

`z_i=(i+.5)/m`, `phi_j=2*pi*(j+.5)/m`,
`local=(sqrt(1-z²)*cos(phi),sqrt(1-z²)*sin(phi),z)`.

z=cos(theta)∈(0,1), phi∈(0,2pi), and dOmega=dz*dphi. Each node has solid-angle
weight **2*pi/m²**, corresponding to equal-area density1/(2pi). Integrate by
**sum** of `f*Li*(n.wi)*visibility*dOmega`; there is no extra sin(theta), mixture
PDF or second division by sample count. There are no horizon endpoints, accepted-
ray redraws or missing-domain normalization. These are deterministic cubature
grids, not independent random samples; they are not nested.

Map nodes to the original normal with the accepted independent reference basis:
t=normalize(n×least-aligned-coordinate-axis), b=n×t. This defines quadrature
azimuth; **material axes remain the original stored T/B/theta**. World directions
are float64; require unit-length and n.wi versus local-z error≤2e−12 and n.wi>0.
No new primary camera rays, normal flip, mesh transform or label changes.

Native visibility preserves E.11D exactly: float32 `wi`, origin
`p.float()+1e-5*wi.float()`, corrected CUDA BVH, finite returned depth and
**depth>=10 visible-environment** policy. Do not replace this inherited cutoff
with an infinite-occluder convention. World units are not established meters.
Reuse one shadow-query set across all eight material cases. BRDF/environment,
weights and accumulation remain float64; accumulate4 probes×16384 directions
at a time. This changes the quadrature only, not the physical model.

## Analytic and synthetic admission controls

Before any real query, run tiny CPU controls at m=4,8,32:
solid-angle area2pi (abs≤2e−13), cosine integralpi (abs≤2e−14), unit directions
(abs≤2e−14), and the **exact midpoint** z² integral
`2*pi*(1/3-1/(12*m²))` (abs≤2e−14). This explicitly retains finite-grid bias.
The control tolerances concern floating-point arithmetic, not the study's RGB
accuracy thresholds.

Focused synthetic tests additionally verify rotated frame moments, the actual
weighted accumulator with analytic Lambert/constant and directional illumination,
the absence of an extra sine/mean factor, analytic-plane visibility with the
original offset/depth cutoff, nonfinite depth rejection and packed-mask ordering.
These are small CPU stand-ins, not Bell rendering, native GPU admission or tests
of absolute accuracy of the real integrals. A preliminary rotated-moment test
used a continuum bound at its floating-point boundary; its preserved failure was
replaced with the stronger exact rotated **discrete** moment check. No old
scientific threshold was changed.

## Comparisons, frozen response criteria and uncertainty

Save diffuse/specular/total linear RGB for84×8 cases at both resolutions. Compare:
512-grid versus256-grid; both against saved16384-direction MIS;512 against saved
4096-direction MIS. Report max absolute/relative L2, RGB RMSE and signed RGB means,
including separate diffuse/specular differences. Preserve all cases/results.

For **each of eight cases**, retain the existing grid-convergence requirements:
max absolute RGB≤.003 **and** relative L2≤.002 across84 probes. The follow-up can
describe its response panel as resolved under this check only if all eight pass.
If any fail, finish only the already approved two-grid panel, preserve uncertainty
and stop; no extra grid, tape, lighting or threshold revision.

Recompute the original finite-step orientation/width/isotropic response metrics
under the new grids, with original criteria: max-channel truth response≥.003,
response≥4×the larger local fine/coarse difference, and≥2 native sRGB codes.
Report continuous responses and code changes as well as pass fractions; keep
the conditional three-column Jacobian interpretation. RGB outside[0,1] remains
raw and explicitly outside the PNG domain; do not clip or rescale it. Full arrays
permit a file-only spatial/probe analysis after execution without more queries.

**E.11D remains FIX_AND_PROCEED with neither light admitted**, regardless of this
outcome. One camera cannot satisfy a three-view benchmark criterion. Agreement
of two deterministic grids does not prove absolute accuracy: grids share kernel,
geometry, visibility implementation and basis choices and can share errors.
Differences from one fixed MIS tape may indicate finite quadrature/proposal error;
signed means alone cannot establish statistical sampling bias. No independent-
tape variance or global material identifiability claim is authorized.

If both new grids agree and the width responses remain weak, propose a **separate
roughness-anisotropy-only factor**, keeping lighting/color/metal fixed. Do not
add a metallic change or another lighting search to this study. If references
remain unresolved, report that blocker rather than infer physical ambiguity.

## Bounded execution, provenance and cleanup

Exactly84*(256²+512²)=**27,525,120** shadow directions and eight times that,
**220,200,960** BRDF point-directions. Require matching attempted/returned counts;
zero new primary rays, native material models/shaders, gradients or optimizer
updates. No new MIS tapes or original experiment repetition.

Use existing corrected image
`sha256:40024bfedce45251e359d6af63f26bf5a497136d2916d8b06046cc09b5f45331`,
sidecar SHA256
`800606e1544c42f3e750978d0cce747c5b69c15996391a8709fdc3669274f3e5`.
One A100 GPU0,4 CPU threads,64GiB host memory/no swap,25% Torch CUDA allocator
fraction,256MiB output ceiling. One foreground900-second watchdog/30-second kill
grace, inner840s/20s. Estimated1–5 minutes, **unmeasured**, not an execution result.
No image build/install, permission changes, detached launches, retries or extension.

The companion preparation manifest specifies fresh host output/authorization
paths. `experiments/run_bell_quadrature_e11e.sh` checks a clean committed source,
immutable image and exact authorization. Mount source, v3 package and E.11D
captures read-only; output separately writable; non-root, no network, read-only
root filesystem. Remove only the run's own label-bound container on termination.
If Codex cannot access Docker/GPU, report the restriction and supply the identical
approved researcher-side foreground command without bypassing permissions.

Archive nodes/weights/PDF plus their byte hashes, bit-packed visibility masks
(probe-major, C-order directions, big-endian bit order) and per-probe visible
counts. Archive full component arrays/probe IDs for both grids, comparison JSON,
source/input hashes, image/backend/runtime identity, actual counters, memory,
elapsed time, console/exit receipt and terminal `quadrature.json`. A rolling
progress record is intentionally incomplete until finalization. On failure save
completed-probe partial arrays with an explicit validity mask; zero uncompleted
slots are not measurements. Cleanup must return before a complete status is
written; missing/failed cleanup never becomes an affirmative claim.

Approval covers **one** launch and subsequent file-only analysis/figures/report,
commit and push. A subsequent benchmark or material-factor study needs separate
approval. Scientific follow-up results remain separate from E.11D's original
failed acceptance decision and never silently replace its evidence.
