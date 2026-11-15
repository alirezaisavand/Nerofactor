# E.11F v1 — controlled stronger anisotropy, cached visibility only

**Implementation/freeze only until one explicit execution approval.** Base:
`151d027c4e13808e80589f978b029283ab371c69`. Preserve
[E.11E results](../BELL_QUADRATURE_E11E_RESULTS_V1.md), its executed study and
the original failed E.11D decisions. Do not rerun or request E.11E authorization.
No full dataset, neural material training, new lighting or optimization study.

## One material-only intervention

At the original camera58's84 probes use exactly:

`r=mx/my`, `mx_new=mx*sqrt(r)`, `my_new=my/sqrt(r)`.

Positive ordered float64 widths are required; the transformed truth and all
perturbations must stay in[.04,1]. Reject invalid values rather than clipping,
changing the factor or excluding locations. Geometric-mean width, principal
directions, color/metallic, geometry, cameras, normals, lighting and the additive
BRDF remain fixed. Numerically record product-invariance residue; do not pretend
rounded arithmetic is exact. The geometric mean does not fix every other
roughness statistic or directional reflectance.

Use the unchanged E.11D independent Cartesian kernel, L2 lighting and original
physical counterfactual definitions. Apply the strength transformation to truth
**before** its original±15° orientation /±10% width perturbations. Seven new
conditions: truth,theta+,theta−,major+,major−,minor+,minor−. No rescaling or new
neural decoder. The isotropic eighth column is explicitly reused, not evaluated.

## Exact mathematical equivalence of the reused control

For s=sqrt(mx/my)>0, `(mx*s)*(my/s)=mx*my`. Hence the determinant-matched
isotropic width remains `g=sqrt(original mx*my)`. Its BRDF is independent of
principal-axis orientation. Other inputs, nodes, weights, visibility, color,
metal and outgoing direction are unchanged, so its integral is mathematically
the same as the accepted E.11E isotropic integral.

The experiment explicitly defines the **canonical** invariant g from the original
widths. A naive floating-point recomputation from rounded transformed widths may
differ by a few ULPs. Before any BRDF call, require product and naive-isotropic
width agreement within16*float64-epsilon relative (atol0); record the actual
differences, verify all non-width inputs bitwise and hash the canonical widths.
This is a floating-arithmetic integrity gate, not a new scientific response
tolerance. Do not adjust the requested transformation to force a bitwise product.
Copy accepted E.11E case7 arrays **exactly** at both grids and verify exact byte
equality afterwards. No new original-material or isotropic integration is allowed.

## Reuse and identity gates

Required accepted E.11E report SHA256:
`589f894d653b715a55739aad2e7667ce0105b14eb0119bb5b54e69010f7cc383`.
Load only the explicit original v3/E.11D/E.11E paths recorded in the preparation
manifest. Verify their hashes, all consumed cache records and the source62-file
freeze lineage. No latest/best lookup, checkpoint/camera-pickle loading or cache
regeneration. NPZ reads use `allow_pickle=False`.

Reuse the exact256×256 and512×512 node archives, solid-angle weights,84 probe
IDs and bit-packed visibility. Verify node hashes/domain/weights and packed bit
counts against their accepted records. World directions use the identical
normal/basis arithmetic; require unit/cosine error≤2e−12 and positive hemisphere.
All material axes remain original. **No ray tracer or native model module may
be imported**; no primary/shadow query is permitted. The tracing identity is
recorded as cached provenance, not a claim that the extension was loaded again.

All seven new cases use identical cached visibility. Original E.11E material
controls are reused. New diffuse RGB must agree with its unchanged original
counterpart to the existing float64 equivalence bound2e−10; no substitution of
diffuse results is permitted. Record actual differences. Verify all frozen
source/consumed input hashes again before declaring completion.

## Comparisons and predeclared interpretation

Keep all original gates: each case's max absolute RGB difference≤.003 and
relative L2≤.002 between grids; local finite response≥.003,≥4*local grid
difference and≥2 native sRGB codes. Retain all84 probes/cases and failures. No
threshold/factor/initialization tuning after observing outcomes. Record raw
linear components and original quantization; no clipping, exposure fitting or
renormalization. Invalid reference contributions stop; a completed panel with
failed reference convergence remains inconclusive and triggers no extra grids.

Compare original versus stronger continuous orientation, both widths and
anisotropic/isotropic contrast, code effects, per-probe gate memberships,
reference convergence and conditional Jacobian diagnostics. The inherited
helper's control keys are explicitly renamed to E.11E/original-material terms;
do not label reused quadrature as a new MIS experiment.

Use the existing conditional three-column finite-difference matrix: columns are
half the ±15°/±10% RGB contrasts, with original numerical rank convention
smin>max(1e−6*smax,1e−8). Report singular values, finite condition-number
distributions, column norms and pairwise absolute column cosines, without a new
conditioning acceptance cutoff. This is one-view RGB at fixed color/metal,
not a seven-parameter or spatial-neural identifiability proof.

Report spatial cell IDs/counts using the original16×16 image cells; unmeasured
locations are not interpolated. Include joint orientation+both-width success
coverage. Report the original25% per-factor coverage floors and10-percentage-
point orientation-gain marker as inherited **single-panel screens**. They do not
authorize full128-view construction or replace the original three-view policy.
Even favorable findings need a separately reviewed cross-view/spatial-coverage
step before claiming a full informative benchmark.

For mechanism assessment reuse cached visible fractions and reconstruct the
known diffuse irradiance `pi*L_diffuse/[color*(1-metal)]` from saved components.
Low irradiance/visibility and weak columns are consistent with coverage limits;
nonzero aligned columns/poor singular spectra support local coupling. Positive
diffuse irradiance does not establish useful angular illumination. Both mechanisms
may coexist; this one-light/one-view panel cannot fully separate their causes.
If weak, recommend one focused discriminating next experiment from the measured
evidence, without executing it or starting a lighting/optimizer search.

## Bounds, outputs and execution boundary

Exactly **192,675,840 new BRDF point-directions**:84*(256²+512²)*7.
Exactly27,525,120 environment point-directions, once per reused direction/probe;
zero new shadow/primary queries, original-material BRDF calls, isotropic BRDF
calls, model/gradient/optimizer updates. Reused isotropic data is labeled cached,
not included in the new-work count. No new quadrature nodes or sampling tapes.

Same existing image
`sha256:40024bfedce45251e359d6af63f26bf5a497136d2916d8b06046cc09b5f45331`,
A100 GPU0,4 CPU threads,64GiB host RAM/no swap,25% Torch allocator fraction,
256MiB output cap;4 probes×16384 directions per chunk. One foreground900s
watchdog/30s kill grace, inner840s/20s; estimated1–5 minutes, unmeasured.
Non-root, no network, read-only source/v3/E.11D/E.11E/root; fresh writable output.
No image build/install, permission changes, detached launch, retries or extension.
If Codex lacks Docker access, provide the identical approved researcher-side
foreground command without bypassing restrictions.

Save material widths/canonical mean, both84×8 component arrays with an explicit
reused-case mask, isotropic-equivalence proof/residue, cached input identities,
actual counters, comparison/coverage/coupling metrics, source/runtime/image
identities, resource statistics, console and exit receipt. No original data
is overwritten. Preserve partial arrays with explicit completed-new-probe masks
on failure; zeros in incomplete slots are not measurements. Cleanup must return
before complete status is acknowledged. The rolling progress record remains a
pre-terminal snapshot. Only the run's label-bound container may be cleaned up.

Approval binds the final commit, freeze, fresh host output and exact bounds. It
permits **one E.11F execution**, file-only analysis/figures and evidence commit/
push within this task. It does not approve any full dataset or material training.
