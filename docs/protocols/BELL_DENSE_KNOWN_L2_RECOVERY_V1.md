# Dense Bell known-illumination recovery and method decision

Use the hash-bound 18-view Dense Bell v2 reference. The admission candidate
is pending. After explicit researcher admission, write its deterministic
accepted record separately in the evidence directory. That record admits
only this dataset and carries no training authorization. A separate exact
authorization binds the accepted record, code commit, freeze, image,
and fresh output directory to one foreground training launch.
[Reference verification](../results/bell_dense_reference_v2/README.md) ·
[Admission candidate](../evidence/BELL_DENSE_KNOWN_L2_ADMISSION_CANDIDATE_V1.json)

## Matched histories

Run K1-Dense, K2-Dense and K3-Dense in order, for 10,000 Adam returns each.
Use the accepted K1, K2 and K3 recipes for the corresponding dense arms.
The only training intervention is the fitting acquisition. The dense
pool concatenates the four original and twelve added fitting views in
frozen panel order, then retains their original valid-pixel order. The
training step uses positions, normals, outgoing directions and the total
observed linear RGB from the existing component array. The same held-out cameras 105 and 114 and the same
reference, initialization and paired-chart/ambient gates apply. Preserve
known analytic L2, zero indirect illumination, the original loss and
fixed legacy priors, seed 6033, 512 fitting observations per update,
the unchanged Adam settings and learning-rate schedule, render sampling,
and the original transported-Q domain, coefficient and stream.

Each arm keeps the uninterrupted seeded NumPy permutation with incomplete
tails discarded. Verify the indices the existing trainer actually passes
to its position lookup, one batch at a time, against an independently
replayed stream. Save the actual little-endian uint32 index arrays as a
sidecar and require their exact common SHA-256 at completion. The observer
returns the ordinary NumPy selection and makes no model, random, or
optimizer calls. Fitting-camera surface-count metadata is recomputed for
dense coverage. The geometry-defined sampler and loss do not use it.

The fitting pool grows from 17,573 to 65,419 observations. Each run still
uses 5,120,000 fitting slots, about 291.36 versus 78.26 effective pool
passes. Report this exposure difference alongside the density, highlight
and angle differences. The diagnostic recordings increase the evaluation
work at the same six fixed states (0 through 10,000, spaced by 2,000).
Record fitted and held-out maps and checkpoint archives with the existing
evaluation pathway. Retain the established native lighting and
initialization checks, including zero learned-light calls or optimizer
entries. Stop on the first invalid gate or missing terminal evidence.

| Native work | K1-Dense | K2-Dense | K3-Dense |
| --- | ---: | ---: | ---: |
| Adam returns | 10,000 | 10,000 | 10,000 |
| Shader calls | 17,184 | 17,186 | 17,186 |
| Shadow-ray slots | 13,210,853,376 | 13,210,927,104 | 13,210,927,104 |
| Component-gradient reverse sweeps | 30 | 36 | 36 |
| Geometry-only surface pairs | 0 | 1,280,000 | 1,280,000 |
| Diagnostic boundary predictor calls | 12 | 12 | 12 |

Use the reviewed corrected image on A100 GPU 0, four CPU threads, 64 GiB
RAM without swap, 128 PIDs, the 25% CUDA allocator fraction and 8 GiB
output per arm. The foreground timeout is 11,160 seconds with 30 seconds
of kill grace. Per-arm host and container limits are 3,600 and 3,540
seconds, with 30 and 20 seconds of kill grace. Keep root, source and
inputs read-only, run as a non-root UID and disable networking.
No launch retry, budget extension, historical rerun, or new reference
rendering is authorized.

## Evaluation and final method decision

Use the 10,000-update endpoints. Compare each dense arm with its own
sparse counterpart on identical held-out and original-fitting
observations. Report both observation-weighted and observed-triangle-area
metrics. Report Q median and p95, width MAE and signed bias, orientation
on the common eligible subset and eligibility counts, boundary-band
errors and signed boundary jumps on the frozen probes, fitting and
held-out RGB, and width saturation. Keep both sparse and dense support
and highlight strata on the original held-out points. A zero-count
cohort has no estimate. Do not select an earlier checkpoint to improve
the endpoint ranking. A density effect here jointly changes angular
coverage, fitting pool and average exposure.

Evaluate whether the signed K3-minus-K1 held-out Q gap shrinks or
closes. Consult Q tails and RGB, both width errors, angle eligibility,
boundary behavior and low-support strata alongside the median.
The matched K2/K3 pair isolates chart versus ambient prediction within
its common tensor recipe. K1/K3 differ in other parts of the recipe.
If ambient gains carry through the matched pair and the K3 dense recipe
is competitive with K1 across these metrics without a clear robustness
regression, promote ambient prediction to final validation as a
candidate. Otherwise, the smallest further representation-versus-recipe
control is a matched chart/ambient pair sharing a common physical
initialization and the same K1-style objective where possible, retaining the accepted legacy
priors, omitting the added spatial term in both arms, and sharing the
tensor evaluator and sampling. Freeze
that separate control only if the results warrant it. Do not launch it
under this protocol.

Method readiness still requires repeated seeds and an additional
acquisition or scene for robustness, matched oracle-versus-reconstructed
geometry to measure sensitivity to normal and visibility error, held-out
illumination with linear radiometric evaluation for relighting, and an
external baseline using the same observations, masks and compute
accounting. A favorable one-seed known-light result supports a candidate
decision and does not settle those requirements.
