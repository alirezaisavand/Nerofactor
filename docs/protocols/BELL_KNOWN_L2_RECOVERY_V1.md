# Heterogeneous Bell known-illumination recovery

The batch isolates how access to the admitted analytic L2 environment
affects material recovery on the existing heterogeneous Bell benchmark.
The original fitted and held-out views and their visibility remain fixed.
The intervention turns off illumination estimation. It provides no oracle
material supervision to the fitting objective.

| Arm | Completed learned-light control | Material and spatial objective |
| --- | --- | --- |
| K1 | D2 | Corrected chart renderer and original legacy priors |
| K2 | D8-C | Chart prediction, shared D8 tensor renderer, retained D8 priors and original transported Q term |
| K3 | D8-T | Ambient prediction, shared D8 tensor renderer, retained D8 priors and original transported Q term |

Keep the exact 10,000-update recipes, initialized material fields and shared
scalar heads, original seed 6033, 512 training rays, fixed photometric
sampling policy, optimizer settings, schedule and clipping. Keep D8-C and
D8-T's initial width, common world Gaussian tapes, diffuse samples, two-arm
initial parity gate and existing uncapped surface weights. The cap and
pseudo-Huber from other Bell experiments are outside this comparison. Start
each arm in a fresh process with its normal initialization. The learned
archives are used for metadata and saved-array comparisons.

## Analytic light and legacy prior semantics

Use the admitted L2 ambient and six lobes through the existing
corrected-core known environment route. The native BVH still controls
visibility. The existing zero-bounce branch returns zero on mesh
occlusions, and human light remains disabled. Both the corrected chart
renderer and shared tensor renderer call the same native light dispatch.
Neither shader math nor analytic illumination changes.

The trainer checks the known route in the no-gradient compatibility
probe, in both sides of the D8 initial pair, in camera training and
evaluation, and in each environment export. The observer counts actual
analytic calls and rejects a learned-module call or overridden environment.
A dispatch with no visible rays can have zero outer light calls. The
illumination-map path must use the analytic function on all six exports.
Each export must agree with the independent L2 grid evaluation within
absolute 2e-5 and relative L2 2e-6, the float32 agreement bounds. In the
accepted Donut D1 known-light archive the same two map paths agreed exactly.
The chart and tensor native/reference compatibility probes stay unchanged.
Those no-gradient diagnostic probes use 24 oracle materials as they did
in the learned controls. Only geometry and observed linear RGB reach
the fitting pool.

Construct the original Adam with the same hyperparameters and schedule
using material parameters alone. Keep the unused light-module weights
in the state dictionary for strict historical initialization and checkpoint
parity. At every existing export, verify unchanged light parameter and
buffer hashes and zero illumination entries in the optimizer groups and
state. Report zero light gradients and fail if the learned modules
are invoked.

Keep the legacy diffuse-white weight .01 in the optimizer's objective.
Under fixed L2 it measures the chromaticity of sampled visible diffuse
light, so its value may vary with the batch and samples. Diffuse
directions depend on the fixed normals and sampled azimuth, and the
visibility geometry is fixed. The term has zero derivative with respect
to each fitted material parameter and cannot whiten the fixed environment.
Verify that derivative on the six existing gradient-diagnostic batches.
The energy prior continues to use the Fresnel/cosine proxy and additive
diffuse allocation. The specular prior continues to use the detached
NDF weights and diffuse allocation. Neither proxy explicitly contains
the queried incident radiance. Their values and gradients can still
change as fitted material predictions change under L2.

## Fixed evaluation

Save initialization, evaluations and original checkpoint outputs at the
existing cadence of 2,000 updates. Use the fixed 10,000-update endpoint
and admitted fitting cameras 0, 4, 19, 2 and held-out cameras 105, 114.
Compare each K arm with its corresponding learned arm and compare K2 with
K3 and K1 with K3. Replay exact initial material predictions, native
reference-gate arrays and learning-rate sequences. Report held relative
Q, signed bias and MAE for sorted widths, scalar material and RGB errors,
and orientation on matched eligible points. Report observed-area weights,
boundary-band and complete fixed same-face probe jumps, near- and exactly
isotropic strata, and cohorts with little fitting or highlight support.
Retain the preclip joint and per-term gradient diagnostics and time/cost.
Assess reduced illumination ambiguity separately from identifiability
under sparse viewing, weak specular signal and the fixed optimizer budget.

## Bounded execution

Use one foreground batch in order K1, K2 and K3. The ceilings are 30,000
Adam updates, 37,444 shader calls, 29,049,937,920 shadow rays, 102 component
gradient probes, 2,560,000 sampled surface pairs and 36 boundary predictor
calls covering 89,064 points. The only added diagnostic work relative to
the matched histories is six diffuse-white gradient probes per arm
and fixed-light routing, map and state checks. Use the already reviewed
corrected Docker image on A100 GPU 0, four CPU threads, 64 GiB host memory
without swap, 128 PIDs, 25% CUDA allocator limit and 8 GiB outputs per arm.
Use 11,160 seconds in the foreground with 30 seconds of kill grace and
the original per-arm 3,600-second host and 3,540-second container limits.
Require read-only inputs, source and container root, non-root execution
and no network.

Stop at the first mandatory failure and preserve partial evidence.
This approval includes no historical rerun, new reference, coefficient
sweep, or localized photometric arm. Denser multiview acquisition,
variance-reduced photometric optimization and reconstructed-geometry
validation remain in the research backlog.
