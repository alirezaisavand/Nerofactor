# 3B-2: controlled anisotropic estimator ablation v1

The single authorized CPU experiment completed all 25 parameter/view cases.
B1 removes the accepted-count specular bias: **71/71 converged specular
comparisons** satisfy the frozen uncertainty budget, versus 13/71 for B0.
B2 corrects the diffuse generating measure, while the prescribed finite height
pattern retains a measurable deterministic integration error. B3 combines both
corrections; it does not ensure energy conservation, repair angle decoding, or
establish improved reconstruction.

## Scope, evidence and identities

Base: `759a1253bf1db207221d6989ec4ca6f9c3ae3225`.
Branch: `feat/anisotropic-estimator-ablation-v1`.
Frozen material A: `e1e859779301faced46d6476691e20b546c27ae0`.
Frozen `network/field.py` SHA-256:
`b8386007dc009442cdefdeccc3c6ecc0114f6d0e7feaf804008b0bd9020091fe`.

- [Pre-execution protocol](ANISOTROPIC_ESTIMATOR_ABLATION_V1_PROTOCOL.md),
  SHA-256 `330e6daf67a34aa268fa7db3aaa50a0ae6977d69f0a2bcb62761d910b6931655`.
- [Complete machine-readable observations](evidence/ANISOTROPIC_ESTIMATOR_ABLATION_V1.json),
  SHA-256 `08f47a373ff25edaa95971df705f9288c2fb92eac0160d70e7569c6c9b10bc1f`,
  2,329,958 bytes. This is the only full 3B-2 run; it was not retried.
- [Experimental operators](../experiments/anisotropic_estimator_v1.py),
  SHA-256 `e6560f646011095a3b029db5aa03ea4bbd72bb78917be6eeaad3abdd4c911986`.
- [Fixed experiment driver](../experiments/run_anisotropic_ablation_v1.py),
  SHA-256 `fc1f5757171993e5becaa165560838e382909e8e6d7fa55c9d945d78c8f83ab8`.
- Independent Cartesian reference algebra is reused from
  [the accepted audit](../audits/anisotropic_brdf.py), SHA-256
  `7706a54ee2e3c4a5568f297b75d4b9de1e979ab0d35483724e596bcfc7245268`.
  Its old experiment and test suite were not rerun.

The JSON records source-method AST hashes, sample construction/stream identities,
settings, raw seed observations, references, every component comparison, paired
uncertainties, acceptance/counts, decoder diagnostics, timings and budgets.
The source closure is manifest-verified without scientific imports. Pure method
bodies execute in plain CPU objects; no neural network is constructed. Frozen
G/A, legacy YAML, historical artifacts and accepted evidence remain unchanged.
This is a new analytical experiment, not a native renderer/GPU equivalence claim.

## Formula changes and baseline fidelity

For sample contribution X_j, original reflected-direction mask I_j and total M:

- B0: specular estimate = sum(I_j X_j)/max(sum(I_j),1).
- B1: specular estimate = sum(I_j X_j)/M; rejected samples contribute zero.
- B2: B0 specular, with diffuse coefficients cos(theta_i)=sqrt(z),
  sin(theta_i)=sqrt(1-z), z=j/512, j=0,...,511.
- B3: B1 specular and B2 diffuse.

B0's pure functions reproduce a separately extracted original A fixture
bitwise. B1 changes exactly one hash-admitted AST denominator in compute_radiance;
all other statements remain original. In the ideal valid-frame domain, X_j is
F*ci/ch times the same incoming radiance. The sampler is Beckmann despite its
GGX method name. The intended a=0 specular kernel remains
f_s=F*D(h)/(4*(wo dot h)); no Smith G is introduced. Conditional normalization
instead tends to the integral divided by view-dependent acceptance P(wo).
Fixed M restores the unconditioned valid-domain expectation, subject to the
original finite-precision/clamp behavior and valid orthonormal frames.

B2 recovers exact heights from original sample_sphere integer indices. The
stored elevation is e=1-2*arcsin(z)/pi, not z. Double-precision construction
recovery error is 5.55e-17; inverting rounded stored e would introduce up to
4.61e-8 height error. Stored azimuths, basis construction, counts, random azimuth
shift and draw order are preserved. Only the corrected square roots omit the
old added 1e-7, following the approved exact mapping. B0/B1 keep that perturbation.
No jitter, half-bin shift, normal repair or new random number is added.

All arms retain original Fresnel, albedo/metallic terms, masks, weights, returned
PDF and angle decoder. Metallic=.001, RGB albedo=.999 gives d=.998001 and
F0=.040959. All lights and channels are achromatic. Legacy returned f_s_sum and
L_spec diagnostics remain unchanged, and are not used for optimization here.
Any future training variant must explicitly review their relationship to the
corrected estimator rather than silently changing energy/specular regularization.

## Design and uncertainty

Five roughness pairs (.04,.04),(.1,.6),(.3,.7),(.6,.9),(1,1) cross outgoing
polar angles 0,30,60,80,89 degrees, azimuth zero and theta=.31 radians. Normal
is +Z; source algebra is float32 and independent integration is float64.
Each of 16 private seeds supplies identical paired sample tapes to all four
arms and all cases: 1024 specular samples and 512 fixed diffuse heights with
one random azimuth shift. Global NumPy/Torch CPU RNG states and input tapes
were verified unchanged. Cross-case reuse introduces correlation; descriptive
case counts are not independent significance tests.

The three fixed lights are 1, max(ci,0)^2, and
.1+.9*max(dot(wi,(1,1,1)/sqrt(3)),0)^8. References integrate Lambertian and
independent Cartesian Kurt/Beckmann kernels on exactly 256x512 and 512x1024
hemisphere grids. Absolute two-grid agreement must be <=2e-5 per component;
a total also requires converged components. Agreement is an operational check,
not a rigorous integration error bound. No grid or threshold was changed.

SE=sqrt(sample_variance(ddof=1)/16); nominal 95% mean intervals use
2.131449545559323*SE. The frozen comparison budget is
3*SE+2e-5+16*float32_epsilon. Bias is the mean estimate minus the converged fine
reference, not a population-bias proof. Relative bias is null near zero;
bias/accuracy fields are null when the reference is unresolved. Paired
comparisons use seed-level differences, retaining their covariance.

## Quantitative findings

Counts below are within/outside/unresolved, across 75 light/case combinations.

| Arm | Diffuse | Specular | Total |
| --- | --- | --- | --- |
| B0 | 25 / 50 / 0 | 13 / 58 / 4 | 4 / 67 / 4 |
| B1 | 25 / 50 / 0 | 71 / 0 / 4 | 23 / 48 / 4 |
| B2 | 50 / 25 / 0 | 13 / 58 / 4 | 10 / 61 / 4 |
| B3 | 50 / 25 / 0 | 71 / 0 / 4 | 47 / 24 / 4 |

Mean absolute bias of the 16-seed mean, restricted to converged references:

| Arm | Diffuse (75) | Specular (71) | Total (71) |
| --- | --- | --- | --- |
| B0 | .0502759080 | .00720129235 | .0542210786 |
| B1 | .0502759080 | .00004088650 | .0510093664 |
| B2 | .00033234258 | .00720129235 | .00700166502 |
| B3 | .00033234258 | .00004088650 | .00036906756 |

These aggregate errors describe this fixed grid, coefficient choice and paired
streams. They are not an uncertainty estimate or a reconstruction score.

At (mx,my)=(.6,.9), outgoing 80 degrees (values are mean +/- SE):

| Quantity | Converged reference | B0 | Relevant correction |
| --- | --- | --- | --- |
| Specular, constant light | .03289085190 | .06019268231 +/- .00022220949 | B1 .03290128626 +/- .00029303555 |
| Diffuse, cos-squared light | .49900050000 | .36169523001 +/- 0 | B2 .49802589417 +/- 0 |
| Total, directional lobe | .20827921812 | .22516339459 +/- .00005285983 | B3 .20824171044 +/- .00004464505 |

At that same case, paired seed-level absolute-error changes (corrected minus
B0; negative means improvement) are:

| Comparison | Component/light | Mean +/- SE | Nominal 95% interval |
| --- | --- | --- | --- |
| B1-B0 | specular / constant | -.02638053441 +/- .00030109640 | [-.02702230620, -.02573876263] |
| B2-B0 | diffuse / cos-squared | -.13633066416 +/- 0 | [-.13633066416, -.13633066416] |
| B3-B0 | total / lobe | -.01673071841 +/- .00005101924 | [-.01683946334, -.01662197347] |

The JSON also provides paired estimate/signed-error differences and every
case's variance/interval; these differ from absolute-error changes.

### Diffuse finite-pattern limitation

All 25 corrected cos-squared diffuse estimates fall outside the frozen bound.
The common value is .498025894165, reference .499000499999657, signed bias
-.000974605835, and azimuth-seed SE zero. This matches the **predeclared**
left-endpoint height error -d/1024. The incorrect source mapping's corresponding
bias is -.137305269992. Zero SE cannot measure deterministic radial error.
All 25 corrected lobe diffuse cases meet their uncertainty budget, compared to
zero for B0; constant light matches all arms and cannot distinguish the mapping.
All 24 converged B3 total failures are cos-squared cases; the 25th such total
is unresolved because its specular reference fails. Thus H3 is supported as a
component-estimator derivation, not as universal finite-pattern agreement.
The error is retained; no sample/tolerance adjustment or rerun occurred.

### Unresolved fixed-grid references

| Roughness | View | Light | Specular absolute grid delta |
| --- | --- | --- | --- |
| (.04,.04) | 89 degrees | constant | .0121669020982 |
| (.04,.04) | 89 degrees | cos-squared | .0000690052438 |
| (.04,.04) | 89 degrees | lobe | .00121669020982 |
| (.1,.6) | 89 degrees | constant | .000197737741404 |

Specular and total are unresolved for these four lighting cases, for every arm.
Diffuse remains converged. No fine-grid number there is admitted as ground truth.
For (.1,.6),89 degrees the lobe delta is .0000197737741404, just inside the
predeclared threshold; the report retains its full value without tightening or
loosening the criterion. Any improved reference study requires separate review.

### Acceptance, energy and reciprocity

Per-seed accepted/rejected counts are identical across all arms. Independent
acceptance ranges from .5024617258 to 1 across the grid; the maximum discrepancy
between measured seed-mean acceptance and its slope-domain reference is
.00451653420. For (.6,.9),80 degrees P=.5455377071 and measured P=.5465698242,
explaining the large conditional-normalization inflation in B0.

Under unit illumination, all 25 cases in every arm have nominal mean CI95
entirely above 1. B3's energy means range **1.0113966465 to 1.0663488209**;
its smallest CI lower bound is 1.0111934416. B0 means range 1.0191961899 to
1.0942409933. These analytical coefficient counterexamples verify that changing
estimators does not impose physical energy allocation. Unresolved quadrature
does not erase the measured energy observations; no accuracy claim against
those unconverged references is made.

For the ideal continuum effective kernels, the maximum specular reciprocity
difference changes from .00459645576 in B0/B2 to 1.74e-18 in B1/B3. The total
maximum is .326430382 in B0/B1, .00459645576 in B2, and floating-point zero in
B3. This is a fixed-frame analytical kernel comparison, not a claim of exact
reciprocity for finite renderer patterns, degenerate frames or trained models.

### Unchanged decoder

At (-1,0), T and B norms remain .0010000000475, determinant about 1e-6;
16 sampled half-vector norms range .749075532 to .997460306. At (0,0), both
frame norms are .707106769 and half-vector norms range .883491278 to .998730958.
Near-negative-axis unit encodings have norms about 1.000000477; regular unit
orientations remain unit within float32 error. Near-zero nonunit encodings
are explicitly separate: a unit vector cannot approach (0,0). All diagnostic
values are finite, which does not make degenerate frames valid. No decoder
correction was made in any arm.

## Execution, checks and preservation

The exact one-run command was:

```bash
cd /home/isavand/paper/Nerofactor-paper
PYTHONDONTWRITEBYTECODE=1 timeout --signal=TERM --kill-after=10s 600s \
  python3 -B -m experiments.run_anisotropic_ablation_v1 \
  --output /home/isavand/paper/Nerofactor-paper/docs/evidence/ANISOTROPIC_ESTIMATOR_ABLATION_V1.json
```

It exited successfully in 9.350 seconds scoped wall time / 9.368 process CPU
seconds (driver timing excludes process startup). Runtime: Python3.9.12,
NumPy1.21.5, Torch2.4.0+cu121, CPU with one scoped Torch thread. The installed
Torch version label does not imply CUDA use. No GPU queries/kernels, network
construction, checkpoint loading, data access, Docker, training or installs
were performed. The historical Python3.8/Torch1.8 runtime was not executed.

| Actual work | Count |
| --- | --- |
| Parameter/view cases | 25 |
| Radiance calls | 4,800 |
| Specular half vectors | 1,638,400 |
| Diffuse directions | 819,200 |
| Main random requests / requested floats | 4,800 / 3,278,400 |
| Unique private uniforms | 32,784 |
| Sample scalar-light evaluations | 7,372,800 |
| Reference directional-kernel evaluations | 16,384,000 |
| Precomputed reference scalar-light evaluations | 1,966,080 |
| Decoder half vectors / random requests | 128 / 16 |
| Decoder reused uniform values | 256 |
| Nonfinite observations | 0 |

Global CPU RNG states and paired-stream hashes stayed unchanged; owned thread
count was restored. JSON status complete means the protocol finished, not that
all physical or accuracy checks passed.

Only the new focused test module ran:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests/scientific \
  -p test_anisotropic_estimator_ablation_v1.py -v
```

Final result: **16 passed, 0 failed, 0 skipped** (0.846 seconds test-body time).
A preceding test attempt had 15 passes/1 failure because its expected reduction
used a scalar-channel layout with different float32 rounding. The fixture was
corrected to the source's three-channel layout before the full experiment;
scientific operators were unchanged. Tests cover B0 fidelity, isolated/factorial
changes, exact denominator, unchanged regularization diagnostics, zero accepted
samples, height recovery, unchanged decoder, RNG, CPU restrictions, unresolved
references, paired statistics, output collision and failure cleanup. Synthetic
fixtures never execute the full experiment. No historical suites were rerun.
Four new Python files also pass Python3.8 AST parsing. Static evidence checks
verify JSON dimensions, budgets/statistics, current implementation/protocol
hashes and unchanged accepted-source/evidence scope; Git whitespace passes.
AST parsing is not execution under Python3.8. These checks do not admit GPU or
native rendering equivalence.

## Scientific conclusions and proposed next step

- **H1 supported:** total-count normalization is mathematically appropriate for
  unconditional reflected-direction sampling, with strong finite evidence on
  all converged component references.
- **H2 supported with a finite-pattern qualification:** the corrected map gives
  the intended cosine measure. The existing deterministic endpoint heights
  retain predictable quadrature error that azimuth randomization cannot remove.
- **H3 qualified:** combining the two changes reduces component and total error,
  but does not satisfy every frozen finite-sample comparison. Do not relabel the
  24 converged total failures or four unresolved totals as passes.
- **H4 supported:** energy violations and singular decoding remain independent
  defects. No physical-admissibility claim follows from the estimator fixes.

Recommend both corrections as separately selectable B1/B2 factors in a future
explicit experimental training variant, with B3 as the combined candidate and
unchanged B0 retained for attribution. This milestone does not wire them into
training or authorize a run. Before that implementation, review the finite
radial-pattern error, regularization diagnostic semantics and degenerate-frame
domain separately. A new analytical protocol could study approved midpoint or
randomized strata and improved grazing references; neither change was made here.
Keep decoder and energy redesign as distinct proposals/ablations. Any subsequent
GPU/runtime or reconstruction/relighting comparison requires separate approval.
