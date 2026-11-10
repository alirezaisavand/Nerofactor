# 3B-2 frozen protocol: anisotropic estimator ablation v1

Base `759a1253bf1db207221d6989ec4ca6f9c3ae3225`, branch
`feat/anisotropic-estimator-ablation-v1`. This document is written **before**
the single full experiment. Its SHA-256 is stored at execution; do not edit it
afterward. Interpretation belongs in the separate results report.

## Scientific changes and preserved baseline

Frozen A is `e1e859779301faced46d6476691e20b546c27ae0`, field.py SHA-256
`b8386007dc009442cdefdeccc3c6ecc0114f6d0e7feaf804008b0bd9020091fe`.
Use the accepted3B-1 approach: compile only the explicitly listed pure method
bodies from the manifest-verified source into an object with no neural modules,
constructor or scientific imports. No prior test suite/diagnostic is executed.
Independent reference functions are reused without rerunning3B-1's experiment.

| Variant | Specular normalization | Diffuse directions |
| --- | --- | --- |
| B0 / v1 | Original max(Nvalid,1) | Original elevation-remapped pattern |
| B1 / v1 | Total M | Original pattern |
| B2 / v1 | Original max(Nvalid,1) | Correct cosine pattern |
| B3 / v1 | Total M | Correct cosine pattern |

B1 is a separate compiled experimental copy with exactly one AST replacement:
`specular = spec_weighted.sum(dim=1) / valid_counts` becomes division by M_spec.
All masks, contributions, original Beckmann sampler, Fresnel reconstruction,
returned PDF and all other statements remain identical. In particular, retained
legacy diagnostics f_s_sum/L_spec are not changed or used to train anything.
This experiment does not redesign a regularizer; future training must review how
its diagnostics/energy prior should relate to an approved estimator variant.

B2 overrides only sample_diffuse_directions with the approved radial construction.
Preserve original get_orthogonal_directions/cross basis, stored azimuth values,
operation order for angle multiplication/modulo, the one per-point random azimuth
shift and sample counts. Reconstruct exact z from the original sample_sphere(N,0)
indices N..2N−1: z=j/N. Verify sin(original double elevation)=z. The source's
stored remapped quantity e=1−2*arcsin(z)/pi is rounded to float32; merely using
sqrt(e) is incorrect and inverse-rounding e is unnecessary. Record its inverse
roundoff; use the exact construction instead. Correct coefficients are
sqrt(z) and sqrt(1−z), exactly as approved. No1e-7 is added to those corrected
square roots; the original1e-7 perturbations remain in B0/B1 unchanged. No
half-bin shifts, extra radial randomness or normalization repair is introduced.

The same scalar coefficients and RGB channel assumptions are used in all arms:
metallic=.001, albedo=.999 (achromatic), d=k*(1−m), F0=.04*(1−m)+m*k.
These valid head values were the accepted3B-1 energy counterexample; this fixed
choice is declared before measurement, not selected after comparing outcomes.
a=0 and ks=1 remain original. No Smith G, head/decoder change, energy redesign,
new regularization, production integration or original file edit is authorized.

## Fixed factorial and shared streams

- Roughness pairs (.04,.04),(.1,.6),(.3,.7),(.6,.9),(1,1); endpoints are the
  requested limiting width values, not a claim about finite sigmoid saturation.
- Outgoing polar0/30/60/80/89 degrees, world azimuth0; unit normal+Z;
  theta=.31 radians, original double-angle decoding. All regular-frame half
  vectors/basis lengths must be within2e-6 of unit length; no repair if not.
- Source CPU tensors float32, independent reference directions/algebra float64.
  Float32 representation differences are retained with a predeclared16-epsilon
  absolute roundoff allowance in numerical comparisons, not silently transformed.
- Diffuse512, specular1024; is_train=True, random_azimuth=True. Exactly two
  source specular rand calls then one source diffuse-azimuth rand call per arm.
- Private np.random.RandomState(seed), seeds0–15. For each seed generate U and V
  arrays(1,1024), then azimuth array(1,1,1), round to float32 and supply them to
  the original torch.rand call sites with a checked CPU-only tape. No global RNG
  seeds or states are consumed/changed. The same tapes are reused across all
  four variants and all25 roughness/view pairs (cross-case correlations noted).
  This is a new controlled fixture stream, not claimed to reproduce a historical
  PyTorch seed. Record tape hashes, exact request counts and no-extra-draw gates.
- Lighting, evaluated unchanged at the supplied directions in all arms:
  L0=1; L1=max(wi_z,0)^2; L2=.1+.9*max(dot(wi,(1,1,1)/sqrt(3)),0)^8.
  All RGB channels are equal. No learned light, mesh, camera, ray tracer or SDF.

25 parameter/view pairs ×16 seeds ×4 arms ×3 lights =4,800 original or
single-correction radiance calls. Main sampling budgets are1,638,400 specular
half vectors and819,200 diffuse directions,4,800 rand requests (3,278,400
requested floats). Unique private uniforms are32,784; repeated requests are
paired copies. Reference grids cost16,384,000 directional kernel evaluations
across25 cases; the three reference lights are precomputed once per grid and
reused. Record actual counters and elapsed wall/process CPU times. Scalar
lighting evaluations are distinguished from sampled direction counts.

## Independent references and uncertainty

Intended diffuse is the Lambertian integral of d/pi. Intended specular is the
independent Cartesian Beckmann/Kurt a=0 kernel, integrated over the incoming
hemisphere. Use only the fixed Gauss-Legendre-in-cosine / midpoint-azimuth grids
256×512 and512×1024. They do not use the Monte Carlo estimator under test.
Compare the two values **per lighting/component**, absolute tolerance2e-5.
A total is converged only when its components and total satisfy that bound.
If not, keep coarse/fine values/difference and mark comparisons unresolved.
Do not increase resolution, tune thresholds or automatically retry any case.
Two-grid agreement is an operational convergence check, not a rigorous bound.

For16 seed observations, report the mean, unbiased sample variance(ddof1),
standard error=sqrt(var/16), and nominal Student-t(df15)95% interval using
2.131449545559323*SE. Report absolute/relative signed bias and absolute relative
bias against a **converged** fine reference; relative errors are null for
|reference|≤1e-12. Unconverged cases have null bias/accuracy claims. Store raw
seed-level estimates and accepted/rejected sample counts for reproducibility.

The diagnostic agreement budget is3*SE+2e-5+16*float32_epsilon. The2e-5 term is
the operational reference uncertainty allowance, not a proof of exact integration.
Counts across many correlated cases are descriptive, not a familywise significance
test or proof of unbiasedness. Paired changes compare B1−B0,B2−B0,B3−B0,B3−B1,
B3−B2; record uncertainty of signed-error/estimate differences and absolute-error
differences under the common reference, never independent-arm error bars.

A predeclared finite-pattern caveat: even a correct cosine generating measure
uses fixed heights z=j/512, so mean(z)=(511/1024). Under L=ci² the corrected
finite diffuse value is d*511/1024, while the integral is d/2: deterministic
bias −d/1024≈−0.00097461. Random azimuth cannot reveal this radial quadrature
error; SE across seeds is zero there. **Do not fix this by changing heights or
claim H2/H3 establish exact finite-sample agreement.** The original incorrect
measure instead has dense-pattern second moment1−2/pi. Constant light alone
cannot discriminate the diffuse mapping.

## Energy, reciprocity, decoder, interpretation

Directional energy is total reflected radiance under L0=1, with its reference
and seed uncertainty, per view and variant. Record violations above1 and whether
CI95 lies entirely above1; never claim a soft prior or these corrections enforce
energy conservation. No model fitting is performed.

Reciprocity uses a fixed second direction normalize(.2,.5,1), the current outgoing
direction and the fixed physical frame. Report intended kernels in both orders
and ideal effective kernels: B0/B2 specular=f_s/P(wo), B1/B3=f_s; B0/B1 diffuse=
d*qd(wi)/ci, B2/B3=d/pi. P is separately integrated in the Gaussian slope-domain
acceptance circle with16,384 azimuth samples. This characterizes the ideal
large-M/valid-frame limit, not finite-pattern pointwise BRDF recovery or a new
native rendering result. Store independent P and measured acceptance/counts.

Targeted decoder cases are(-1,0),(0,0), normalized(-1,±1e-7),(1,0), regular
(cos(.62),sin(.62)), and nonunit near-zero(1e-8,0),(0,1e-8). Unit vectors
cannot approach the zero vector; that ambiguity in the request is explicitly
resolved by labeling nonunit cases. Reuse the first16 values of the existing
seed0 U/V tapes, original widths(.3,.7), outgoing+Z. Record frame norms/dots/
handedness, all half-vector norm extrema/mean and finiteness. No decoder correction.
8×16=128 extra half vectors,16 tape requests, no additional unique random values.

H1: fixed-M has the intended unconditioned expectation under valid sampling.
H2: corrected diffuse mapping has the intended cosine measure, with its fixed
finite-pattern caveat above. H3: combining these restores the two intended
component estimators in their valid domain, not guaranteed finite-grid accuracy.
H4: neither change repairs energy allocation or singular angle decoding.
Passing unit tests means these operators/measurements were implemented correctly;
physical admissibility, native equivalence and reconstruction improvement do not
follow. Preserve every unresolved case and original accepted evidence.

## One authorized execution

First run only the NEW focused test module (no3A/3B-1 suite). Then execute the
full grid once. The exact command used will be recorded in the separate report:

```bash
cd /home/isavand/paper/Nerofactor-paper
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests/scientific -p test_anisotropic_estimator_ablation_v1.py -v
PYTHONDONTWRITEBYTECODE=1 timeout --signal=TERM --kill-after=10s 600s \
  python3 -B -m experiments.run_anisotropic_ablation_v1 \
  --output /home/isavand/paper/Nerofactor-paper/docs/evidence/ANISOTROPIC_ESTIMATOR_ABLATION_V1.json
```

The output must be new. It records running/partial state and each completed case;
errors remain failures/incomplete evidence. No full-grid override flags, implicit
retry or overwrite. CPU thread count is scoped/restored; global NumPy/Torch CPU
RNG states and paired input hashes must remain unchanged. No CUDA/GPU queries,
NN construction/inference/training, checkpoint loading, data/mesh access, Docker,
installs or historical diagnostics. Stop after analysis/report and commit/push of
this feature branch; do not merge or start another scientific experiment.
