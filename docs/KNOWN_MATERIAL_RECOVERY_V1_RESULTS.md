# 3C-1: first known-material recovery — completed, recovery unsuccessful

The single approved CPU experiment completed both mandatory scientific gates, all five initializations and exactly250 updates per start. **Zero of five starts meets the full predeclared recovery criteria.** This is an unsuccessful bounded recovery result, not a numerical execution failure. No settings, thresholds, samples, views, lights or budgets were changed after observing results; no experiment was repeated.

Base `179e36eb01e976d216a45fd323cc764308b49bd0`; feature branch `feat/known-material-recovery-v1`. Only A2/atan2 on the known flat, noise-free case was used. No neural network, GPU, native dataset, historical checkpoint, mesh, Docker or package installation was involved.

## Evidence and reproducibility

- [Complete one-run JSON](evidence/KNOWN_MATERIAL_RECOVERY_V1.json), 1896859 bytes, SHA-256 `f3c7a93e8f24ef7e3aac2de66b91da52962cb2f75f0d64cc66a199ba159da451`. It preserves all84 two-grid targets,28 perturbed reference evaluations, two-step/two-grid Jacobians, every initial/final vector, histories, final per-observation/component/RGB predictions and metrics, Adam state, tape hashes and actual budgets.
- [Approved protocol](protocols/KNOWN_MATERIAL_RECOVERY_V1.md), unchanged SHA-256 `7b13c54c62d1e04f7041af0784e0bf8f7118d8b616abfb1cc6e3258bdeb2539b`. Its proposal wording is preserved; the researcher's subsequent 3C-1 instruction authorized execution and criteria.
- [Pre-execution specification](KNOWN_MATERIAL_RECOVERY_V1_EXECUTION.md), frozen SHA-256 `5104c4a7145c892ac54487867ad0574876463194533c99b2cfa37a086f443295`. It records ordering, batching, conservative derivative-stability admission and the exact command. It was not edited after execution.
- [Independent reference/Jacobian implementation](../experiments/known_material_reference_v1.py), SHA-256 `bad7b5be7da2d6e569bc373a6f888c0f257846148e7fe0f1aece323f9ed247da`. NumPy Cartesian integrals do not call the source sampler or fitted shader.
- [Bounded runner](../experiments/run_known_material_recovery_v1.py), SHA-256 `b168dc1a0432a246dd6f2346d7b6150fa6d88b2f816ec062fb5c9674aaab2231`. The approved shader is unchanged, SHA-256 `de79ef3b0199b5f6e372feed6a9f16492ad966802b814e2cc9aab6700d35014f`. All numerical dependencies used by the runner are hashed in the JSON.

The run executed the hashed implementation as new working-tree files from the stated base; the later publication commit also contains its results. A future publication commit is not presented as the already-existing launch commit. Full precision is retained in JSON; tables below round only for readability.

## Scientific gates passed before optimization

All84 RGB observations and all components passed the fixed256x512/512x1024 reference gate, abs(delta)<=2e-6+5e-5*abs(fine). Maximum component discrepancy was 3.92574862e-13; maximum discrepancy divided by its allowed bound was 1.55408566e-08. Every plus/minus reference used by the Jacobians also passed. Totals required converged diffuse and specular components. No reference refinement or tolerance change occurred.

Fine-grid SVDs of the72x7 scale-normalized independent-integral Jacobian:

| FD step | Singular values (descending) | min/max ratio | Required ratio |
| --- | --- | --- | --- |
| 0.0001 | 2.57866978, 2.41561576, 2.41560948, 0.331140198, 0.281228502, 0.153031345, 0.0752670041 | 0.0291883066 | >=1e-4 |
| 5e-05 | 2.57866978, 2.41561576, 2.41560948, 0.3311402, 0.281228524, 0.15303135, 0.0752670055 | 0.0291883071 | >=1e-4 |

Both numerical ranks are seven, with no null direction at the declared ratio. Cross-step spectral discrepancy is 2.83653404e-08; coarse/fine derivative discrepancies are 9.04777572e-11, 1.42227899e-10. Each is below the smallest retained singular value 0.0752670041, passing the pre-execution stability margin. This establishes local rank at the true parameters for the independent continuous model, not global uniqueness, fixed-pattern shader identifiability everywhere, or optimizer convergence.

## Every initial and final material

Physical variable order is mx,my,theta(rad),kR,kG,kB,metallic. Truth is (.20,.55,.31,.30,.45,.60,.25). Initial values below are the private float64 draws; actual float32 logits and decoded starts are also retained in JSON. No start was excluded or restarted.

| Seed | Initial mx | my | theta | kR | kG | kB | metallic |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 101 | 0.459839177 | 0.492400552 | -0.606433506 | 0.217084536 | 0.302912994 | 0.611166189 | 0.76711749 |
| 102 | 0.508608951 | 0.555592143 | -1.04536965 | 0.379569045 | 0.638444161 | 0.547243635 | 0.747214136 |
| 103 | 0.409266723 | 0.254529158 | 1.01374407 | 0.302566211 | 0.696579349 | 0.552302756 | 0.467482997 |
| 104 | 0.238752375 | 0.286979014 | -0.2970705 | 0.684496017 | 0.379084745 | 0.311424045 | 0.717709846 |
| 105 | 0.200243954 | 0.350103775 | -0.638262995 | 0.715613447 | 0.263412101 | 0.521384026 | 0.150951656 |

| Seed | Final mx | my | theta | kR | kG | kB | metallic | Adam returns |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 101 | 0.461403072 | 0.400354356 | -1.38920236 | 0.34875378 | 0.51833576 | 0.687869787 | 0.528234482 | 250 |
| 102 | 0.56196022 | 0.463196576 | -1.33705938 | 0.357078135 | 0.530381441 | 0.703084707 | 0.507403314 | 250 |
| 103 | 0.644783616 | 0.448524654 | 1.86020899 | 0.35949719 | 0.53429091 | 0.708617091 | 0.493959725 | 250 |
| 104 | 0.350048572 | 0.539357722 | 0.274869502 | 0.334969431 | 0.478356659 | 0.628671825 | 0.352556497 | 250 |
| 105 | 0.539910853 | 0.69284296 | 0.304908305 | 0.389385343 | 0.529402316 | 0.709202409 | 0.457646877 | 250 |

Widths are physical slope parameters, not squared network outputs. Raw theta/axis values are gauge-dependent: sorted-axis, physical Q and major-axis modulo-pi metrics below account for swapping widths plus pi/2. Seed103 theta crossed the atan2 half-angle branch during its recorded path (initial1.013744, final1.860209); no wrapping or repair was added. This observation alone does not attribute the recovery failure to the branch cut.

## Recovery errors and unchanged success criteria

| Seed | Q relative | Major-axis deg | Max albedo abs | Metallic abs | Held-out relative L2 | All passed |
| --- | --- | --- | --- | --- | --- | --- |
| 101 | 0.496485594 | 7.35712356 | 0.0878697872 | 0.278234482 | 0.0662600899 | False |
| 102 | 0.576587835 | 4.369551 | 0.103084707 | 0.257403314 | 0.038004788 | False |
| 103 | 0.645956965 | 1.17956761 | 0.108617091 | 0.243959725 | 0.0270019971 | False |
| 104 | 0.275276255 | 2.01282929 | 0.0349694312 | 0.102556497 | 0.0251677636 | False |
| 105 | 1.00891683 | 0.291732611 | 0.109202409 | 0.207646877 | 0.0365425562 | False |

Required thresholds remain Q<=.05, identifiable orientation<=5 degrees, max albedo error<=.02, metallic error<=.03 and held-out relative L2<=.01. All predictions are above the .1 anisotropy-gap criterion, so no orientation result is hidden as unidentifiable. Four starts pass orientation alone; no start passes Q, albedo, metallic or held-out radiance criteria. There is no newly invented relighting pass threshold.

| Seed | Sorted width abs errors | Sorted width relative errors | Albedo RGB abs errors | Albedo RMSE |
| --- | --- | --- | --- | --- |
| 101 | 0.200354356, 0.0885969281 | 1.00177178, 0.161085324 | 0.0487537801, 0.0683357596, 0.0878697872 | 0.0701612586 |
| 102 | 0.263196576, 0.0119602203 | 1.31598288, 0.0217458552 | 0.057078135, 0.0803814411, 0.103084707 | 0.0823519812 |
| 103 | 0.248524654, 0.0947836161 | 1.24262327, 0.172333847 | 0.0594971895, 0.0842909098, 0.108617091 | 0.0864918985 |
| 104 | 0.150048572, 0.0106422782 | 0.750242859, 0.0193495967 | 0.0349694312, 0.0283566594, 0.0286718249 | 0.0308168503 |
| 105 | 0.339910853, 0.14284296 | 1.69955426, 0.259714473 | 0.0893853426, 0.0794023156, 0.109202409 | 0.0934873134 |

No predicted metallic is within the declared .01 boundary interval. Pairwise final Q Frobenius distances span 0.0875494035 to 0.298130599. Every pair, raw parameter median and range is retained; raw statistics are explicitly labeled gauge-dependent. These five trajectories do not establish stable material recovery.

## Held-out views and unseen lighting

These are local linear-RGB radiance observations, not rendered camera images or thesis NVS scores. Fit24, held-out24 and relight36 observations are disjoint. All final evaluations use the four independent declared evaluation tapes1000..1003; fit optimization used fixed tapes0..3. Four predictions are averaged before MSE. Held-out data were never used for selection or updates.

| Seed | Fit RMSE | Held-out RMSE | Relight RMSE | Fit rel L2 | Held-out rel L2 | Relight rel L2 |
| --- | --- | --- | --- | --- | --- | --- |
| 101 | 0.00992404216 | 0.0156018788 | 0.00550169328 | 0.0411679815 | 0.0662600899 | 0.0800186519 |
| 102 | 0.00580530873 | 0.00894876682 | 0.0035465433 | 0.0240822075 | 0.038004788 | 0.051582231 |
| 103 | 0.00397848322 | 0.00635800353 | 0.0029051924 | 0.016503973 | 0.0270019971 | 0.0422541876 |
| 104 | 0.00536165293 | 0.00592610721 | 0.00227750404 | 0.0222417867 | 0.0251677636 | 0.033124857 |
| 105 | 0.00859655921 | 0.00860446358 | 0.00399236662 | 0.0356611737 | 0.0365425562 | 0.0580664496 |

| Seed | Unseen light / seen view rel L2 | Unseen light / unseen view rel L2 | Held-out total max abs |
| --- | --- | --- | --- |
| 101 | 0.0705979911 | 0.089752369 | 0.0486092355 |
| 102 | 0.0454113139 | 0.0579465542 | 0.027541795 |
| 103 | 0.0402936202 | 0.0444286023 | 0.0222234091 |
| 104 | 0.0324328918 | 0.0339129482 | 0.0183955988 |
| 105 | 0.0602155358 | 0.0554586851 | 0.0246023999 |

Component relative errors on held-out observations reveal substantial material compensation:

| Seed | Diffuse relative L2 | Specular relative L2 | Total relative L2 |
| --- | --- | --- | --- |
| 101 | 0.276431134 | 0.933789782 | 0.0662600899 |
| 102 | 0.227337951 | 0.794004071 | 0.038004788 |
| 103 | 0.200249932 | 0.724443851 | 0.0270019971 |
| 104 | 0.0855276843 | 0.284156497 | 0.0251677636 |
| 105 | 0.138109814 | 0.562542982 | 0.0365425562 |

Diffuse is underpredicted and specular overpredicted in signed aggregate for these final fits; their errors partially cancel in total radiance. Thus comparatively small total residuals do not mean recovered components or parameters. The JSON contains all component RMSE/MAE/relative/max errors and predictions. This compensation is an observation, not proof of a globally unidentifiable model.

## Optimization and bounded execution

Exactly250 successful CPU Adam returns per start,1,250 total. Settings remained lr.01, betas(.9,.999), eps1e-8, weight_decay0; no scheduler, clipping or priors. Actual remaining Torch Adam flags and seven-scalar moment state are recorded per start. There was no optimizer restore or implicit continuation.

| Seed | Initial fit MSE (update0) | Last logged fit MSE (update249) | Last gradient norm | Fit seconds | Status |
| --- | --- | --- | --- | --- | --- |
| 101 | 0.00815954432 | 8.86582056e-05 | 0.000220531583 | 20.0316922 | complete |
| 102 | 0.00367439655 | 2.94322017e-05 | 8.97321006e-05 | 20.0728915 | complete |
| 103 | 0.00396219315 | 1.34708243e-05 | 3.21283369e-05 | 20.2281725 | complete |
| 104 | 0.0180193745 | 3.20089603e-05 | 0.000557559018 | 20.2363522 | complete |
| 105 | 0.0193509515 | 7.27734878e-05 | 0.000859725405 | 20.3140948 | complete |

Logs every10 returned updates explicitly distinguish pre-update loss/gradient from post-update parameter values. The final250 evaluation above uses the final parameters and separate tapes. No extra forward on fit tapes, best-iterate selection, more updates or reruns were added. Last logged gradients are nonzero and losses remain below earlier values; this does not prove stationarity or a global optimum. No numerical, nonfinite or gate failure occurred; unsuccessful recovery is distinct from execution status complete.

Runtime: Python3.9.12, NumPy1.21.5, Torch2.4.0+cu121 **CPU only**, one Torch/BLAS/OpenMP thread; GPUs hidden by the explicit empty CUDA_VISIBLE_DEVICES. No CUDA kernels or neural model were used. Scoped wall time 115.16995s and process CPU 115.15865s, below the1,800s external/internal cap. Interpreter startup is additionally covered by the external timeout. Historical Torch1.8/native GPU equivalence remains untested.

| Actual work | Count |
| --- | --- |
| optimizer_returns | 1250 |
| fit_shader_points | 120000 |
| evaluation_shader_points | 1680 |
| reference_direction_evaluations | 117964800 |
| reference_grid_directions | 655360 |
| reference_precomputed_scalar_lights | 4587520 |
| target_direction_evaluations | 7864320 |
| jacobian_direction_evaluations | 110100480 |
| fit_specular_directions | 122880000 |
| fit_diffuse_directions | 61440000 |
| fit_shader_calls | 1250 |
| evaluation_specular_directions | 1720320 |
| evaluation_diffuse_directions | 860160 |
| evaluation_shader_calls | 5 |

Independent per-view kernels and affine Fresnel factors were reused only for mathematically identical light/channel integrals. Consequently actual target7,864,320 and Jacobian110,100,480 node evaluations are below the predeclared upper bounds55,050,240 and440,401,920. Sample budgets match the approved limits exactly. Fixed tapes and global NumPy/Torch CPU RNG states remained unchanged; owned thread count was restored.

Exact single-run command (also frozen in the execution specification):

```bash
cd /home/isavand/paper/Nerofactor-paper
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 timeout --signal=TERM --kill-after=10s 1800s \
  python3 -B -m experiments.run_known_material_recovery_v1 \
  --output /home/isavand/paper/Nerofactor-paper/docs/evidence/KNOWN_MATERIAL_RECOVERY_V1.json
```

## Tests, preservation and scientific limits

Only the NEW test module ran:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s tests/scientific \
  -p test_known_material_recovery_v1.py -v
```

Final result: **16 passed,0 failed,0 skipped**,1.054s test-body time. Earlier development execution of the then15-test module also passed; the new seed-batching test was added before the full experiment. Fixtures cover exact scene/splits/starts, private tapes, world lighting, seed-major mean, independent reference algebra/convergence, Jacobian scale/rank/stability and failed perturbations, mandatory gate ordering, exact250 toy optimizer returns, nonfinite stop, unchanged criteria, timeout/budget failure and exclusive-output collision. The toy optimizer is not the real material experiment. No older suites or experiments were rerun.

Static checks cover Python3.8 syntax for three new Python files, current implementation/specification hashes, saved JSON metrics and budgets, Markdown links/whitespace and preservation of all29 previous evidence files plus approved shader/protocol/frozen sources. Python3.8 syntax is not execution under that interpreter. The old protocol, 3C-0 shader and G/A snapshots are unchanged.

The independent ground-truth renderer uses a different numerical approximation from fitting, avoiding the finite-shader-only inverse crime. It shares the explicitly intended material law; this is not evidence for real-material accuracy or energy conservation. Two-grid agreement and local full rank do not establish global uniqueness. No posterior, formal uncertainty interval or multi-scene/material generalization claim is made.

## Recommended next step — separate approval required

Do not begin neural/GPU reconstruction or report this as successful parameter recovery. Retain these five unsuccessful starts as the baseline. First review the saved trajectories and component compensation. Then propose a small, fixed-budget **forward/gradient diagnosis at the true and saved terminal parameters**, with independent references and prescribed common tapes: quantify finite-estimator error and check local derivative fidelity, rejection-mask and angle-branch effects. Such evaluations were not added to v1's exhausted budgets. Only after that review should a separately versioned experiment consider optimization duration, parameter scaling or another controlled factor. No longer run, alternative seed, tolerance change, added regularizer or model modification is authorized here.

Unresolved: how much failure comes from incomplete optimization versus fixed-pattern model/gradient error; conditioning away from truth; global ambiguities; pathwise angle/mask artifacts; and whether any revised protocol can recover all seven parameters reliably. The local gate rules out a rank-deficient independent Jacobian at this truth under the stated operational criterion; it does not resolve those broader questions. No scientific correction is inferred or implemented from these failures.
