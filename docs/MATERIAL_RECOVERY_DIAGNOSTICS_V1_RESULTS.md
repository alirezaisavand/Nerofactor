# 3C-2: diagnosis of failed anisotropic material recovery

The single approved CPU diagnostic completed with **zero optimizer calls**. The strongest supported explanation for the3C-1 result is unfinished optimization with material compensation. Finite-pattern forward/gradient error and half-angle branch artifacts are also measured; neither is silently corrected. The measurements do not establish a unique causal explanation or prove that extending optimization will recover every parameter.

Base `72629688bee84c0b06f4fa3c19b207917d2dd450`; branch `feat/material-recovery-diagnostics-v1`. A separate worktree at `/home/isavand/paper/Nerofactor-3c2-diagnostics` preserves the researcher's existing edit in the original workspace. No accepted shader, source, protocol or old evidence was modified. No new initialization, parameter optimization, neural model, GPU execution, dataset/mesh/checkpoint access, Docker or installation occurred.

## Evidence and scope

- [Diagnostic JSON](evidence/MATERIAL_RECOVERY_DIAGNOSTICS_V1.json), 10701457 bytes, SHA-256 `6106278d6f3bb41eed0933b668627626c462faaf20433c13c34d1c5d7685539f`. It retains per-tape/per-observation component RGB errors, objective/gradient vectors, finite-difference probes and mask changes,145 new terminal reference evaluations, spectra/Jacobians, trajectory replay and branch probes.
- [Offline trajectory JSON](evidence/MATERIAL_RECOVERY_TRAJECTORIES_V1.json), SHA-256 `4f69828f70d0f8d99063d2ffd10169aa991c312b8e930dbb558d7366d2c2693e`. This was computed before any shader evaluation. It retains all recorded trajectories, late improvements, descriptive correlations, compensation and branch brackets.
- [Frozen pre-execution protocol](protocols/MATERIAL_RECOVERY_DIAGNOSTICS_V1.md), SHA-256 `7b4ba5797d826d3845bf29e40d341876265edb96083974bd09c9a7d87bb4b6b6`. No edits occurred after execution. It specifies exact states/tapes/steps/tolerances/budgets and the literal command.
- Input remains the admitted3C-1 JSON, SHA-256 `f3c7a93e8f24ef7e3aac2de66b91da52962cb2f75f0d64cc66a199ba159da451`. All its numerical source, original protocol/execution and results identities were checked. The new implementation hashes are stored in the diagnostic; the old evidence is not regenerated or overwritten.

The reference is the intended continuous Beckmann/Kurt a=0 law, not a new energy-conserving BRDF. Shader behavior is the accepted A2/atan2, float32,512 diffuse/1024 specular samples. Physical coordinates are (mx,my,theta,kR,kG,kB,m); comparisons respect axis-swap/theta-periodicity using Q. Unconstrained coordinates are exactly the old seven scalar optimizer variables. New files were executed as hashed working-tree code from the stated base; their later publication commit is not misrepresented as the launch commit.

## Offline findings: unfinished trajectories and compensation

Offline analysis used saved values only, without running the shader. The losses below are evaluated at recorded pre-update indices239 and249; physical/Q differences are between saved post-update states240 and250. This timing offset is explicit, not silently equated.

| Seed | Last10 loss reduction | Last20 loss reduction | Physical delta norm240->250 | Q delta norm240->250 | Recorded branch crossings |
| --- | --- | --- | --- | --- | --- |
| 101 | 7.92528077% | 15.2452635% | 0.0149264153 | 0.00686171234 | 2 |
| 102 | 8.19926491% | 16.1840985% | 0.00877776056 | 0.00793710201 | 2 |
| 103 | 3.25999605% | 6.87491952% | 0.00359775993 | 0.00377991019 | 1 |
| 104 | 24.3813292% | 43.5751424% | 0.00667227011 | 0.00557488038 | 0 |
| 105 | 18.9083979% | 34.9663203% | 0.00730181507 | 0.00333581545 | 0 |

Loss, gradient and parameter histories provide no evidence of an attained stationary point. Small gradients or decreasing losses alone do not prove convergence. Recorded width/albedo/metallic/orientation-strength correlations vary substantially across starts; all values are retained as dependent-trajectory descriptions, without p-values or causal attribution. For example, mean albedo versus metallic correlation ranges from -.934 to+.965, so a single universal anticorrelation is not asserted.

| Seed | Fit residual diffuse/specular cosine | Cancellation fraction | Recorded branch brackets |
| --- | --- | --- | --- |
| 101 | -0.982229785 | 0.905596869 | 110->120; 120->130 |
| 102 | -0.99100409 | 0.932793149 | 50->60; 140->150 |
| 103 | -0.99468705 | 0.947887832 | 60->70 |
| 104 | -0.945925269 | 0.835429276 | none observed |
| 105 | -0.953114665 | 0.841976043 | none observed |

Cancellation fraction is1-||total residual||/(||diffuse residual||+||specular residual||), computed from saved evaluation-tape predictions. Its .835-.948 range confirms that total-radiance agreement hides substantial component errors. Q errors remain real after gauge handling: these are not merely swapped-axis angle equivalents. Compensation is confirmed; an exact global ambiguity is not. Unobserved intermediate steps cannot be reconstructed from every-ten-update logs.

Offline preparation first hit its provisional40-state guard because there were more recorded branch crossings than expected. Before the pre-execution protocol was frozen, the guard was set to45 and44 selected states were saved. This was a metadata-only preparation correction; no shader, independent reference or optimization diagnostic was rerun.

## A. Forward error at the true parameters

With the original fit tapes0..3 at the explicitly requested true physical vector, the actual float32 fitting objective is **3.07078125e-07**. Its direct physical gradient norm is 8.51808505e-05, and its chain-converted coordinate norm is 1.98768544e-05. Thus the finite objective has a nonzero residual and a nonzero measured derivative at truth, while the reused fine continuous objective/gradient there is exactly zero by construction. Cosine/relative discrepancy against that zero gradient is null, not an invented perfect or failed alignment.

| Fit-tape truth component | RMSE | Relative L2 | Max absolute |
| --- | --- | --- | --- |
| diffuse | 2.59656285e-06 | 1.37270062e-05 | 7.94093657e-06 |
| specular | 0.000553731324 | 0.0104869996 | 0.001786216 |
| total | 0.000554148303 | 0.00229877773 | 0.00178630988 |

| Truth lighting group | Diffuse RMSE | Specular RMSE | Total RMSE | Total relative L2 |
| --- | --- | --- | --- | --- |
| constant | 9.27229514e-08 | 0.000727226245 | 0.000727238651 | 0.00163299652 |
| directional | 2.99777457e-06 | 0.000482222461 | 0.000482854468 | 0.00452790132 |

Individual fit-tape total MSEs against the saved float64 reference are 3.6899952e-07, 1.82467974e-06, 1.48126802e-06, 6.76000797e-07. The combined objective averages the four predictions before squaring, so it is not their average MSE. Per-observation mean residuals, RMS across tapes, mean-prediction RMS and four-tape standard errors are retained for all components/RGB channels. This is observed fixed-pattern error, not proof of statistical bias over random seeds. The residual is dominated by the specular component; diffuse error is much smaller.

The separately labeled inverse-sigmoid float32 roundtrip has objective 3.07078494e-07 and genuine coordinate-autograd norm 1.98770954e-05. Its largest physical roundtrip offset is 3.57627868e-08. Direct truth and this slightly shifted coordinate representation are not silently interchanged. Residual analysis uses saved float64 targets; the original fit MSE uses the original float32 target/mean arithmetic.

| Truth under separate evaluation tapes1000..1003 | Total RMSE | Relative L2 |
| --- | --- | --- |
| fit | 0.000594776525 | 0.0024673161 |
| held_out | 0.000436804136 | 0.00185507667 |
| relight | 0.00020225678 | 0.00294169705 |

The true-parameter held-out relative error is about .186%, below the original1% radiance criterion. Old terminals were2.52%-6.63%. Terminal evaluation-tape predictions are reused unchanged from3C-1; they were not redundantly regenerated. The corrected finite model can predict these observations much better at truth, even though its fixed objective is not exactly zero there.

## B. Pathwise versus continuous gradients

| State | Finite fit objective | Continuous objective | Terminal finite loss / truth finite loss | Worst autograd-vs-FD relative error | Worst FD cosine |
| --- | --- | --- | --- | --- | --- |
| truth | 3.07078125e-07 | 0 | truth baseline | 0.00300900115 | 0.999996691 |
| seed-101 | 8.79327636e-05 | 9.86598863e-05 | 286.353069 | 0.00205895522 | 0.999999743 |
| seed-102 | 2.91898905e-05 | 3.46359318e-05 | 95.0568866 | 0.000673607737 | 0.999999773 |
| seed-103 | 1.34295124e-05 | 1.66338387e-05 | 43.7332108 | 0.0020807029 | 0.99999802 |
| seed-104 | 3.11546755e-05 | 2.96675154e-05 | 101.45521 | 0.00106958754 | 0.999999543 |
| seed-105 | 7.13120244e-05 | 7.09477542e-05 | 232.227628 | 0.000965571188 | 0.999999864 |

The ratio column for terminals is their finite objective divided by the finite objective at truth (44-286 times); it is not a ratio of finite to continuous loss. Both numerical FD steps retain the prescribed physical scales/denominators. Actual represented float32 endpoints and mask changes are recorded. Autograd matches the finite objective FD with at most .301% relative discrepancy and cosine>=.99999669 at these regular states. This supports correct local pathwise differentiation here; it is not proof of unbiased continuous gradients or behavior exactly at a discontinuity.

| Terminal | ||g_z finite|| | ||g_z continuous|| | Cosine | Relative gradient discrepancy | Continuous derivative along -g_z finite |
| --- | --- | --- | --- | --- | --- |
| seed-101 | 0.000219178902 | 0.000236258047 | 0.999092056 | 0.0831293724 | -5.17357633e-08 |
| seed-102 | 8.89322312e-05 | 0.000101612935 | 0.99163225 | 0.17384072 | -8.96104849e-09 |
| seed-103 | 3.18764638e-05 | 4.66209597e-05 | 0.981413623 | 0.354173428 | -1.45848991e-09 |
| seed-104 | 0.000546983421 | 0.000548303232 | 0.998538376 | 0.0540555767 | -2.99474418e-07 |
| seed-105 | 0.000847415495 | 0.000868692898 | 0.997951687 | 0.0677954273 | -7.34635969e-07 |

This compact table uses the fine grid and5e-5 FD step; both grids and both steps are preserved. Terminal finite-versus-continuous coordinate gradients differ by5.4%-35.4% in relative norm, despite high directional cosines(.9814-.9991). Hence correct pathwise gradients and exact continuous gradients are different claims. Every listed continuous directional derivative is negative: the current finite negative-gradient direction is also a local continuous-model descent direction. No such step was executed. Coordinate conversion is meaningful because all states are interior with nonvanishing dp/dz; direct physical gradients and converted vectors are both retained.

All145 new baseline/perturbed terminal reference evaluations passed the original convergence bound. Maximum component discrepancy is 3.50386387e-13. True references/Jacobians were reused, not regenerated. Coarse/fine and cross-step differences remain explicit in the conditioning evidence.

## C. Local sensitivity and optimization status

| State | Scale-normalized singular ratio | Rank at1e-4 | Derived optimizer-coordinate condition number | Step/grid rank stable |
| --- | --- | --- | --- | --- |
| truth | 0.0291883071 | 7 | 42.9710071 | True |
| seed-101 | 0.0321790824 | 7 | 30.8144303 | True |
| seed-102 | 0.0201102679 | 7 | 48.8224902 | True |
| seed-103 | 0.0203352236 | 7 | 48.7225425 | True |
| seed-104 | 0.025781944 | 7 | 39.931286 | True |
| seed-105 | 0.0125159447 | 7 | 79.5123749 | True |

Optimizer-coordinate condition numbers are derived read-only from saved J_normalized/scale times dp/dz; no new rendering is used. They range30.8-79.5 at terminals, versus43.0 at truth. All tested independent Jacobians remain full rank, with no near-null direction at the original1e-4 ratio. This does not diagnose catastrophic local rank deficiency. Nonlinear landscape, coordinate scaling, optimizer state and conditioning elsewhere on the path remain possible influences.

Together, nonzero independent gradients, negative local descent derivatives, continuing late improvements and a known much-lower-loss true point strongly support incomplete optimization under250 updates. They do not prove that the next Adam step decreases the loss, that a longer run will meet every recovery criterion, or that a global optimum has been found. The present diagnosis performs no optimizer update or learning-rate change.

## D. Compensation versus global ambiguity

Incorrect roughness, albedo and metallic combinations partially cancel diffuse/specular residuals, confirmed from the saved and new component predictions. Original parameter/Q errors remain unchanged; the five fitted states are physically different under Q, not merely orientation gauges. Local full rank at truth and terminals is compatible with weakly informative directions or distant alternative minima, but no exact global ambiguity is established. All measured continuous terminal objectives remain above zero and have nonzero gradients. Calling these states converged ambiguous solutions would overstate the evidence.

## E. Masks and half-angle branch artifacts

| State | Total XOR across28 local probes | Max XOR for one probe /98,304 samples |
| --- | --- | --- |
| truth | 15 | 4 |
| seed-101 | 11 | 4 |
| seed-102 | 27 | 7 |
| seed-103 | 39 | 9 |
| seed-104 | 19 | 3 |
| seed-105 | 29 | 7 |

Local probes change at most nine validity bits among98,304 specular draws. Pathwise FD agreement remains close. Therefore changing masks are observed, but these local measurements do not support mask transitions as the dominant gradient defect at the tested endpoints. Binary masks are not resampled or smoothed, and correct fixed-mask/pathwise behavior is not automatically an unbiased integral gradient.

| Background, +pi/2 cut | h (angle offset=h*pi) | Across-cut radiance RMS jump | Q jump | Pointwise physical kernel max jump | Mask XOR /98,304 |
| --- | --- | --- | --- | --- | --- |
| truth_material | 0.0001 | 0.000930329098 | 0.000233251339 | 1.67062169e-05 | 13294 |
| truth_material | 5e-05 | 0.000930321821 | 0.000116625675 | 8.35310805e-06 | 13292 |
| seed103_material | 0.0001 | 0.002101988 | 0.00019066324 | 6.01560563e-06 | 26702 |
| seed103_material | 5e-05 | 0.0021024952 | 9.53316247e-05 | 3.00780305e-06 | 26704 |

The -pi/2 probes give the same paired jumps. Halving the offset approximately halves the physical Q/kernel change, while the finite radiance jump stays near .0009303 for truth material and .002102 for seed103 material. Pointwise physical pi-periodicity errors are <=1.04083409e-17. This confirms a finite common-sample branch artifact while respecting physical distribution symmetry. The three pointwise kernel checks are not hemisphere integrals or integral-error bounds. Actual float32 encodings at the centers are recorded; a computed center is not pretended to be literal(-1,0).

Saved/replayed crossing brackets:101 at110->120 and120->130;102 at50->60 and140->150;103 at60->70. Their corresponding mask XOR counts are9,856/10,178;15,555/17,666;17,830. Parameters also change within these intervals, so those trajectory differences alone cannot isolate a causal branch contribution. Controlled local probes isolate the finite branch behavior, but no counterfactual optimization was run. No crossing is observed in the recorded104/105 snapshots; unlogged intermediate steps remain unknown. Branch artifacts may affect some paths, but are not established as the sole explanation for all five failures.

## Ranked explanation and evidence strength

| Rank | Mechanism | Evidence strength | Limit |
| --- | --- | --- | --- |
| 1 | Incomplete optimization with parameter compensation | Strongly supported: continuing improvements, nonzero independent gradients, aligned descent and lower-loss truth | Longer-run recovery not tested |
| 2 | Diffuse/specular material compensation | Confirmed: residual cosine -.946 to-.995 and cancellation .835-.948 | Does not prove exact global ambiguity |
| 3 | Finite-pattern forward/gradient mismatch | Confirmed for fixed tapes: nonzero truth loss/gradient and5.4%-35.4% terminal gradient discrepancy | Population bias and parameter-error floor not quantified |
| 4 | Half-angle branch artifact | Confirmed in controlled probes; recorded crossings on three paths | Contribution to each failed trajectory remains unresolved |
| 5 | Incorrect local autograd implementation | Not supported at measured regular states: FD agreement within .301% | Branch points and all intermediate states not covered |
| 6 | Severe local rank deficiency / proven global ambiguity | Not supported: full-rank local Jacobians and nonstationary endpoints | Global uniqueness and other path locations remain unresolved |

No single mechanism is claimed to explain the failure. In particular, a nonzero truth fitting objective matters, but it is far below every terminal objective and cannot by itself establish that the terminal errors are an unavoidable finite-sample floor.

## Execution, tests and preservation

One diagnostic execution, status complete, wall 81.4901008s / process CPU 81.4868573s, below the1,800s cap. Python3.9.12, NumPy1.21.5, Torch2.4.0+cu121, CPU with one Torch/BLAS/OpenMP thread and GPUs hidden. The CUDA tag in the installed version is not GPU execution. Historical Torch1.8 equivalence was not tested. No new dependencies were installed.

| Actual diagnostic work | Count |
| --- | --- |
| optimizer_calls | 0 |
| shader_calls | 235 |
| fit_points | 22464 |
| evaluation_points | 336 |
| specular_directions | 23347200 |
| diffuse_directions | 11673600 |
| new_reference_directions | 570163200 |
| new_grid_directions | 655360 |
| precomputed_scalar_lights | 4587520 |

The frozen protocol contains the exact command. Actual work matches the declared235 shader calls /22,800 points /35,020,800 sample directions and570,163,200 new terminal-only reference directions. Forty-four selected snapshots include five reused terminals; twenty branch probes use no new hemisphere integration. The original fit/evaluation tape hashes match, tapes/global CPU RNG remain unchanged and owned thread count is restored. Zero optimizer calls is also enforced by absence of update paths and new fixture guards; no optimizer state is restored or changed.

Only NEW `test_material_recovery_diagnostics_v1.py` ran: **17 passed,0 failed,0 skipped**, .096s test-body time. Its synthetic toy shader verifies diagnostic plumbing, derivatives, identity/budget/convergence failure handling, mask accounting, branch selection, no optimizer/network/checkpoint calls and exclusive output creation. It does not execute a real A2 diagnostic or old experiment. Three new Python files pass Python3.8 AST parsing; this is not execution under Python3.8. Additional read-only static checks verify artifact/source/protocol hashes, saved metrics/Jacobian arithmetic, counters, links/whitespace and all30 prior evidence files unchanged. Previous test suites and3C-1 optimization/reference generation were not repeated.

## Next single controlled experiment — recommendation only

Test **insufficient optimization duration under the same finite objective**. Propose one explicitly approved continuation of each of the same five3C-1 starts from saved update250 to a fixed total1000 (exactly750 additional Adam returns per start, at most3,750 total). Restore only the seven saved scalar coordinates and their saved JSON Adam step/moments; verify their identities before continuation. Keep A2/atan2, data,24 fitting observations, target arrays, tape hashes, sample counts, loss, lr.01, betas/eps/weight decay and evaluation tapes unchanged. Do not rerun the250-update prefix or introduce a new start, regularizer, decoder fix, sampling change, schedule or held-out selection. Use fresh outputs and a separately frozen CPU/runtime cap and final1000 evaluation.

The **only manipulated factor is the update budget**. Compare all five250-versus1000 parameter/Q/component/held-out metrics against the same unchanged recovery criteria. Improvement would support the duration hypothesis; persistent nonstationarity, biased plateaus or path instability would constrain it and motivate a later separate sampling/branch experiment. This is not a promise of success. No continuation code, optimizer construction or extra update is implemented/executed in3C-2; separate approval is required.

Remaining uncertainties: global uniqueness; how much finite-tape error shifts the optimum; causal impact of earlier branch crossings; behavior between recorded snapshots; robustness across independent tape sets and materials; and whether a longer unchanged optimizer reaches reliable recovery. No new physical model, energy prior or source correction is inferred from this diagnostic.
