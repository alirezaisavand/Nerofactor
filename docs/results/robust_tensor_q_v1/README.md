# Robust transported Q v1 launch failure

Donut-R1 completed **2,568** Adam updates and stopped during the surface-neighborhood
predictor call for update **2,569**. Bell-R1 did not start. The launcher exited
with code 1 after 327.17 seconds of reported Donut-R1 runtime. The initial-state,
surface-domain and known-light compatibility checks passed. Saved evaluations
cover updates 0 and 2,000. The step-2,000 checkpoint and maps are available.
The failed receipt has `scientific_work_complete=false` and
`cleanup_verified=false`. [Saved evidence identities and offline analysis](failure-analysis.json)

The frozen corrected-core chart predictor checks the norm of the raw
double-angle vector `2 * rotation_raw - 1` before normalization. At least one
row in a sampled surface endpoint batch fell below the existing `1e-6` norm
threshold. The check raised before that batch produced its transported residual
or pseudo-Huber loss, and before the pending optimizer update. The archived
error contains a joint SHA-256 of the points and their raw orientations. The
failing row, measured norm, and anisotropy gap are unknown. The frozen sampler
replay identifies the 128 candidate pairs for the failed update. Their minimum
distance was 0.006693 native units and maximum geometry multiplier was 8.9304.

## Partial comparison at the same saved update

Observation-weighted held-out metrics at update 2,000 are shown below. Q error
is relative Frobenius error. The final two columns show median signed
projection of the predicted Q jump onto the reference jump, with one denoting
full recovery. The scores were recomputed from the saved arrays.

| Arm | RGB relative L2 ↓ | Median Q error ↓ | Q p95 ↓ | Boundary median Q error ↓ | Q jump camera 28 ↑ | Q jump camera 34 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D2 | 0.2129 | 0.4310 | 0.8552 | 0.4728 | 0.0109 | 0.0111 |
| D6 | 0.2307 | 0.4779 | 0.8858 | 0.5102 | 0.0077 | 0.0122 |
| Donut-R1 | 0.2161 | 0.3968 | 0.8791 | 0.4524 | 0.0105 | 0.0119 |

At step 2,000, Donut-R1 improved median Q and boundary-band Q error versus D6,
while the projected jumps remained near 1% of the reference. Its minor-width
MAE was 0.0334 on held-out views versus 0.0291 for D6. The same-batch spatial
gradient norm was 0.001259 for the robust loss versus 0.002456 for the
quadratic counterfactual, about 51%. The median robust-to-quadratic penalty
ratio over 2,568 complete batches was 0.9003. These early results show
attenuation and changes in physical error, with little recovery of the actual
discontinuity. The planned 10,000-update endpoint is missing.

## Decision

The paired result is **blocked**. Preserve the gate and the failed artifacts.
Restoring the checkpoint and continuing would require separate authorization
and would still use the same chart representation. The smallest independent
scientific follow-up is a separately approved Bell-R1-only run against B2 if
the robustness hypothesis remains the priority. A future Donut test with the
ambient tensor predictor would need its own matched quadratic tensor control.
No new rendering, checkpoint deserialization, training, or changes to the
frozen source were needed for this analysis.
