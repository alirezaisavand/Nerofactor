# Known-L2 Bell tensor completion v3

The v2 K2 run passed the native-reference and paired-initialization gates.
The chart and ambient initial RGB components agreed exactly. Their maximum
world-Q difference was 1.49011e-8. K2 then stopped while scoring the
camera 0 material map at step 0. Its completion adapter did not expose
`TARGET_KEYS`, the ordered set of oracle fields the evaluator passes to the
existing physical-error calculation. The failed run used 106 shader calls
and 55,259,136 shadow rays, returned zero Adam updates, and saved no completed
evaluation. K3 never started. K1's verified 10,000-update run remains the
known-light counterpart to D2.

## Correction and checks

The v3 completion adapter inherits the full existing known-L2 interface.
It keeps its own two-arm selection, freeze, authorization and previous-run
evidence checks. The trainer's evaluation code, physical metrics, tensor
renderer, light, loss and thresholds use their existing implementations.

A CPU regression executes the existing step-0 evaluator for both arm labels
with synthetic shader outputs. It checks pointwise metrics, boundary and
surface diagnostics, lossless maps, the analytic illumination comparison,
and completion of the endpoint. A separate test checks the attributes used
by the shared trainer and offline analyzer against the v3 adapter. A
file-only check verified the failed run's archived camera 0 map and scored
all 4,395 admitted foreground observations through the inherited scorer.
All 14 error arrays were finite. The reconstructed world-Q and directly
exported tensor agreed to 1.14720e-8 maximum absolute error.

## Bounded execution

One fresh authorization must bind the v3 commit, freeze and batch path.
The batch runs K2 then K3, with 10,000 Adam returns per arm and evaluation
every 2,000 updates. The host verifies that the failed v2 K2 container is
absent by exact ID and ownership label before creating the new output.
The v3 freeze pins the v2 failure receipts, camera 0 capture and prior
K1 verification. Preserve the earlier outputs and approvals.

Use the same admitted dataset, D8-C and D8-T recipes, initial-state and
sampling gates, corrected image and A100 GPU 0. The limits are 20,000 new
Adam returns, 24,964 shader calls, 19,366,674,432 shadow rays, 72 component
gradient probes, 2,560,000 sampled surface pairs and 24 boundary predictor
calls covering 59,376 points. Use four CPU threads, 64 GiB RAM with no swap,
128 PIDs, a 25% CUDA allocator limit and 8 GiB output per arm. The foreground
watchdog is 7,500 seconds with 30 seconds of kill grace. Host and container
arm limits remain 3,600 and 3,540 seconds respectively. Run as non-root
with read-only source, inputs and container root, and no network. Stop on
the first mandatory failure. The authorization covers this one launch
and file-only comparison with K1 and the learned-light controls.
