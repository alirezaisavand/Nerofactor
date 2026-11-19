# Known-L2 Bell tensor completion v2

The original batch completed K1 and stopped during K2's initial paired gate.
K1 returned 10,000 Adam steps, saved every planned evaluation, and certified
cleanup. Its 897 native output files and all six evaluation states passed
file-only verification. Preserve K1 as the completed known-light counterpart
to D2. [Original experiment](BELL_KNOWN_L2_RECOVERY_V1.md)

K2 passed initialization parity and reached the paired gate after 32 native
compatibility draws. It completed zero optimizer updates and saved no
evaluation endpoint. K3 never started. The batch preserves the failure
receipt, authorization, freeze and logs.

## Correction and remaining work

The trainer selected K2's native D8-C representation, then compared
candidate sibling names against the launch label K2. Both candidates
differ from that label, so it selected the first D8-C entry again. The
initial gate wrote both temporary captures under the same key prefix
and then lacked the required D8-T world-Q array.

Resolve the sibling from the native representation and require exactly
two distinct valid representations. K2 uses D8-C with a D8-T sibling.
K3 uses D8-T with a D8-C sibling. Leave the frozen gate, rendering,
shaders, initialization, lighting, tensors, optimizer, losses, data,
sampling, regularizers and numerical thresholds unchanged. The completion
freeze inherits both exact tensor recipes and includes the prior K1
analysis and K2 failure identities.

Launch only K2 and K3 in that order in a fresh batch. Verify the failed
K2 container is absent by its exact ID and ownership label, including
stopped containers, before starting training. Record that verification
in the new batch. The old approval, output directory and files remain
historical evidence. K1 does not run again.

## Verified K1 comparison

At 10,000 updates, K1 improved held-out median relative Q error from
.24669 to .19211 versus D2 and minor-width MAE from .03108 to .01458.
Orientation p95 on the same 6,767 eligible observations fell from
28.20° to 18.39°. Held-out RGB relative L2 fell from .13346 to .12203.
Median projected Q jump recovery remained weak, with 2.25% for D2 and
2.42% for K1. Compare the completed K1 and new K2/K3 endpoints together
after verifying the new receipts, keeping the two execution identities.

## Bounded execution

Keep 10,000 updates per remaining arm with evaluations every 2,000 updates.
The limits are 20,000 new Adam returns, 24,964 shader calls, 19,366,674,432
shadow rays, 72 component-gradient probes, 2,560,000 sampled surface
pairs and 24 boundary predictor calls covering 59,376 points. Use the
same corrected image on A100 GPU 0 with four CPU threads, 64 GiB RAM without
swap, 128 PIDs, a 25% CUDA allocator limit and 8 GiB output per arm.
Use a 7,500-second foreground watchdog and 30 seconds of kill grace.
The inherited 3,600-second host arm and 3,540-second container limits
remain in force. Use read-only inputs, source and root, non-root execution
and no network. Stop on the first mandatory failure without retry.
