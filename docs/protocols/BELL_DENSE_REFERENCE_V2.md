# Heterogeneous Bell dense reference continuation

Dense reference v1 stopped at its 32-patch per-view limit. Camera 127's
assembled maximum RGB difference passed at .001480, while its pooled
relative L2 was .002085 against the .002 limit. The failure has been
replayed from the baseline and all 32 archived patches. All reflectance
and ordinary-angle controls passed. The remaining largest RGB residuals
are specular-dominated and nearly head-on. The archived prefix remains
incomplete scientific evidence. [Failure audit](../results/bell_dense_reference_failure_v1/README.md)

The continuation keeps the 16 fitting and two held-out camera IDs and
all geometry, material, L2 lighting, convergence tolerances and
quadrature definitions from v1. Verify and copy its six admitted sparse
reference views byte-for-byte. Copy camera 127's baseline and its 32
patches from the failed batch under their original hashes. Replay the
exact saved patch order before any new quadrature. Continue camera 127
from the next ranked pixel, then process the eleven unstarted views in
the original order.

The per-view patch cap is 64 for all appended views. The original
128-patch panel cap includes the 32 copied camera 127 patches, allowing
at most 96 new patches across the panel. Keep 256²/512² RGB baselines,
1024²/2048² refinements, the unchanged absolute and pooled selector,
one refinement per pixel, the independent slope and ordinary-angle
checks, and the .003 absolute and .002 pooled-relative bounds. Stop on
failed refined absolute convergence, a failed final gate, or an
exhausted budget. Convergence is unverified beyond the saved prefix.

| Work | Maximum newly executed |
| --- | ---: |
| Primary rays and CPU primary intersection witnesses | 0 |
| Shadow directions and BRDF point-directions, each | 15,099,822,080 |
| Slope nodes attempted and returned, each | 3,649,126,400 |
| Native material model calls and optimizer updates | 0 |

The reused and new work together stay within the original panel-wide
ceiling. Record the prior and new work separately. Use the same
corrected image and runtime on A100 GPU 0 with four CPU threads, 64 GiB
of host RAM without swap, 128 PIDs, a 25% CUDA allocator cap and 512 MiB
of output. The foreground/inner time limits remain 7,200/7,080 seconds
with 30/20 seconds of kill grace. Require a fresh batch, non-root UID,
read-only original inputs and source, and no network. Check that the
failed container is absent before launching. One exact authorization
covers only this continuation and its offline replay.

The offline verifier checks both old and new identities, selected patch
order, artifact bytes, final arrays, previews, masks, coverage and
counters. Preserve the failed directory. The new reference still
requires researcher admission before a separate training approval.
