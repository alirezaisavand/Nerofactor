# Donut D6 surface-wide world-Q regularization

D6 is one additive ablation of D2. It uses the same corrected-core shader,
free learned direct lighting, admitted observations, matched fresh material
and light state, seeds, optimizer, schedule, sampling, 10,000 updates,
diffuse-white prior and all other legacy material priors. The initial
state hash must match D2 before any update. D4 is a separate calibrated-light
reference. D1–D5 share the accepted evidence at
`6eac9499e55fa92dd9373df43df267236fc4abb9`.

## Surface domain and sampling

Reuse the shared `surface_pairs` topology and `world_q` and
`transported_q_loss` definitions. A pair consists of two centroids of
faces sharing exactly one manifold edge, subject to the original 15°
face dihedral limit and 0.02 maximum surface path length. The distance
follows the two segments through the shared edge midpoint. Interpolate
and normalize the oracle's smooth loop normals at each centroid.
Validate their hemisphere and minimal transport. Inputs to domain
construction are vertices, triangles, triangle loop indices and smooth
normals. Neither a camera visibility gate nor any material boundary
weights are applied for this object.

There are 18,432 mesh faces and 27,648 eligible pairs. Reserve the exact
1,024 existing D2 diagnostic edges. The remaining 26,624 edges touch
all faces. With the fitting-coverage proxy (a face receives one or
more valid fitting camera samples), 9,052 faces lack fitting support
and account for 44.73% of mesh area. The training domain contains
10,345 pairs with two supported faces, 6,387 with one, and 9,892 with
neither. These counts describe coverage and have no effect on sampling
or weights. Coverage labels use only fitting ray hit identities.

Draw 128 pairs each update through uniform whole-pool permutations
without replacement within a sweep. A dedicated NumPy RandomState uses
seed 26061. Any unfinished permutation continues next update. One
complete sweep takes 208 updates and the final 10,000-update budget
draws 1,280,000 pairs. Material field queries are deterministic and
do not draw photometric RNG. Track actual pair counts, visited faces
and area by fitting support, and store the sampler state in each checkpoint.
The 1,024 reserved pairs are never drawn by this stream. They are
independent of the training neighborhood choice at the edge level.
Their faces may be represented by other training edges.

## Fixed loss and coefficient

At a pair of points compute the predicted physical tensors
`Q_i = mx_i² t_i t_iᵀ + my_i² b_i b_iᵀ` in world coordinates and
transport the second tensor to the first normal with the existing
minimal normal rotation R. The one added term is

```text
0.05 * mean[(0.02 / surface_distance)^2 * ||Q0 - R Q1 R^T||_F^2]
```

This is the Q branch of `surface_world_q_v1` with uniform edge weights.
The prior's separate color/metallic spatial term is absent from D6's
new loss. The original D2 objective still applies unchanged and no
ground-truth Q, orientation, color, widths or held-out RGB enter the
new loss. Ground-truth Q remains available for evaluation.

Keep the previous coefficient 0.05 and reference length 0.02. For a
pre-execution scale check, select each fitting-supported face's nearest
observed point to its centroid using the saved D2 fitting-only predictions.
Transport those predicted world-Q tensors using their saved smooth normals
and apply the centroid path length. Over 10,345 training pairs with two
supported faces, the mean weighted proxy was 1.463e−5, with median
6.585e−7. The last 1,000 D2 fitting updates had median native
material-prior loss 1.937e−5 and RGB loss 0.032144. RGB loss includes the
Charbonnier floor of sqrt(0.001). The Q proxy is about 2.8% of that RGB
median above the floor. This is a scale check on saved fitting
predictions. It does not estimate unseen-region loss or select a coefficient
using held-out results.

The Donut reference has spatially constant widths, color and metal with
a spatially varying director. The additive prior assumes the physical Q
field varies smoothly after local transport. It can penalize genuine
material transitions on other objects and does not enforce a globally
parallel field on a curved surface. Similar tensors can also be achieved
by shrinking anisotropy. Evaluate the anisotropy gap and absolute recovery
alongside transported variation.

## Bounded execution and evaluation

Run D6 once on the existing reviewed image, A100 GPU 0. Use the previous
arm limits of 3,600 s host and 3,540 s inner, with a 3,720 s foreground
watchdog, 30 s outer/host and 20 s inner kill grace, four CPU threads,
64 GiB host memory, 128 PIDs, 25% CUDA allocator fraction and 8 GiB
owned output. Inputs/source/root remain read-only, execution is
non-root and network is disabled. A new authorization must bind
the exact source, freeze, fresh output and dataset inputs. Stop on a
failure and preserve partial evidence.

Caps are 10,000 Adam returns, 19,248 shader calls and 15,113,256,960 shadow
rays. The spatial loss uses 20,000 predictor calls over 2,560,000 points,
excluding the original diagnostic batches, and adds no ray queries.
Reserve 30 component reverse-sweep slots over six probes (photo,
material, energy, specular and added surface-Q). Record training Q loss,
gradient-group norms, photo/Q cosine and CUDA forward time for spatial
work. Compare total elapsed time with D2.
Run-to-run differences do not isolate backward cost.

Save all six evaluation states and five checkpoints at the existing
2,000-update cadence. The **primary endpoint is 10,000**. Compare with
D2 on overall and held-out Q and photometric recovery, orientation
eligibility, both width errors and saturation, albedo, metallic,
illumination, and camera 34 frozen coverage strata. Include D4 as a distinct
calibrated reference. Report the same raw 1,024-pair transported consistency
score as D2 and archive D6 per-edge errors, both widths, anisotropy gap
derived from the widths, and counts for these reserved neighborhoods.
D2 archived the aggregate pair score, so there is no paired
per-edge D2 distribution on this panel. Existing per-pixel physical
maps give the matched D2/D4 width and gap distributions. Lower Q variation
alone is insufficient if the minor/major widths converge or physical errors
remain high. The experiment measures one seed and one material. It neither
reruns controls nor authorizes a chart-independent tensor predictor.
