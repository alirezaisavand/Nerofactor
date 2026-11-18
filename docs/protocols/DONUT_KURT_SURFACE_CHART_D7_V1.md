# Donut D7 matched surface chart control

D7 isolates the added neighbor penalty in the accepted D6 experiment.
Use D6's original A corrected-core trainable field, learned direct MLP,
zero indirect light, original material and diffuse-white priors, initial
weights/seed 6033, fitting cameras, sampler, ray counts, Adam and schedule.
D7 starts fresh and uses exactly 10,000 updates. The D2 and D6 results at
`b68d2299121b6202d898699fee70f17c9bd5a4ad` serve as archived controls.

## Domain and term

Use the same 26,624 eligible edges, 1,024 reserved diagnostic edges,
centroids, smooth loop normals and surface distances as D6. Draw 128
edges per update from the same independent seed 26061 and permutation
stream. Check every regenerated domain array against D6, including the
smooth normals. Check the count-vector identity at all six saved states
and exclude reserved edges from the training stream. Keep the D6
reserved-edge evaluation in transported world Q.

Keep ordered predictor channels in the original local shader chart.
Use conventional coordinate differences with squared-width coordinates
and a periodic angle distance. Like the existing sphere chart control,
compare the axial double-angle vectors in their respective charts. The
D7 control uses a fixed angle weight and a common squared distance scale:

```text
A_i = (mx_i², my_i²)
r_i = normalized (cos(2 theta_i), sin(2 theta_i))
L = 0.05 mean[(0.02 / distance)^2
              (||A0 − A1||² + 0.00638 ||r0 − r1||²)]
```

The axial chord respects 180° rotations. The fixed labels can penalize
a physically equivalent swap at one endpoint. Equal coordinates can hide
physical director changes as the two charts move, and orientation
differences contribute even close to isotropy. These are properties of the
coordinate control. The term uses the same two predictor calls and adds
no albedo/metallic term, ray query or RNG draw. Both arms use smooth
normals for the same renderer, domain checks and physical diagnostics.
D7 compares its chart parameters without applying a transport rotation.

## Fitting-only scale

Use the same 10,345 pairs with both faces supported by fitting observations
as the D6 scale check. Select the closest saved D2 fitting observation to
each incident face centroid, breaking ties in fixed camera/pixel order.
Use state 10,000 predictions only. For equal squared widths across a
pair in a shared fixed frame, the orientation part of the Q squared
distance equals `(mx² − my²)² / 2` times the squared axial chord.
Freeze 0.00638 from half the square of the median observed absolute
squared-width gap, rounded to five decimals. No GT width or other
material label enters this calculation.

At the common coefficient 0.05, the fitting proxy means for D6/D7
are 1.463e−5/2.591e−5. RMS per-pair gradients against the two endpoints'
`(mx,my,theta)` coordinates are 2.049e−3/1.025e−3. Matching the loss
alone implies an outer coefficient near 0.028. Matching aggregate
coordinate gradients alone implies 0.100. Their geometric midpoint is
about 0.053. Retain the existing 0.05 as a practical compromise. The
rotation gradient is larger in the D7 proxy and the width gradients
are smaller. The calculation gives approximate strength matching in
fitting coordinates. It cannot establish equal network gradients or
scales on unseen faces. Report the six live component-gradient probes
and the observed training loss scales. Keep all coefficients fixed.
[Fitting-only scale evidence](../evidence/donut_surface_chart_d7_preparation_v1/fitting-scale.json).

## Execution and comparison

Run one exact-source authorized D7 history in the reviewed image on A100
GPU 0, with read-only source/inputs/root, a non-root user, no network,
four CPU threads, 64 GiB host RAM with no swap, 128 PIDs, a 25% CUDA
allocator fraction and 8 GiB owned output. Limits are 3,720 s foreground
with 30 s kill grace, 3,600 s host-arm and 3,540 s inner with 20 s grace.
Use the D6 caps of 19,248 shader calls, 15,113,256,960 shadow rays, 30
component reverse-sweep slots, 1,280,000 sampled pairs and 20,000 extra
spatial predictor calls over 2,560,000 points. Save the six snapshots
at updates 0/2,000/4,000/6,000/8,000/10,000 and five checkpoints.
Stop on a mandatory gate or resource failure and retain partials.
An authorization is for one fresh output directory and one launch.

Use the fixed 10,000-update endpoint as primary. Compare D2 vs D7 to
measure spatial chart smoothing versus no added surface prior, and D7
vs D6 to compare penalties on the same neighborhoods. Report pooled and
held-out physical Q, eligible axial orientation tails, major/minor
width bias, gap and saturation, camera 34 fixed coverage strata,
held-out RGB/color/metallic/illumination and raw reserved-edge Q
consistency. Include actual coverage, timings and gradient interactions.
Do not claim a metric advantage based only on lower Q variation or a
favorable intermediate checkpoint. D6–D7 can distinguish these two
specific regularizers on this object, with only approximate strength
matching. It cannot isolate every possible chart metric or establish
boundary-aware generalization. No historical arm or D8 is launched.

The D6 execution freeze still points to its original source commit
`1a66ff579b4a330942f4a78ea25374b0c2e87332`. Its offline replay uses the
accepted results commit above, where the shared launcher has its frozen
D6 bytes. Use that commit for historical replay.
