# Donut D10 camera 28 specular attribution

D10 tests the extra bright ring seen at the accepted D8-T update 10,000
endpoint. D6 remains the reconstruction baseline and D8-T the primary
ambient-tensor candidate. This intervention keeps D8-T's recovered base color
and metallicity fixed, along with the admitted camera 28 surface positions,
smooth normals, outgoing directions, mesh, corrected BVH, zero indirect
illumination and the D8 tensor renderer.

| Role | Physical world Q | Directional light | Base color and metallicity |
| --- | --- | --- | --- |
| R1 | D8-T | D8-T directional MLP | D8-T |
| R2 | Oracle | D8-T directional MLP | D8-T |
| R3 | D8-T | Analytic L2 | D8-T |
| R4 | Oracle | Analytic L2 | D8-T |
| R5 | Oracle | Analytic L2 | Oracle |

R5 measures native-versus-independent integration with oracle scalar
materials. Its contrast with R4 measures the **joint** scalar intervention.
A nonzero R5 residual can include fixed diffuse-quadrature discrepancy,
specular Monte Carlo error, uncertainty of the independent reference and
systematic estimator differences. The analysis reports these discrepancies
by component. It counts channels exceeding
`0.001 + abs(fine_reference - coarse_reference) + 6 * empirical_SE + 64 * eps32`
as a descriptive compatibility diagnostic. The absolute guard and SE
multiplier follow the D8 compatibility settings. The full image diagnostic
does not inherit its simultaneous statistical interpretation or create a new
acceptance threshold.

## Work and replay

Restore only the network weights of the explicitly frozen, trusted D8-T
checkpoint. PyTorch 1.8 unpickles on CPU inside the isolated container.
Verify the checkpoint header, representation, keys, tensor dtypes and
shapes. Leave the optimizer and the saved RNG payload unused. Test the
recovered and oracle Q and scalar fields against the saved data before
native shading. Maximum absolute discrepancy is 1e-6 in each field.

For each condition use contiguous 512-observation chunks of the original
4,849 ordered camera 28 foreground samples. Run 64 matched draw indices with
the D8 seed `800000 + 28000 + chunk_start + draw` and common world Gaussian
tape seed `1600000 + seed`. Execute 512 fixed evaluation diffuse directions
and 1,024 specular directions per shaded point as in D8. Require R1's first
eight draw means and empirical channel SEs to agree with the archived
values within 1e-6 absolute before running the other conditions. The replay
draws count toward the 64. Preserve per-draw unclipped linear diffuse,
specular and combined RGB samples. Rendering uses the directional MLP from
the checkpoint whenever the light is recovered.

The output ceiling is 4,480 native shader calls and 3,336,732,672 proposed
shadow-ray slots including conditional swaps. The execution makes zero
primary ray queries, optimizer updates and new independent reference
integrations. Use the reviewed corrected image on one A100 GPU 0 with a
25% CUDA allocator cap, four CPU threads and cores, 64 GiB host RAM with
no swap, 128 PIDs and a cooperative 512 MiB output cap. Use a foreground
1,320-second watchdog with 30-second kill grace and an inner 1,200-second
limit with 20-second grace. Native timings are unmeasured. The container
runs non-root, networkless and with read-only source, root filesystem,
checkpoint and other inputs. It writes only to its new output batch and
temporary filesystem. An input, replay, timing, numerical or cleanup
failure ends the authorized run and leaves its partial evidence intact.

## Frozen analyses

Use the exact 485-pixel D9 D8-T positive-excess ring and the entire 4,849-pixel
foreground. Report signed luminance residual, RGB squared error, component
errors and the diffuse/specular squared-error cross term. Use the mean
render across draws and whole-draw paired delete-one jackknife intervals
with a t critical value at 63 degrees of freedom. The intervals summarize
native specular sampling uncertainty conditional on the fixed scene and
evaluation settings. Fixed-diffuse and reference integration errors are
outside those intervals. Report the Q and light effects at both levels of
the other factor and the interaction R4−R3−R2+R1. The ring was derived from
D8-T errors and its reuse localizes this observed artifact.

If the Q swap reduces RGB SSE by at least 25% on the ring and the lower
paired 95% bound for that reduction exceeds zero, activate the two
separate width and orientation swaps under the same light. Check recovered
light first, then oracle light. For width-only, keep the recovered
principal director while replacing both sorted widths. For orientation-only,
keep both recovered widths while replacing the principal director. Both
interventions keep recovered color and metallicity fixed. Apply both swaps
only where recovered `(major²−minor²)/(major²+minor²) >= 0.1`. Keep recovered
Q on the remaining pixels and report the eligible ring and foreground
counts and effects explicitly. Rendering differences within these fixed
conditions support controlled attribution under the recovered renderer and
dataset. They do not establish what caused training to converge to these
fields.

Use the saved fitting-camera samples at native radii 0.005, 0.01 and 0.02
with a normal separation of at most 30 degrees, together with same-triangle
camera counts. Compare ring and non-ring support. Assess existing reference
highlights within radius 0.01 using the admitted per-view highlight mask.
This is a proximity proxy for photometric supervision of the surface.
It does not test additional visibility, training-time sample diversity or
whether supervision identifies the physical material.

## Launch and evaluation

The freeze pins the exact D8-T receipt and endpoint data, Donut reference
and admission, D9 ring, sources and fresh output path. Before approval
the Python entry point runs a file-only preflight without importing Torch.
After approval, the foreground `experiments/run_donut_specular_d10.sh`
launcher binds the authorization to the clean Git commit, source freeze,
image and output path. It refuses an existing batch.

The native run writes the standalone R1 replay, conditional render arrays,
panel and hash-bound `render-receipt.json` under the fresh batch's `run/`
directory. The launcher saves the exit code and console log. The file-only
`experiments/analyze_donut_specular_d10.py` validates the work counts,
artifact hashes, R1 replay and conditional decision before creating the
comparison figure and analysis JSON in a new directory. The large draws,
checkpoints and reference remain outside Git.
