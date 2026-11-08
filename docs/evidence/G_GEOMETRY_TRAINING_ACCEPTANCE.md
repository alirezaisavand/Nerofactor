# G two-update regression acceptance v1

1D.6A accepts **only the fresh two-update prefix** of the selected G geometry
implementation. The researcher executed the native runs; Codex inspected the
existing reports, source identities and declared output bytes without rerunning
the experiment. The [versioned manifest](g-geometry-training-acceptance-v1.json)
records exact paths, hashes, runtime, comparison and exclusions.

Review baseline: `596c070664ffd9cfff50805939026cff9ce21883`.
G scientific revision: `79252178e65914ad1acbc14790b1514f276bfea0`.
Evidence root: `/home/isavand/paper/evidence/bell-G-two-update.rHme8GuO`.

| Report relative to evidence root | SHA-256 |
| --- | --- |
| `r/stage-run.json` | `f869b197b1439d5804a8d606f175e7b5a7f0c44c826ace2ed6b9a3cf613b5b92` |
| `r-repeat/stage-run.json` | `52c9a429b9340d08240690dffaec936e4cecc26df289b1b519f35fe5607c90c1` |
| `c/stage-run.json` | `e4fbbcac1c87ccba4a6e0c5a0554c0a2ec59143642ee2b2f682e35e6b68df30f` |
| `final-comparison/G-training-comparison.json` | `b732549a157a972d68d1fbbc6623501aed8e512fc7bdc85db5b59720f7a94444` |

Verified report result: `G_regression_passed`, after independent R-repeat
repeatability. All 16 declared output files were hash-checked. Reported source
hashes match Git objects at the reviewed infrastructure and G scientific commits.
The comparison was recomputed before this milestone's implementation edits.
Its live source-identity gate deliberately will not accept old reports against
the newly changed infrastructure; archive acceptance binds the reviewed old
source rather than weakening that gate.

Reports declare Python 3.8.10, Torch 1.8.1+cu111, CUDA 11.1, ABI 0, A100 80GB
PCIe, driver 550.163.01 and image
`sha256:2572950d9f662b7603b604878613e85c1a0fec90842447e9a256751a130ee137`.
These are file-backed runtime observations, not a new Docker inspection by Codex.
The exact immutable image declaration is retained; no image was built or run.

Both original G optimizer returns completed. LR at step 0 was zero: Adam state
advanced while parameters did not change. Step 1 used the original warm-up LR
and changed parameters. G split is ordered training `1` through `127`, validation
`0`, with no NVS partition. G uses its original Adam defaults, cosine schedule
and no clipping. No checkpoint, natural validation or selection boundary was
reached. Hash/scalar agreement over this prefix is not full training
reproducibility, historical producing-run attribution, validation, serialization,
mesh extraction or G-to-A handoff acceptance.

[Supersession v2](dual-baseline-supersession-v2.json) adds this scope alone;
v1, historical A camera-0/image-58 observations and their original reports remain
unchanged. Native compatibility, optimizer diagnostic, G validation and manual
roundtrip are separate future evidence gates described in the
[lifecycle audit](../G_GEOMETRY_LIFECYCLE.md).
