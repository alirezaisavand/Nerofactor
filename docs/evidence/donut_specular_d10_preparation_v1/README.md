# Donut D10 preparation

The frozen camera 28 panel contains 4,849 ordered foreground observations.
The selected 485 pixels match the accepted D9 localization exactly. The
complete D8-T receipt, checkpoint bytes, endpoint material/radiance
arrays, independent reference and research admission were checked without
checkpoint deserialization. The D8 tensor shader, corrected core and
pinned scientific sources retain their accepted bytes.

| Fitting-view observation proxy at radius 0.01 | Ring (485) | Elsewhere (4,364) |
| --- | ---: | ---: |
| At least one nearby normal-compatible observation | 484 (99.8%) | 4,352 (99.7%) |
| At least three fitting cameras with nearby observations | 275 (56.7%) | 1,211 (27.7%) |
| At least one fitting camera with a nearby reference highlight | 93 (19.2%) | 2,238 (51.3%) |
| Median peak nearby specular luminance among supported points | 0.01835 | 0.03493 |

The ring has substantial fitting-view spatial support by this proxy and
much weaker fitting-view specular observations. These saved data do not
separate material from illumination error. The intervention in the frozen
protocol addresses that question. The archived eight-draw endpoint has a
combined RGB standard-error RMS of 0.001318 across the foreground. Sixty-four
common draws reduce the expected specular Monte Carlo SE by roughly
`sqrt(8 / 64)` if draws have comparable variance. The analysis measures
the realized paired uncertainty and treats fixed diffuse sampling and
reference discrepancy separately.

[Coverage counts by camera and radius](camera28-fitting-coverage.json).
[Frozen protocol](../../protocols/DONUT_SPECULAR_ATTRIBUTION_D10_V1.md).

Four focused CPU tests, shell syntax, Python 3.8 grammar, a synthetic plot
and the real metadata-only preflight passed. Live Docker image inspection
was denied by the local socket permissions. The researcher-side launcher
checks the immutable image ID before starting. No native render,
checkpoint deserialization or optimizer update occurred in preparation.
