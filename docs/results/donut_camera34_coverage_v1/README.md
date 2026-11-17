# Donut camera 34 fitting coverage

Read-only comparison of the admitted reference and accepted D1/D2 endpoints.
The nearest compatible fitting sample is measured in native half-scale world
coordinates. A fitting camera counts once if a valid sample falls within the
radius and its smooth normal is within 30° of the camera 34 normal.

| Match radius | No nearby fitting camera | D1 median relative Q, no match / match | D2 median relative Q, no match / match |
| --- | ---: | ---: | ---: |
| 0.005 | 1,387 / 2,495 | 0.1623 / 0.0273 | 0.2106 / 0.0604 |
| 0.010 | 729 / 2,495 | 0.4935 / 0.0293 | 0.5387 / 0.0646 |
| 0.020 | 521 / 2,495 | 0.5495 / 0.0317 | 0.6145 / 0.0690 |

At 0.01, median orientation errors without versus with coverage are
21.82° / 0.87° for D1 and 23.93° / 1.67° for D2. The higher Q errors
without nearby coverage also occur within every normal-to-view-angle band.
Using same-triangle support gives a similar contrast. This pattern is
consistent with limited spatial support contributing to camera 34's tail.
Nearby observations do not establish exact visibility or angular sensitivity.
Supported regions also contain errors, especially for D2.

The JSON records counts, complete strata and input hashes. The NPZ gives pixel
coordinates, camera counts, triangle support and facing cosines so the same
strata can be joined with D3's final saved arrays. The
[analysis script](../../../experiments/analyze_donut_camera34_coverage.py)
reads the original evidence files and can write a separate result with D3
after its receipt and arrays are available.
