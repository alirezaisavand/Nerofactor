# Completed E.11D lighting pilot: saved-data results

[Scientific report](../../docs/BELL_HIGHLIGHT_PILOT_E11D_RESULTS_V1.md),
[evidence manifest](../../docs/evidence/BELL_HIGHLIGHT_PILOT_E11D_RESULTS_V1.json),
and [unchanged protocol](../../docs/protocols/BELL_HIGHLIGHT_PILOT_E11D_V1.md).
Neither L1 nor L2 passes; the plots are not v4 admission evidence.

Tracked CSVs/figures are byte-identical copies of new offline outputs. The
6.9MB original native capture batch and earlier failures remain outside Git.
RGB is the actual saved sRGB observation; specular maps use one shared linear
luminance scale. Response maps use one shared amplitude scale per factor across
all panels. Dots are the real246 probe pixels; unsampled grey surface is
unmeasured, with no interpolation. Cyan local passes do not override failed
reference convergence. The separate quantization plot shows unchanged/one-code/
two-code responses and composite gates. Full probe rows retain source pixel IDs.

To repeat **file-only verification/plotting** in a new explicit directory:

```bash
cd /home/isavand/paper/Nerofactor-3d2-e2a
MPLCONFIGDIR=/tmp/e11d-analysis-mpl python3 -B -m experiments.analyze_bell_highlight_e11d \
  --batch /home/isavand/paper/evidence/bell-highlight-e11d-08fb9def2696483ba97f8828fa86da6c \
  --benchmark /home/isavand/paper/evidence/bell-oracle-e11-v3-fea6430e2a644b679b0b35989569c240/benchmark \
  --output /home/isavand/paper/evidence/bell-e11d-offline-review-NEW-UNUSED-ID
```

This command has no Torch/native imports or rendering. It never relaunches the
pilot, changes its arrays, alters thresholds or creates a dataset. Existing
directories are rejected. The original analysis figures remain unchanged;
tracked figures come from its separate `paper-figures-v2/` layout revision.
Matplotlib/NumPy versions can affect display/replay bytes; scientific decisions,
artifact identities and counters must match exactly. Diagnostic float64 replay
bounds do not relax any scientific threshold. Any next rendering requires
separate explicit approval; no full128-view generation/training is authorized.
