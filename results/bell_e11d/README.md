# Bell E.11D saved cues and proposed lighting pilot

[Findings](../../docs/BELL_HIGHLIGHT_E11D_V1.md),
[protocol](../../docs/protocols/BELL_HIGHLIGHT_PILOT_E11D_V1.md) and
[manifest](../../docs/evidence/BELL_HIGHLIGHT_E11D_V1.json).
The CSV/figures use only saved v3 observations; no new rendering or neural calls.
Original arrays, complete region tables and test JUnit remain at manifest paths.

To reproduce the **offline analysis**, select a fresh output directory:

```bash
cd /home/isavand/paper/Nerofactor-3d2-e2a
MPLCONFIGDIR=/tmp/e11d-mpl python3 -B -m experiments.bell_highlight_e11d \
  --root /home/isavand/paper/evidence/bell-oracle-e11-v3-fea6430e2a644b679b0b35989569c240/benchmark \
  --output /home/isavand/paper/evidence/bell-e11d-offline-NEW-UNUSED-ID
```

This requires no Docker, native shader, model, reference integration or GPU.
The output must not already exist. Figure scales are explicit in the report;
never interpret the display's specular color scale as clipping reference radiance.

The following **scientific pilot** command is prepared for the researcher's tmux
terminal **only after** exact commit/freeze/output approval and creation of its
bound authorization file. It is not an executable authorization. The launcher
rejects an absent/mismatched approval or existing output directory, keeps a
foreground supervisor/watchdog and removes only its own label-bound container.

```bash
tmux new-session -s bell-e11d
cd /home/isavand/paper/Nerofactor-3d2-e2a
E11D_SOURCE=/home/isavand/paper/Nerofactor-3d2-e2a \
E11D_BENCHMARK=/home/isavand/paper/evidence/bell-oracle-e11-v3-fea6430e2a644b679b0b35989569c240/benchmark \
E11D_BATCH=/home/isavand/paper/evidence/bell-highlight-e11d-08fb9def2696483ba97f8828fa86da6c \
E11D_AUTH=/home/isavand/paper/evidence/bell-highlight-e11d-08fb9def2696483ba97f8828fa86da6c-authorization.json \
E11D_GPU=0 bash experiments/run_bell_highlight_e11d.sh
```

The Docker definition/image is reused unchanged; only independent reference/BVH
work is proposed. Outputs are pilot captures, not a regenerated training dataset.
No material model, optimizer, full128-view generation or2000-update comparison
is authorized. If the pilot completes, analyze its saved captures and freeze any
recommended v4 separately; do not tune failed candidates or automatically retry.
