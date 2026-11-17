# Donut camera0 launch correction v2

The authorized v1 launch from cd1e5e9d42f9dd1f60051df964f9b60cadd92219
stopped with Blender exit2 after about2 seconds. It did not open the source scene
or call the render operator. No EXR, PNG or comparison was generated.

The host accepted the relative freeze path from its repository working directory,
then started Blender with the fresh run directory as cwd. The unchanged relative
argument consequently named an absent file. The previous metadata-only preflight
used an absolute path and did not exercise that working-directory boundary.

The two-line correction resolves the freeze and authorization paths to absolute
existing paths before validation or subprocess construction. It changes no
scientific settings, render code, image encodings, metrics or resource bounds.
A new CPU fixture exercises launch with relative CLI paths and a different
child cwd. Its fake process performs the actual child-relative file lookup;
it never starts Blender. All5 focused tests pass.

Failed v1 evidence remains unchanged at
`/home/isavand/paper/evidence/donut-camera0-reproduction-a0dc1eb96ffa49d2b727e83e1da8de42`.
Its original freeze, authorization, console and execution receipt are hashed in
`docs/evidence/donut_camera0_correction_v2/failure.json`. The old v1 freeze and
preparation evidence remain historical records. Supervisor cleanup completed.

The v2 freeze proposes a fresh output directory and corrected source. No retry
or second launch has occurred. The previous one-time approval is not reused;
another native launch requires explicit approval of the new commit/freeze/output.
The v1 scientific protocol remains unchanged. Until then, no reproduced image
or historical image-formation conclusion is available.
