# Authoritative scientific snapshots v1

Geometry: G `79252178e65914ad1acbc14790b1514f276bfea0`.
Material: A `e1e859779301faced46d6476691e20b546c27ae0`.

`manifest-v1.json` lists every copied file, original path, size and SHA-256.
Files in `geometry_g/` and `material_a/` are exact Git blob copies, not edited
implementations or merged branches. Both snapshots include complete local
network/dataset/train/utils/raytracing module trees, the two reachable COLMAP
readers, original configurations and split pickle, BSDF asset, README and
training/extraction/evaluation/relighting entry sources. No dataset, checkpoint,
third-party extension checkout or generated artifact is copied here. The original
branch histories and root legacy science remain untouched.

Keep these trees immutable. A changed scientific baseline needs a new snapshot/
manifest/protocol and explicit review. Do not run their legacy scripts directly:
use `run_stage_experiment.py` to select one cold scientific namespace with pinned
source identity and output isolation. Snapshot config paths are historical input
records; new envelopes contain no host paths and only runtime dataset/mesh/output
locations are rebound. Existing loss/shader configuration scopes are unchanged.

No top-level LICENSE/COPYING file exists at the pinned source refs. Original
README and embedded third-party/source license headers are retained byte-for-byte;
this integration does not invent a redistribution license. Resolve external
redistribution licensing before publishing copied sources outside this repository.
