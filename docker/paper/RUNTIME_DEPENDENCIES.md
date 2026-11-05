# Recovered dependency pins: Step 1D.3B-2B

The tree started clean on `paper/infrastructure` at
`9bcb8603687a56798b5c35b8f20fadac88c6afe4`.
[runtime.reference.json](runtime.reference.json) records all **102 exact installed
Python versions**, their recorded dependency edges/Python constraints, 21 Debian
role versions and recovered tools. It is reference evidence, **not an install
lock, reviewed `resolved.json`, or a build-ready artifact bundle**. The accepted
Dockerfile, preflight boundary, scientific code and numerical settings are unchanged.

## Inventory validation and scope

Input: researcher-supplied `/home/isavand/paper/runtime_inventory.json`, 37,564
bytes, SHA-256
`a5515797526ebe0f41b3b309b4e2181b60e2f8b29953dc17e9ae03d7bfdba61e`.
The original file was read without modification. It is attributed to the reference
container `intelligent_keldysh`; it does not itself contain the capture-time image
ID/container ID, source revisions, ABI query or wheel provenance. Prior supplied
image/source/ABI evidence remains separately recorded in [reference.json](reference.json).

Validation checked duplicate keys, schema/field types, normalized unique package
names, PEP 440 versions, prior direct/extension pins, dependency targets and
specifier satisfaction, requested extras and graph reachability. Findings:

- 102 distributions, all reachable from the collector's 18 runtime and 6 build
  roots; 142 recorded active dependency edges, no missing/conflicting edges.
- All 82 declared `Requires-Python` constraints accept **3.8.10**. Twenty packages
  omit that declaration; omission does not prove compatibility.
- All requested extras are empty. The collector already selected marker-active
  edges; original marker expressions and inactive metadata are not retained.
  The graph can be checked independently, but marker selection must ultimately
  be checked against the actual acquired wheel/source metadata.
- The collector reported zero issues. This is a consistent selected closure,
  not proof that these are the only distributions installed in the container,
  that imports/native linking succeed, or that numerical results reproduce.

## Recovered pins and packaging distinctions

| Component | Installed/reference observation |
| --- | --- |
| Interpreter / platform | `/usr/bin/python3`, CPython 3.8.10, x86_64, `cpython-38-x86_64-linux-gnu` |
| Torch / CUDA / ABI | 1.8.1+cu111; NVCC 11.1.105; ABI 0 from prior evidence, not queried by this inventory |
| GCC / G++ executables | **9.4.0 / 9.4.0** |
| CMake / Eigen / glibc | **3.16.3 / 3.3.7 / 2.31** |
| PyYAML / h5py / plyfile / tqdm | **6.0.2 / 3.11.0 / 1.0.3 / 4.66.5** |
| Matplotlib / Pillow | **3.7.5 / 10.4.0** |
| pip / setuptools / wheel | **24.3.1 / 75.2.0 / 0.34.2** |
| packaging / pybind11 | **24.1 / 2.13.6** |
| Ninja distribution / executable banner | **1.11.1.1** / `1.11.1.git.kitware.jobserver-1` |
| New confirmed raytracing metadata requirements | **trimesh 4.12.2**, **dearpygui 2.3.1** |

Preserve the distinction between Debian wrapper/package versions and executable
versions. `python3`/`python3-dev` are `3.8.2-0ubuntu2`, while `python3.8` and
`python3.8-dev` are `3.8.10-0ubuntu1~20.04.18`: this does not imply downgrading the
interpreter to 3.8.2. GCC/G++ wrappers are `4:9.3.0-1ubuntu2`, their `-9` packages
are `9.4.0-1ubuntu1~20.04.2`, and executables report 9.4.0. Runtime libstdc++/
libgcc/libgomp packages are `10.5.0-1ubuntu1~20.04`; do not infer GCC 10 compilation.

The Python pip distribution 24.3.1 is also separate from Debian `python3-pip`
`20.0.2-5ubuntu1.11`. Recover the active distribution/tool paths and ownership
before deciding how the offline installation reproduces that layering.
The differing Ninja version formats are not independently proof of a mismatch;
its resolved executable path, owning artifact and binary hash remain missing.

Twenty-one exact Debian role versions are in the manifest, including
`build-essential=12.8ubuntu1.1`, `cmake=3.16.3-1ubuntu1.20.04.1`,
`libeigen3-dev=3.3.7-2`, `libc6=2.31-0ubuntu9.18` and
`pkg-config=0.29.1-0ubuntu4`. These are **not** the complete Debian dependency
closure or hashes of `.deb` artifacts.

Five role queries failed: `libegl1`, `libegl1-mesa-dev`, `libgl1-mesa-dev`, `libsm6`,
`libxrender1`. `ninja-build` returned status zero with an empty version and is
therefore **not pinned**. These results do not prove the corresponding libraries
are absent: they may be supplied differently or bundled. Do not invent versions,
install proposed roles blindly or claim the GL plugin is validated by CUDA use.
Follow up read-only with `dpkg-query -W` including `${db:Status-Status}`,
`${binary:Package}` and `${Version}` for those six roles, and query the resolved
Ninja executable's package ownership/hash. Record failed lookups.

## Complete recorded Python closure

The union of training runtime roots reaches **99** distributions. Adding the
existing build roots gives **102**; pip, pybind11 and wheel are the three nodes
not already reached by runtime dependencies. Both native extensions remain
source-built at their separately pinned revisions: **100 prebuilt distribution
artifacts plus 2 source-built distributions** are required for this recorded
selection. This is a version/metadata closure, not yet a verified wheel closure.
The 100-wheel count describes the approved definition's planned artifact set;
the inventory does not establish the original installation method for those packages.

Raytracing 0.1.0 declares eight active, unconditional metadata requirements:
`ninja`, `trimesh`, `opencv-python`, `torch`, `numpy`, `tqdm`, `matplotlib`,
`dearpygui`. Its transitive closure contains **21** nodes including raytracing.
Trimesh and Dear PyGui must not be omitted merely because no GUI/demo is invoked
by training. Do not strip setup metadata or upgrade them to satisfy a resolver.
Compare the recovered installed metadata with the exact pinned source's setup
before building; installed metadata alone does not prove an unmodified source build.

Open3D 0.16.0 declares these active requirements: `numpy>=1.18.0`, `dash>=2.6.0`,
`nbformat==5.5.0`, `configargparse`, `ipywidgets>=7.6.0`, `addict`, `pillow>=8.2.0`,
`matplotlib>=3`, a second constraint `numpy>1.15`, `pandas>=1.0`, `pyyaml>=5.4.1`,
`scikit-learn>=0.21`, `tqdm`, `pyquaternion`. Its closure contains **80** nodes
including Open3D: Dash/Flask, Plotly, IPython/widgets/Jupyter metadata and their
dependencies belong in the recorded environment even for headless training.
All exact versions/edges, including both NumPy constraints, are retained in the
manifest. Neither optional extras nor unrelated whole-image packages are added.

Torchvision and torchaudio are not nodes in this collector's selected roots;
their absence here does not prove they are uninstalled. The prior historical
declarations remain unverified, with an explicit omit/retain decision still needed.

## Wheel and source artifact boundary

**No wheels were downloaded or locally hash-verified.** The three prior publisher
checksum candidates (NumPy/Open3D/PyMCubes) remain in
[artifact_candidates.json](artifact_candidates.json). Two new publisher-only
candidates are recorded alongside the installed metadata:

- [Dear PyGui 2.3.1](https://pypi.org/project/dearpygui/2.3.1/):
  `dearpygui-2.3.1-cp38-cp38-manylinux1_x86_64.whl`, SHA-256
  `cd7d63969fb174388bc8c99357fc71c796a8b799d53c03009c0d107fdbccdcad`.
- [Trimesh 4.12.2](https://pypi.org/project/trimesh/4.12.2/):
  `trimesh-4.12.2-py3-none-any.whl`, SHA-256
  `b5b5afa63c5272345f2858f7676bc8c217dc8a89f4fadf6193fe10a81b5ff2aa`.

These are publisher listings, not proof of the reference-installed artifacts,
download success, binary compatibility or verified attestations. All 100 actual
wheel byte hashes/identities remain unverified. Check CPython 3.8/ABI/platform,
glibc tags and wheel `METADATA` including full markers/extras; retain the observed
pins and fail rather than resolving newer dependencies. Torch's CUDA wheel hash
and every future nvdiffrast/raytracing build-wheel/plugin hash remain missing.

Do not pass this reference JSON directly to `pip` or copy its package objects
into `resolved.python_packages`. A later reviewed resolution needs exact version
strings excluding the two source-built distributions, an actual hashed wheel
lock, every file hash, staged source/workspace inventories and an explicit base
decision. `resolved.example.json` remains deliberately incomplete and unreviewed.
The original `reference.json` root-name list is retained as the existing validator
contract; it no longer describes missing installed versions. No new runtime
configuration or package-selection behavior is introduced by these evidence files.
The added manifest is host-side review evidence, not a new dependency read by
the build validator; the six-file staged definition inventory is unchanged.

## CUDA base candidate: unverified and not historical identity

Candidate for `nvidia/cuda:11.1.1-devel-ubuntu20.04`:
`sha256:299cd42d4c364516bd120fa8097b0397cac6c962e3761ef822656e5c5a25ea40`.
The researcher reports a Docker Hub listing. Codex attempted public Hub tag API
and registry manifest metadata: web access was unavailable and direct sandbox
HTTP requests failed DNS resolution. No Docker/socket/privilege workaround,
image pull or layer download was attempted. **Registry presence, media type,
platform and tag relationship are not independently verified here.**

If permitted in the regular terminal, the following standard anonymous registry
metadata request prints only manifest identity/platform information. Its scoped
anonymous token stays in memory and is never printed; no private credentials,
Docker configuration, image layers or package files are read/downloaded:

```sh
python3 -B - <<'PY'
import hashlib, json, urllib.parse, urllib.request
candidate = 'sha256:299cd42d4c364516bd120fa8097b0397cac6c962e3761ef822656e5c5a25ea40'
query = urllib.parse.urlencode({'service': 'registry.docker.io', 'scope': 'repository:nvidia/cuda:pull'})
with urllib.request.urlopen('https://auth.docker.io/token?' + query, timeout=20) as response:
    token = json.load(response)['token']
accept = ', '.join(['application/vnd.oci.image.index.v1+json', 'application/vnd.oci.image.manifest.v1+json',
                    'application/vnd.docker.distribution.manifest.list.v2+json', 'application/vnd.docker.distribution.manifest.v2+json'])
for reference in [candidate, '11.1.1-devel-ubuntu20.04']:
    request = urllib.request.Request('https://registry-1.docker.io/v2/nvidia/cuda/manifests/' + reference,
                                    headers={'Authorization': 'Bearer ' + token, 'Accept': accept})
    with urllib.request.urlopen(request, timeout=20) as response:
        body = response.read()
        header = response.headers.get('Docker-Content-Digest')
    actual = 'sha256:' + hashlib.sha256(body).hexdigest()
    assert header is None or header == actual
    if reference == candidate:
        assert actual == candidate
    manifest = json.loads(body)
    print(json.dumps({'reference': reference, 'body_digest': actual, 'header_digest': header,
                      'media_type': manifest.get('mediaType'),
                      'children': [{'digest': m['digest'], 'platform': m.get('platform')}
                                   for m in manifest.get('manifests', [])],
                      'config_digest': manifest.get('config', {}).get('digest'),
                      'layer_count': len(manifest.get('layers', []))}, sort_keys=True))
PY
```

For an index, distinguish the index digest from the linux/amd64 child manifest.
For a single manifest, the architecture may still require image-configuration
metadata. A current tag can differ from a candidate digest. Even a verified
registry digest/tag association **does not prove the original build used it**:
require trusted historical build/base metadata or explicit review as a newly
identified candidate base. `reference.image.base_digest` stays null, and no
candidate has been inserted into a reviewed resolution or made a Docker default.

## Remaining gates and the smallest next step

1. Complete permitted registry metadata verification and historical/candidate
   base review. Bind follow-up diagnostics to the current container/image identity
   without retroactively claiming that the supplied file recorded it. Recover the
   six ambiguous Debian roles, active Ninja/distribution
   locations and complete Debian dependency versions against the approved base.
   The 21 role pins do not solve Depends/Pre-Depends/versioned alternatives,
   virtual packages or the base-relative `.deb` selection.
2. Obtain the exact raytracing source export/setup audit (including Eigen lookup,
   flags, licences and metadata), an explicitly selected paper commit export and
   retained native source hashes. Do not execute setup to inspect it or substitute
   sources. Prior nvdiffrast export evidence remains source-only.
3. **Proposed separate artifact-acquisition approval:** start with a small exact
   wheel-only batch: NumPy, PyMCubes, Trimesh and Dear PyGui, using the published
   candidate identities above/in the manifests. Write into a new external owned
   staging directory; verify SHA-256, names/versions, `METADATA`, tags and licences
   without importing/installing them. Record provenance/failures and stop on any
   mismatch. This validates the acquisition procedure before the much larger
   Torch/Open3D wheels and the remaining closure. No acquisition occurred here.
4. Only after the exact base-relative Debian set is reviewed, acquire the missing
   `.deb` bytes and independent hashes plus the remaining exact wheels. Then
   prepare a complete resolution for review and the mandatory host preflight.
   Preserve `reviewed: false` until that actual artifact set is reviewed. First
   image builds, environment changes and native/GPU smoke tests remain separate.

Run the existing checks from [PREREQUISITES.md](PREREQUISITES.md#cpu-evidence-and-the-separately-approved-first-build).
Results on available Python 3.9.12: **28 Docker/prerequisite tests passed, none
failed/skipped; 97 infrastructure tests passed, none failed, 3 skipped** for absent
native OpenCV, OpenCV/transforms3d crop and Open3D backends. Static checks pass for
58 Python files, five offline Docker RUN shell commands, staged-file inventory
and isolation policy. JSON, documented shell/registry-script syntax and whitespace
checks pass; documented registry commands were parsed without execution. Actual
Python 3.8 and native/GPU compatibility are still untested in Codex.
Metadata validation is CPU-only and is not native training, a new measurement of
scientific performance or a claim of complete historical reproducibility.
