# Definition only. Build ONLY from the reviewed generated external context.
# No default mutable base tag; recover and explicitly supply the historical base digest.
ARG PAPER_BASE_IMAGE
FROM ${PAPER_BASE_IMAGE}
ARG PAPER_BASE_IMAGE
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

# inputs/ contains offline hashed .debs, wheels, source exports and resolved.json.
# This context is deliberately not the repository or an original extension checkout.
COPY inputs/ /opt/paper-inputs/
COPY definition/ /opt/paper-environment/
COPY workspace/ /workspace/
RUN --network=none [[ "${PAPER_BASE_IMAGE}" =~ ^nvidia/cuda:11\.1\.1-devel-ubuntu20\.04@sha256:[a-f0-9]{64}$ ]] \
    && cd /opt/paper-inputs/apt \
    && awk 'NF != 2 || length($1) != 64 || $1 !~ /^[a-f0-9]+$/ || $2 !~ /^[A-Za-z0-9_.+%~-]+\.deb$/ || seen[$2]++ {exit 1} END {if (NR == 0) exit 1}' SHA256SUMS \
    && debs=(./*.deb) && test "${#debs[@]}" -eq "$(wc -l < SHA256SUMS)" \
    && sha256sum --strict --check SHA256SUMS \
    && dpkg --unpack ./*.deb \
    && DEBIAN_FRONTEND=noninteractive dpkg --configure -a

# Validate the full artifact set before running pip/setup code. No resolver/network fallback.
RUN --network=none /usr/bin/python3 -B /opt/paper-environment/validate.py --inputs /opt/paper-inputs --workspace /workspace --base-image "${PAPER_BASE_IMAGE}"
RUN --network=none /usr/bin/python3 -m pip install --no-cache-dir --no-index --no-deps --require-hashes \
    --find-links /opt/paper-inputs/wheels -r /opt/paper-inputs/python-complete.txt \
    && /usr/bin/python3 -B /opt/paper-environment/record.py --check-core

# Keep the devel compiler/toolkit for unchanged nvdiffrast lazy compilation at runtime.
# No plugin warm-up or GPU computation occurs in the proposed image build.
ENV TORCH_CUDA_ARCH_LIST=8.0
RUN --network=none /usr/bin/python3 -m pip wheel --no-cache-dir --no-index --no-deps --no-build-isolation \
    --wheel-dir /opt/paper-built-wheels /opt/paper-inputs/sources/nvdiffrast \
    && /usr/bin/python3 -m pip wheel --no-cache-dir --no-index --no-deps --no-build-isolation \
    --wheel-dir /opt/paper-built-wheels /opt/paper-inputs/sources/raytracing \
    && /usr/bin/python3 -m pip install --no-cache-dir --no-index --no-deps --force-reinstall /opt/paper-built-wheels/*.whl \
    && /usr/bin/python3 -m pip check \
    && /usr/bin/python3 -B /opt/paper-environment/record.py --write /opt/paper-environment/build-provenance.json

WORKDIR /workspace
ENV PYTHONDONTWRITEBYTECODE=1 \
    HOME=/outputs/cache/runtime/home \
    XDG_CACHE_HOME=/outputs/cache/runtime/xdg \
    XDG_CONFIG_HOME=/outputs/cache/runtime/config \
    XDG_DATA_HOME=/outputs/cache/runtime/data \
    MPLCONFIGDIR=/outputs/cache/runtime/matplotlib \
    TORCH_HOME=/outputs/cache/runtime/torch \
    TORCH_EXTENSIONS_DIR=/outputs/cache/runtime/torch_extensions \
    CUDA_CACHE_PATH=/outputs/cache/runtime/cuda \
    TMPDIR=/outputs/tmp
# No startup mkdir under /outputs: mounted RunStore must see an empty fresh root.
USER 1000:1000
ENTRYPOINT ["/usr/bin/python3"]
CMD ["--help"]
