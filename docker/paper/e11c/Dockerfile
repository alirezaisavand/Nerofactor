# Sidecar-only derivative: host preflight and post-build base-layer check bind ID.
ARG E11C_BASE_IMAGE
FROM ${E11C_BASE_IMAGE}
COPY e11c/ /opt/e11c/
# The historical site-packages binary is never replaced; no install/setup/RUN here.
