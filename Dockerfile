# Runs the gateway: `nashgate route --config /config/config.yaml`.
# Mount your own config and pass backend API keys as env vars — see
# docs/example.config.yaml and README.md#the-gateway.

FROM python:3.11-slim

RUN useradd --create-home --uid 1000 nashgate
WORKDIR /app

# CPU-only torch, installed separately and pinned to PyTorch's CPU
# index before anything else. The policy is a tiny MLP — there's no
# GPU work in the gateway — and the default PyPI wheel pulls in
# several GB of CUDA libraries for nothing.
RUN pip install --no-cache-dir "torch>=2.3" --index-url https://download.pytorch.org/whl/cpu

COPY --chown=nashgate:nashgate pyproject.toml README.md ./
COPY --chown=nashgate:nashgate nashgate ./nashgate

# --no-deps so this doesn't re-resolve torch off the default index;
# the rest of the dependencies are ordinary PyPI packages.
RUN pip install --no-cache-dir --no-deps . \
    && pip install --no-cache-dir \
        "fastapi>=0.115" "uvicorn>=0.30" "httpx>=0.27" "typer>=0.12" \
        "pyyaml>=6.0" "numpy>=1.26"

USER nashgate

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/healthz', timeout=2)" || exit 1

ENTRYPOINT ["nashgate", "route"]
CMD ["--config", "/config/config.yaml", "--host", "0.0.0.0", "--port", "8000"]
