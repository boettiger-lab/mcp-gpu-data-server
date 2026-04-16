# Stage 1: fetch upstream server files at a pinned ref
FROM alpine/git:latest AS upstream
ARG UPSTREAM_REF=main
RUN git clone --depth 1 --branch ${UPSTREAM_REF} \
    https://github.com/boettiger-lab/mcp-data-server.git /upstream

# Stage 2: GPU server on RAPIDS base
FROM nvcr.io/nvidia/rapidsai/base:25.02-cuda12.8-py3.12

WORKDIR /app

# Install kvikio Python bindings (libkvikio C lib is in base; Python bindings
# are a separate conda package not auto-included in rapidsai/base).
RUN conda install -c rapidsai "kvikio=25.02.01=cuda12_py312_250227_g8fecf06_0" -y --quiet

# Copy upstream server and install its requirements
COPY --from=upstream /upstream /app/
RUN grep -v '^kvikio' requirements.txt | pip install --no-cache-dir -r /dev/stdin

# GPU-specific Python dependencies
COPY requirements-gpu.txt .
RUN pip install --no-cache-dir -r requirements-gpu.txt

# Override with GPU query backend (replaces the built-in DuckDB engine via plugin hook)
COPY query_backend.py sql_rewriter.py h3_functions.py ./

EXPOSE 8000

CMD ["python", "server.py"]
