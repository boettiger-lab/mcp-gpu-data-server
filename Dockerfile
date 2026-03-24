FROM nvcr.io/nvidia/rapidsai/base:25.02-cuda12.8-py3.12

WORKDIR /app

# Install kvikio Python bindings (libkvikio C lib is in base; Python bindings
# are a separate conda package not auto-included in rapidsai/base).
# Pin to the exact cuda12/py312 build that matches the base image.
RUN conda install -c rapidsai "kvikio=25.02.01=cuda12_py312_250227_g8fecf06_0" -y --quiet

# Install remaining Python dependencies (kvikio excluded — installed above).
COPY requirements.txt .
RUN grep -v '^kvikio' requirements.txt | pip install --no-cache-dir -r /dev/stdin

# Copy application code
COPY . .

EXPOSE 8000

CMD ["python", "server.py"]
