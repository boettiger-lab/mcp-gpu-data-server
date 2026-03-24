FROM nvcr.io/nvidia/rapidsai/base:25.02-cuda12.8-py3.12

WORKDIR /app

# Install Python dependencies.
# kvikio is pre-installed in the RAPIDS base image (conda) — skip it here.
COPY requirements.txt .
RUN grep -v '^kvikio' requirements.txt | pip install --no-cache-dir -r /dev/stdin

# Copy application code
COPY . .

EXPOSE 8000

CMD ["python", "server.py"]
