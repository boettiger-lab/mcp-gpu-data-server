FROM nvcr.io/nvidia/rapidsai/base:25.02-cuda12.8-py3.12

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --extra-index-url=https://pypi.nvidia.com polars-gpu

# Copy application code
COPY . .

EXPOSE 8000

CMD ["python", "server.py"]
