FROM nvcr.io/nvidia/rapidsai/base:25.02-cuda12.8-py3.12

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 8000

CMD ["python", "server.py"]
