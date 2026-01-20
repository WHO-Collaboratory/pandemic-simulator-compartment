# 1. Use Python 3.10 or 3.11. 
# While 3.13 is out, 3.10 is the most stable and tested version for JAX TPU runtimes on GCP.
FROM python:3.13-slim

# 2. Install essential system tools for TPU communication
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# 3. Install JAX with TPU support. 
# We explicitly install libtpu at the same time.
RUN pip install --upgrade pip && \
    pip install "jax[tpu]>=0.4.6" \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# 4. DOWNLOAD THE TPU DRIVER (CRITICAL)
# JAX requires libtpu.so to be in /lib to talk to the v5e hardware.
RUN curl -L https://storage.googleapis.com/cloud-tpu-tpuvm-artifacts/libtpu/1.9.0/libtpu.so -o /lib/libtpu.so

# 5. CONFIGURE PJRT RUNTIME (REQUIRED FOR v5e)
# Vertex AI training on TPU v5e requires the PJRT runtime to be enabled via environment variables.
ENV PJRT_DEVICE=TPU
ENV NEXT_PLUGGABLE_DEVICE_USE_C_API=true
ENV TF_PLUGGABLE_DEVICE_LIBRARY_PATH=/lib/libtpu.so

# 6. Setup your app
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

# 7. Entrypoint
ENTRYPOINT ["python", "main.py"]