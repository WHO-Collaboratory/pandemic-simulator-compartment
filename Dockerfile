# Use Python 3.10 as the base
FROM python:3.13-slim

# Install JAX with TPU support
RUN pip install "jax[tpu]>=0.4.6" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Copy your code and config
WORKDIR /app
COPY . .

# Install your project dependencies
RUN pip install -r requirements.txt

# The entry point that Vertex AI will call
ENTRYPOINT ["python", "-m", "compartment.models.covid_jax_model.main", "--mode", "local"]