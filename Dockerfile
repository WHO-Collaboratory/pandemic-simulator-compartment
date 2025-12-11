FROM python:3.13-slim
LABEL maintainer="Christian Osborne cosborne@ruvos.com"

# To use this Dockerfile:
# 1. `docker build . -t compartment-model`
# 2. `docker run -v $(pwd)/reference:/opt/reference -v $(pwd)/results:/opt/results -e MODE=local -e CONFIG_FILE=/opt/reference/pansim-config.json -e OUTPUT_FILE=/opt/results/example.json compartment-model

# Note that to use configs or save results to your local filesystem, you must mount the appropriate directories.


# MODEL_DIR sets the model to run.
# Currently, we expect every model to have a main.py file that takes the MODE argument. 
# To use a different model, change the MODEL_DIR to the appropriate model directory.
ARG MODEL_DIR=compartment/models/covid_jax_model/
ENV MODEL_DIR=/opt/pandemic-simulator-compartment/${MODEL_DIR}

ENV MODE=local
# The CONFIG_FILE environment variable points to a custom config file 
# Use the -v flag to mount the reference directory to your local filesystem.
ENV CONFIG_FILE=${MODEL_DIR}/example-config.json

# This is used for cloud mode.
ENV ENVIRONMENT=dev

ENV PYTHONUNBUFFERED=1

WORKDIR /opt/pandemic-simulator-compartment
COPY . /opt/pandemic-simulator-compartment

# Copy and make entrypoint script executable
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Install dependencies including Lambda Runtime Interface Client
RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir awslambdaric

# Default handler for Lambda (can be overridden via Lambda function config)
# Lambda will set _HANDLER automatically, but we provide a default
ENV _HANDLER=compartment.models.covid_jax_model.main.lambda_handler

# Use entrypoint script that handles both Lambda and local modes
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]