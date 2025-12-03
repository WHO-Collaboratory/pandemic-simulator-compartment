FROM python:3.13-slim
LABEL maintainer="Christian Osborne cosborne@ruvos.com"

# To use this Dockerfile:
# 1. `docker build . -t compartment-model`
# 2. `docker run -v $(pwd)/reference:/opt/reference -v $(pwd)/results:/opt/results -e MODE=local -e CONFIG_FILE=/opt/reference/pansim-config.json -e OUTPUT_FILE=/opt/results/example.json compartment-model

# Note that to use configs or save results to your local filesystem, you must mount the appropriate directories.

# MODEL_DIR sets the model to run.
# Currently, we expect every model to have a main.py file that takes the MODE argument. 
# It is recommended to also take the CONFIG_FILE argument.
# To use a different model, change the MODEL_DIR to the appropriate model directory.
ENV MODEL_DIR=/opt/pandemic-simulator-compartment/compartment/examples/covid_jax_model/

ENV MODE=local
ENV CONFIG_FILE=/opt/reference/pansim-config.json

ENV ENVIRONMENT=dev

ENV PYTHONUNBUFFERED=1

WORKDIR /opt/pandemic-simulator-compartment
COPY . /opt/pandemic-simulator-compartment

RUN pip install --no-cache-dir -e .

# Use exec form with shell to allow environment variable substitution
# This allows runtime overrides via docker run -e flags
CMD ["sh", "-c", "cd $MODEL_DIR && python main.py \
    --mode ${MODE} \
    --config_file ${CONFIG_FILE} \
    ${OUTPUT_FILE:+--output_file $OUTPUT_FILE} \
    ${SIMULATION_JOB_ID:+--simulation_job_id $SIMULATION_JOB_ID}"]