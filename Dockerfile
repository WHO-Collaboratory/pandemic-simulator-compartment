FROM python:3.13-slim
LABEL maintainer="Christian Osborne cosborne@ruvos.com"

ENV MODEL_DIR=/opt/pandemic-simulator-compartment/compartment/examples/covid_jax_model/

ENV MODE=local
ENV CONFIG_FILE=/opt/pandemic-simulator-compartment/reference/pansim-config.json

ENV PYTHONUNBUFFERED=1

WORKDIR /opt/pandemic-simulator-compartment
COPY . /opt/pandemic-simulator-compartment

RUN pip install --no-cache-dir -e .

# Use exec form with shell to allow environment variable substitution
# This allows runtime overrides via docker run -e flags
CMD ["sh", "-c", "cd $MODEL_DIR && python main.py \
    --mode ${MODE} \
    --config_file ${CONFIG_FILE} \
    ${OUTPUT_FILE:+--output_file $OUTPUT_FILE}"]