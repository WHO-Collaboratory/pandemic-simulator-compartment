FROM python:3.13-slim
LABEL maintainer="Christian Osborne cosborne@ruvos.com"

#TODO should files be optional?
ENV MODEL_DIR=/opt/pandemic-simulator-compartment/compartment/examples/covid_jax_model/
ENV CONFIG_FILE=/opt/pandemic-simulator-compartment/reference/pansim-config.json
ENV OUTPUT_FILE=/opt/pandemic-simulator-compartment/results/example-run.json

ENV PYTHONUNBUFFERED=1

WORKDIR /opt/pandemic-simulator-compartment
COPY . /opt/pandemic-simulator-compartment

RUN pip install --no-cache-dir -e .

CMD ["sh", "-c", "cd $MODEL_DIR && python main.py $CONFIG_FILE $OUTPUT_FILE"]