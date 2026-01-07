#!/bin/sh
# Entrypoint script that supports both AWS Lambda and local Docker runs

# Check if we're running in AWS Lambda (Lambda sets AWS_LAMBDA_RUNTIME_API)
if [ -n "${AWS_LAMBDA_RUNTIME_API}" ]; then
    # Lambda mode: use aws-lambda-ric
    # _HANDLER is set by Lambda based on function configuration.
    # If not set, derive it from MODEL_DIR environment variable
    if [ -z "${_HANDLER}" ]; then
        # Extract model name from MODEL_DIR
        # MODEL_DIR format: /opt/pandemic-simulator-compartment/compartment/models/covid_jax_model/
        # or /opt/pandemic-simulator-compartment/compartment/models/dengue_jax_model/
        MODEL_NAME=$(echo "${MODEL_DIR}" | sed 's|.*compartment/models/||' | sed 's|/$||')
        
        if [ -z "${MODEL_NAME}" ]; then
            echo "Error: Could not extract model name from MODEL_DIR: ${MODEL_DIR}" >&2
            exit 1
        fi
        
        _HANDLER="compartment.models.${MODEL_NAME}.main.lambda_handler"
        echo "Derived _HANDLER from MODEL_DIR: ${_HANDLER}"
    fi
    HANDLER="${_HANDLER}"
    exec /usr/local/bin/python -m awslambdaric "$HANDLER"
else
    # Use exec form to override defaults
    cd "${MODEL_DIR}" && python main.py \
        --mode "${MODE}" \
        --config_file "${CONFIG_FILE}" \
        ${OUTPUT_FILE:+--output_file $OUTPUT_FILE} \
        ${SIMULATION_JOB_ID:+--simulation_job_id $SIMULATION_JOB_ID}
fi
