#!/bin/sh
# Entrypoint script that supports both AWS Lambda and local Docker runs

# Check if we're running in AWS Lambda (Lambda sets AWS_LAMBDA_RUNTIME_API)
if [ -n "${AWS_LAMBDA_RUNTIME_API}" ]; then
    # Lambda mode: use aws-lambda-ric
    # _HANDLER is set by Lambda based on function configuration, or use default
    HANDLER="${_HANDLER:-compartment.models.covid_jax_model.main.lambda_handler}"
    exec /usr/local/bin/python -m awslambdaric "$HANDLER"
else
    # Use exec form to override defaults
    cd "${MODEL_DIR}" && python main.py \
        --mode "${MODE}" \
        --config_file "${CONFIG_FILE}" \
        ${OUTPUT_FILE:+--output_file $OUTPUT_FILE} \
        ${SIMULATION_JOB_ID:+--simulation_job_id $SIMULATION_JOB_ID}
fi
