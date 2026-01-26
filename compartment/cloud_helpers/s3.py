import json
from datetime import datetime
import boto3
import logging
from compartment.helpers import convert_dates
from compartment.validation import load_simulation_config

def s3_write_helper(s3_client, bucket_name, key, payload):
    payload = convert_dates(payload)
    s3_client.put_object(
        Bucket=bucket_name,
        Key=key,
        Body=json.dumps(payload, indent=2).encode("utf-8"),
        ContentType="application/json",
    )

def write_to_s3(s3_client, bucket_name, payload, simulation_job_id):
    statuses = []
    base_path = f'{simulation_job_id}/{payload.get('id')}'
    
    # Write parent_admin_total
    parent_admin_total = payload.pop('parent_admin_total')
    s3_key = f'{base_path}/{parent_admin_total.get('admin_unit_id')}.json'
    try:
        s3_write_helper(s3_client, bucket_name, s3_key, parent_admin_total)
        statuses.append({'key': s3_key, 'status': 'success'})
    except Exception as e:
        statuses.append({'key': s3_key, 'status': 'error', 'error': str(e)})

    admin_zones = payload.pop('admin_zones')
    # Write metadata (excluded children and parent time series)
    s3_key = f'{base_path}/metadata.json'
    try:
        s3_write_helper(s3_client, bucket_name, s3_key, payload)
        statuses.append({'key': s3_key, 'status': 'success'})
    except Exception as e:
        statuses.append({'key': s3_key, 'status': 'error', 'error': str(e)})

    # Write each admin_zone
    for admin_zone in admin_zones:
        s3_key = f"{base_path}/{admin_zone.get('admin_unit_id')}.json"
        try:
            s3_write_helper(s3_client, bucket_name, s3_key, admin_zone)
            statuses.append({'key': s3_key, 'status': 'success'})
        except Exception as e:
            statuses.append({'key': s3_key, 'status': 'error', 'error': str(e)})

    return statuses

def upload_validation_result_to_s3(simulation_job_id: str, result: dict, success: bool, environment: str):
    """
    Store validation result in S3, labeled by success/failure, with a timestamp; returns s3 path.
    """
    bucket = f'compartmental-validation-results-{environment}'
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    label = "success" if success else "failure"
    key = f"{simulation_job_id}/validation_{label}_{timestamp}.json"
    s3 = boto3.client("s3", region_name="us-east-1")
    s3_write_helper(s3, bucket, key, result)
    return f"s3://{bucket}/{key}"

def record_and_upload_validation(
    simulation_job_id,
    config,
    disease_type,
    environment=None,
    mode='local'
):
    validation_result = None
    validation_success = False
    try:
        cleaned_config = load_simulation_config(config, disease_type) 
        validation_result = {
            "event": "validation_success",
            "job_id": simulation_job_id,
            "schema": type(cleaned_config).__name__,
            "timestamp": datetime.now().isoformat(),
            "payload": config['data']['getSimulationJob'],
            "model_dump": cleaned_config.model_dump() if hasattr(cleaned_config, "model_dump") else None,
        }
        validation_success = True
    except Exception as e:
        from pydantic import ValidationError
        err_obj = e if isinstance(e, ValidationError) else None
        validation_result = {
            "event": "validation_failure",
            "job_id": simulation_job_id,
            "schema": getattr(e, 'model', None) or "UnknownConfig",
            "timestamp": datetime.now().isoformat(),
            "payload": config['data']['getSimulationJob'],
            "error": err_obj.errors() if err_obj else str(e),
            "error_str": str(e),
        }
        validation_success = False
    # Write to S3 if in cloud mode
    if mode == 'cloud' and environment and simulation_job_id:
        s3_path = upload_validation_result_to_s3(
            simulation_job_id,
            validation_result,
            validation_success,
            environment
        )
        print(f"Validation result persisted to S3 at: {s3_path}")
    # else:
    #     # In local mode, surface the validation result to logs for debugging
    #     try:
    #         logging.getLogger(__name__).info(
    #             "Validation result: %s",
    #             json.dumps(validation_result, default=str, indent=2),
    #         )
    #     except Exception:
    #         # Fallback to plain print if logging/jsonification fails
    #         print("Validation result:", validation_result)
    return validation_success, cleaned_config