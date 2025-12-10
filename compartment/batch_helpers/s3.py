import json
from datetime import datetime
import boto3
from compartment.helpers import convert_dates

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