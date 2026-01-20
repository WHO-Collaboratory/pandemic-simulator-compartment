#import functions_framework
from flask import jsonify
import sys

from google.cloud import storage
import tempfile
import os
from datetime import datetime

# Import your actual logic from the subfolder
# This assumes your folder is compartment/models/covid_jax_model/main.py
from compartment.models.covid_jax_model.main import run_local

from datetime import date, datetime

def convert_dates(obj):
    if isinstance(obj, dict):
        return {k: convert_dates(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_dates(v) for v in obj]
    elif isinstance(obj, (date, datetime)):
        return obj.isoformat()
    else:
        return obj

#@functions_framework.http
def gcp_handler():
    """
    GCP Entry Point: Receives HTTP request and calls your local simulation.
    """
    # 1. Parse the input from the request (e.g., via curl or browser)
    #request_json = request.get_json(silent=True) or {}
    
    # Get the config path from JSON, or use your default
    # config_path = request_json.get(
    #     "config_file", 
    #     "reference/novel-respiratory-advanced-example-config.json"
    # )
    config_path = "reference/novel-respiratory-advanced-example-config.json"

    # 2. Prepare the arguments for your function
    # We still mock sys.argv because your 'run_local' likely 
    # internally calls argparse.parse_args()
    sys.argv = [
        "compartment.models.covid_jax_model.main",
        "--mode", "local",
        "--config_file", config_path
    ]

    try:
        # 3. Call your function
        results = run_local()
        
        # 4. Write results to a temporary file
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as tmpfile:
            import json
            json.dump(convert_dates(results), tmpfile, indent=2)
            tmpfile.flush()
            local_path = tmpfile.name
        
        # 5. Upload to GCS
        bucket_name = "tpu-jax-bucket"
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        blob_name = f"simulation_results/{os.path.basename(config_path)}_{timestamp}.json"
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(local_path)
        os.remove(local_path)

        # 6. Return GCS URI
        gcs_uri = f"gs://{bucket_name}/{blob_name}"
        return {
            "status": "success",
            "config_used": config_path,
            "gcs_uri": gcs_uri,
            "status_code": 200
        }

    # try:
    #     # 3. Call your function
    #     # If your function returns data (like a dict), capture it here
    #     results = run_local() 
        
    #     return jsonify({
    #         "status": "success",
    #         "config_used": config_path,
    #         "data": results
    #     }), 200

    except Exception as e:
        # If the simulation fails, GCP will show this in the logs
        return {
            "status": "error",
            "message": str(e),
            "status_code": 500
        }

if __name__ == "__main__":
    result = gcp_handler()
    print(result)