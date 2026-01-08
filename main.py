import functions_framework
from flask import jsonify
import sys

# Import your actual logic from the subfolder
# This assumes your folder is compartment/models/covid_jax_model/main.py
from compartment.models.covid_jax_model.main import run_local

@functions_framework.http
def gcp_handler(request):
    """
    GCP Entry Point: Receives HTTP request and calls your local simulation.
    """
    # 1. Parse the input from the request (e.g., via curl or browser)
    request_json = request.get_json(silent=True) or {}
    
    # Get the config path from JSON, or use your default
    config_path = request_json.get(
        "config_file", 
        "compartment/models/covid_jax_model/example-config.json"
    )

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
        # If your function returns data (like a dict), capture it here
        results = run_local() 
        
        return jsonify({
            "status": "success",
            "config_used": config_path,
            "data": results
        }), 200

    except Exception as e:
        # If the simulation fails, GCP will show this in the logs
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
