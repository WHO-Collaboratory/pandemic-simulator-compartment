import requests
import logging
from compartment.helpers import setup_logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from compartment.helpers import convert_dates

setup_logging()
logger = logging.getLogger(__name__)

# --------------------------------------------------
# Helper Functions: GQL query
# --------------------------------------------------


def get_simulation_job(job_params=dict, GRAPHQL_QUERY=str) -> dict:
    """Hit GraphlQL to get full simualtion job params"""
    SIMULATION_JOB_ID = job_params.get("SIMULATION_JOB_ID")
    GRAPHQL_APIKEY = job_params.get("GRAPHQL_APIKEY")
    GRAPHQL_ENDPOINT = job_params.get("GRAPHQL_ENDPOINT")

    headers = {"Content-Type": "application/json"}
    variables = {"id": SIMULATION_JOB_ID}

    if GRAPHQL_APIKEY:
        headers["x-api-key"] = GRAPHQL_APIKEY

    try:
        response = requests.post(
            GRAPHQL_ENDPOINT,
            json={"query": GRAPHQL_QUERY, "variables": variables},
            headers=headers,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise ValueError(f"GraphQL validation failed: {str(e)}") from e


def write_to_gql(job_params, results):
    """Write results of model to GraphQL and return responses"""

    responses = {"child_responses": [], "parent_response": None}

    # Mutation for child entities (admin_zones)
    child_query = """
        mutation MyMutation($input: CreateAdminZoneTimeSeriesInput!) {
            createAdminZoneTimeSeries(input: $input) {
                id
            }
        }
    """
    child_result = results["admin_zones"]

    responses["child_responses"] = parallel_write(
        job_params=job_params, inputs=child_result, query=child_query, max_workers=4
    )

    response = gql_write_helper(job_params, results["parent_admin_total"], child_query)
    responses["child_responses"].append(response)

    # Mutation for parent entity (simulation job result)
    parent_query = """
        mutation CreateSimulationJobResult($input: CreateSimulationJobResultInput!) {
            createSimulationJobResult(input: $input) {
                id
                simulation_job_id
            }
        }
    """
    results.pop("admin_zones")  # Remove child data before parent write
    results.pop("parent_admin_total")  # Remove parent admin data before parent write
    responses["parent_response"] = gql_write_helper(job_params, results, parent_query)

    return responses  # Return collected responses


def gql_write_helper(job_params, results, query):
    """Write resutls of model to GraphQL"""

    # Ensure JSON-serializable payload (dates, datetimes, ndarrays → strings/lists)
    safe_results = convert_dates(results)

    payload = {
        "query": query,
        "variables": {"input": safe_results},
    }

    headers = {"Content-Type": "application/json"}
    api_key = job_params.get("GRAPHQL_APIKEY")
    GRAPHQL_ENDPOINT = job_params.get("GRAPHQL_ENDPOINT")
    if api_key:
        headers["x-api-key"] = api_key

    try:
        response = requests.post(GRAPHQL_ENDPOINT, json=payload, headers=headers)
        status_code = response.status_code
        response.raise_for_status()
        gql_response = response.json()
        return {"status_code": status_code, "gql_response": gql_response}
    except requests.RequestException as e:
        status_code = getattr(e.response, "status_code", None)
        return {"status_code": status_code, "error": str(e)}


def parallel_write(job_params, inputs, query, max_workers=8):
    """
    Parallel writes inputs to gql

    Args:
        job_params (dict): GraphQL job parameters
        inputs (list of dict): One dict per record to write
        query (str): GraphQL mutation string for writes
        max_workers (int): Number of threads

    Returns:
        list: Responses from the GraphQL API, in the same order as inputs.
    """
    responses = [None] * len(inputs)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(gql_write_helper, job_params, i, query): idx
            for idx, i in enumerate(inputs)
        }
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                responses[idx] = future.result()
            except Exception as e:
                responses[idx] = {"error": str(e)}
    return responses
