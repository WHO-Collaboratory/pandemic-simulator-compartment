import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from compartment.cloud_helpers.graphql_queries import (
    ADMIN_UNITS_BY_SIM_JOB_QUERY,
    SEARCH_ADMIN_UNITS_QUERY,
)
from compartment.helpers import convert_dates

# --------------------------------------------------
# Helper Functions: GQL query
# --------------------------------------------------
# TEMPORARY GLOBAL
GRAPHQL_QUERY = "query GetSimulationJobById($id: ID!) {\n  getSimulationJob(id: $id) {\n    id\n    admin_0_id\n    admin_1_id\n    admin_2_id\n    createdAt\n    disease_id\n    end_date\n    owner\n    selected_infected_population\n    selected_population\n    simulation_name\n    simulation_type\n    start_date\n    tag_id\n    time_steps\n    updatedAt\n    Admin0 {\n      id\n      admin_0_name\n      admin_0_code\n      admin_0_viz_name\n      center_lat\n      center_lon\n    }\n    Admin1 {\n      id\n      admin_1_name\n      admin_1_code\n      admin_1_viz_name\n      center_lat\n      center_lon\n    }\n    Admin2 {\n      id\n      admin_2_name\n      admin_2_code\n      admin_2_alt_name\n      center_lat\n      center_lon\n    }\n    Disease {\n      id\n      createdAt\n      disease_name\n      disease_nodes {\n        type\n        data {\n          alias\n          label\n        }\n        id\n      }\n      immunity_period\n      interactions_per_period\n      intervention_nodes {\n        data {\n          adherence_max\n          adherence_min\n          alias\n          end_date\n          end_threshold\n          end_threshold_node_id\n          label\n          start_date\n          start_threshold\n          start_threshold_node_id\n        }\n        id\n        type\n      }\n      model_type\n      transmission_edges {\n        data {\n          transmission_rate\n        }\n        id\n        source\n        target\n        type\n      }\n      updatedAt\n    }\n    interventions {\n      adherence_max\n      adherence_min\n      end_date\n      end_threshold\n      end_threshold_node_id\n      id\n      label\n      start_date\n      start_threshold\n      start_threshold_node_id\n      type\n      transmission_percentage\n      hour_reduction\n    }\n    case_file {\n      admin_zones {\n        id\n        admin_code\n        admin_level\n        center_lat\n        center_lon\n        infected_population\n        name\n        osm_id\n        population\n        viz_name\n      }\n      demographics {\n        age_0_17\n        age_18_55\n        age_56_plus\n      }\n    }\n  }\n}"


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


def _gql_query(job_params: dict, query: str, variables: dict) -> dict:
    """Execute a GraphQL query and return the parsed response data."""
    headers = {"Content-Type": "application/json"}
    api_key = job_params.get("GRAPHQL_APIKEY")
    endpoint = job_params.get("GRAPHQL_ENDPOINT")

    if api_key:
        headers["x-api-key"] = api_key

    try:
        response = requests.post(
            endpoint,
            json={"query": query, "variables": variables},
            headers=headers,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise ValueError(f"GraphQL query failed: {str(e)}") from e


def get_simulation_job_admin_units(job_params: dict, simulation_job_id: str) -> list:
    """
    Fetch all SimulationJobAdminUnit records for a simulation job.

    Handles pagination via nextToken to ensure all records are returned.

    Args:
        job_params: Dict with GRAPHQL_APIKEY and GRAPHQL_ENDPOINT.
        simulation_job_id: The simulation job ID to query by.

    Returns:
        List of SimulationJobAdminUnit dicts, or empty list if none found.
    """
    all_items = []
    next_token = None

    while True:
        variables = {
            "simulation_job_id": simulation_job_id,
            "limit": 1000,
        }
        if next_token:
            variables["nextToken"] = next_token

        data = _gql_query(job_params, ADMIN_UNITS_BY_SIM_JOB_QUERY, variables)
        result = data.get("data", {}).get(
            "simulationJobAdminUnitsBySimulationJobId", {}
        )
        items = result.get("items", [])
        all_items.extend(items)

        next_token = result.get("nextToken")
        if not next_token:
            break

    logger.info(
        f"Fetched {len(all_items)} SimulationJobAdminUnit records for job {simulation_job_id}"
    )
    return all_items


def get_admin_unit_references(job_params: dict, admin_unit_ids: list) -> dict:
    """
    Batch-fetch AdminUnit reference records by ID using searchAdminUnits.

    Builds an OR filter of id.eq clauses to fetch all referenced AdminUnit
    records in a single request.

    Args:
        job_params: Dict with GRAPHQL_APIKEY and GRAPHQL_ENDPOINT.
        admin_unit_ids: List of AdminUnit ID strings to fetch.

    Returns:
        Dict keyed by AdminUnit ID mapping to the AdminUnit record dict.
        E.g. {"GBR.1_1": {"id": "GBR.1_1", "center_lat": 51.5, ...}, ...}
    """
    if not admin_unit_ids:
        return {}

    # Build OR filter: {or: [{id: {eq: "ID1"}}, {id: {eq: "ID2"}}, ...]}
    or_clauses = [{"id": {"eq": uid}} for uid in admin_unit_ids]
    variables = {
        "filter": {"or": or_clauses},
        "limit": len(admin_unit_ids),
    }

    data = _gql_query(job_params, SEARCH_ADMIN_UNITS_QUERY, variables)
    items = data.get("data", {}).get("searchAdminUnits", {}).get("items", [])

    # Key by ID for O(1) lookup during transformation
    refs = {item["id"]: item for item in items if item.get("id")}

    logger.info(
        f"Fetched {len(refs)} AdminUnit references for {len(admin_unit_ids)} requested IDs"
    )
    return refs