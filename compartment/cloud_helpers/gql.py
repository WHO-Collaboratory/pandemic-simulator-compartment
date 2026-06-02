import json
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
# TEMPORARY GLOBAL
GRAPHQL_QUERY = "query GetSimulationJobById($id: ID!) {\n  getSimulationJob(id: $id) {\n    id\n    admin_unit_id\n    createdAt\n    disease_id\n    end_date\n    owner\n    selected_infected_population\n    selected_population\n    simulation_name\n    simulation_type\n    start_date\n    tag_id\n    time_steps\n    updatedAt\n    AdminUnit {\n      id\n      center_lat\n      admin_level\n      ParentAdminUnit {\n        id\n        center_lat\n        admin_level\n        ParentAdminUnit {\n          id\n          center_lat\n          admin_level\n        }\n      }\n    }\n    Disease {\n      id\n      createdAt\n      disease_name\n      disease_nodes {\n        type\n        data {\n          alias\n          label\n        }\n        id\n      }\n      immunity_period\n      interactions_per_period\n      intervention_nodes {\n        data {\n          adherence_max\n          adherence_min\n          alias\n          end_date\n          end_threshold\n          end_threshold_node_id\n          label\n          start_date\n          start_threshold\n          start_threshold_node_id\n        }\n        id\n        type\n      }\n      model_type\n      transmission_edges {\n        data {\n          transmission_rate\n        }\n        id\n        source\n        target\n        type\n      }\n      updatedAt\n    }\n    interventions {\n      adherence_max\n      adherence_min\n      end_date\n      end_threshold\n      end_threshold_node_id\n      id\n      label\n      start_date\n      start_threshold\n      start_threshold_node_id\n      type\n      transmission_percentage\n      hour_reduction\n    }\n    case_file {\n      admin_zones {\n        id\n        admin_code\n        admin_level\n        center_lat\n        center_lon\n        infected_population\n        name\n        osm_id\n        population\n        viz_name\n      }\n      demographics {\n        age_0_17\n        age_18_55\n        age_56_plus\n      }\n    }\n  }\n}"


def _cast_custom_field_value(value: str, value_type: str):
    """Cast a custom field string value to the appropriate Python type.

    SimulationJobCustomField stores all values as strings.  The CustomField
    metadata carries the ``value_type`` which tells us how to interpret it.

    Args:
        value: The raw string value from the database.
        value_type: One of the ValueType enum strings (RATE, DAYS, PERCENTAGE,
            NUMBER, COUNT, BOOLEAN, DATE, TEXT, SELECT, FLOAT, INTEGER, COORDINATE).

    Returns:
        The value cast to the correct Python type.
    """
    if value is None:
        return None

    int_types = {"DAYS", "COUNT", "INTEGER"}
    float_types = {"RATE", "PERCENTAGE", "NUMBER", "FLOAT", "COORDINATE"}

    if value_type in int_types:
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return None
    elif value_type in float_types:
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    elif value_type == "BOOLEAN":
        return value.lower() in ("true", "1", "yes")
    else:
        # DATE, TEXT, SELECT — keep as string
        return value


def get_simulation_job(job_params: dict, graphql_query: str) -> dict:
    """Fetch and assemble the complete simulation job config.

    Hits the main getSimulationJob query, then enriches the config with:
    - SimulationJobAdminUnits (population, temps, custom_field_values)
    - SimulationJobCustomFields (disease parameters injected into Disease dict)
    - SimulationJobDemographicGroups (injected into case_file.demographics)

    Args:
        job_params: Dict with SIMULATION_JOB_ID, GRAPHQL_APIKEY, GRAPHQL_ENDPOINT.
        graphql_query: The main getSimulationJob GraphQL query string.

    Returns:
        Complete config dict ready for validation.
    """
    simulation_job_id = job_params.get("SIMULATION_JOB_ID")

    # 1. Fetch the main simulation job
    config = _gql_query(job_params, graphql_query, {"id": simulation_job_id})

    # 2. Fetch SimulationJobAdminUnits and merge into case_file.admin_zones
    sim_job_admin_units = get_simulation_job_admin_units(job_params, simulation_job_id)

    if sim_job_admin_units:
        logger.info(
            f"Using {len(sim_job_admin_units)} admin units from SimulationJobAdminUnit table"
        )
        # Fetch geo references for these admin units
        admin_unit_ids = [u["admin_unit_id"] for u in sim_job_admin_units]
        admin_unit_refs = get_admin_unit_references(job_params, admin_unit_ids)

        # Merge: start with geo reference, overlay user-set values from join table
        admin_zones = []
        for unit in sim_job_admin_units:
            admin_unit_id = unit.get("admin_unit_id")
            ref = admin_unit_refs.get(admin_unit_id, {})

            zone = {**ref, **unit, "id": admin_unit_id}
            # Remove join-table metadata that isn't part of the zone
            zone.pop("admin_unit_id", None)

            # Unpack admin-zone custom field values (stored as AWSJSON)
            # so they become top-level keys on the zone dict, accessible
            # to CaseFileAdminZone (which has extra="allow").
            raw_cfv = zone.pop("custom_field_values", None)
            if raw_cfv:
                try:
                    parsed = (
                        json.loads(raw_cfv) if isinstance(raw_cfv, str) else raw_cfv
                    )
                    if isinstance(parsed, dict):
                        zone.update(parsed)
                except (json.JSONDecodeError, TypeError):
                    logger.warning(
                        f"Failed to parse custom_field_values for admin unit {admin_unit_id}"
                    )

            admin_zones.append(zone)

        config["data"]["getSimulationJob"]["case_file"]["admin_zones"] = admin_zones
    else:
        logger.info(
            "No SimulationJobAdminUnits found, using embedded case_file.admin_zones"
        )

    # 3. Fetch SimulationJobCustomFields and merge into config
    sim_job_custom_fields = get_simulation_job_custom_fields(
        job_params, simulation_job_id
    )
    disease_param_field_configs = []
    if sim_job_custom_fields:
        disease_dict = config["data"]["getSimulationJob"].setdefault("Disease", {})

        for cf in sim_job_custom_fields:
            custom_field = cf.get("CustomField") or {}
            category = custom_field.get("category")
            field_name = custom_field.get("name")
            raw_value = cf.get("value")

            if not field_name:
                continue

            metadata = custom_field.get("metadata") or {}
            value_type = metadata.get("value_type", "TEXT")
            typed_value = _cast_custom_field_value(raw_value, value_type)

            if category == "disease_parameter":
                # Inject into Disease dict so the auto-generated Pydantic
                # disease config picks it up by field name.
                disease_dict[field_name] = typed_value
                logger.debug(
                    f"Injected disease parameter '{field_name}' = {typed_value}"
                )

                # Preserve FieldConfig entries for uncertainty sampling.
                fc_section = cf.get("FieldConfig") or {}
                fc_items = fc_section.get("items", []) if fc_section else []
                for fc in fc_items:
                    if fc.get("has_variance"):
                        disease_param_field_configs.append(
                            {
                                "param": field_name,
                                "dist": (
                                    fc.get("distribution_type") or "uniform"
                                ).lower(),
                                "min": fc.get("min", 0),
                                "max": fc.get("max", 0),
                            }
                        )

        logger.info(
            f"Processed {len(sim_job_custom_fields)} custom field(s) for job {simulation_job_id}"
        )

    # Store disease parameter variance configs for uncertainty runs.
    # run_simulation.py reads this before validation (which ignores extra keys).
    if disease_param_field_configs:
        config["_disease_param_field_configs"] = disease_param_field_configs
        logger.info(
            f"Collected {len(disease_param_field_configs)} disease parameter variance config(s)"
        )

    # 4. Fetch SimulationJobDemographicGroups and merge into case_file.demographics
    sim_job_demo_groups = get_simulation_job_demographic_groups(
        job_params, simulation_job_id
    )
    if sim_job_demo_groups:
        demographics = {}
        for dg in sim_job_demo_groups:
            demo_group = dg.get("DemographicGroup") or {}
            group_name = demo_group.get("name")
            value = dg.get("value")
            if group_name is not None and value is not None:
                demographics[group_name] = float(value)

        if demographics:
            config["data"]["getSimulationJob"].setdefault("case_file", {})[
                "demographics"
            ] = demographics
            logger.info(
                f"Injected {len(demographics)} demographic group(s) into case_file.demographics"
            )

    return config


def _np_safe(o):
    """json.dumps default for numpy scalars inside AWSJSON payloads."""
    if hasattr(o, "item"):  # numpy float/int scalar -> native Python
        return o.item()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _to_awsjson(obj):
    """Serialize a dict/list to a JSON string for an AWSJSON GraphQL field."""
    return json.dumps(obj, default=_np_safe)


def _build_time_series_v2(time_series):
    """Repackage [{date, <comp>: {...}}] into [{date, compartments: <AWSJSON>}].

    Compartment keys are arbitrary, so the whole per-timestep compartment map is
    stored as a single AWSJSON string — letting any model (rodent/human, AMR,
    etc.) write its time series without a closed-field schema type. Derived from
    the already-built legacy ``time_series`` (which already carries the bespoke
    keys; only the old schema type rejected them on write).
    """
    v2 = []
    for rec in time_series or []:
        comps = {k: v for k, v in rec.items() if k != "date"}
        v2.append({"date": rec.get("date"), "compartments": _to_awsjson(comps)})
    return v2


# Compartment keys representable by the closed-field TimeSeriesStratification /
# SimulationDiseaseStateMetrics schema types. Legacy fields are pruned to these
# so the write never fails for models with bespoke compartments; the full data
# is preserved in the v2 (AWSJSON) fields.
_LEGACY_COMPARTMENT_KEYS = {
    "SV", "EV", "IV", "S", "E", "I", "H", "D", "R", "C", "Snot", "E2", "I2",
}


def _sanitize_legacy_time_series(time_series):
    """Drop compartment keys the closed TimeSeriesStratification type rejects."""
    return [
        {k: v for k, v in rec.items() if k == "date" or k in _LEGACY_COMPARTMENT_KEYS}
        for rec in time_series or []
    ]


def _add_v2_payloads(results):
    """Derive compartment-agnostic v2 fields and prune the legacy ones, in place.

    For each payload, build the full time_series_v2 / compartment_deltas_v2 (any
    compartment key, stored as AWSJSON), then prune the legacy time_series /
    compartment_deltas down to the keys the closed-field schema types accept — so
    the write never fails for models with bespoke compartments. Known models
    (COVID/dengue) keep their legacy data intact; novel models carry their full
    data only in v2. Single chokepoint for every builder path (deterministic
    3D/4D + uncertainty), which all converge on the same ``results`` shape here.
    """
    for zone in results.get("admin_zones", []) or []:
        if isinstance(zone, dict) and "time_series" in zone:
            zone["time_series_v2"] = _build_time_series_v2(zone["time_series"])
            zone["time_series"] = _sanitize_legacy_time_series(zone["time_series"])
    parent = results.get("parent_admin_total")
    if isinstance(parent, dict) and "time_series" in parent:
        parent["time_series_v2"] = _build_time_series_v2(parent["time_series"])
        parent["time_series"] = _sanitize_legacy_time_series(parent["time_series"])
    deltas = results.get("compartment_deltas")
    if deltas is not None:
        results["compartment_deltas_v2"] = _to_awsjson(deltas)
        if isinstance(deltas, dict):
            results["compartment_deltas"] = {
                k: v for k, v in deltas.items() if k in _LEGACY_COMPARTMENT_KEYS
            }
    return results


def write_to_gql(job_params, results):
    """Write results of model to GraphQL and return responses"""

    # Derive compartment-agnostic v2 payloads (time_series_v2 / compartment_deltas_v2)
    # before splitting/writing, so any compartment key is representable.
    _add_v2_payloads(results)

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
    from compartment.cloud_helpers.graphql_queries import ADMIN_UNITS_BY_SIM_JOB_QUERY

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


def get_simulation_job_custom_fields(job_params: dict, simulation_job_id: str) -> list:
    """
    Fetch all SimulationJobCustomField records for a simulation job.

    Handles pagination via nextToken to ensure all records are returned.
    Each item includes the linked CustomField template (name, category,
    metadata) and any FieldConfig records (for variance/uncertainty).

    Args:
        job_params: Dict with GRAPHQL_APIKEY and GRAPHQL_ENDPOINT.
        simulation_job_id: The simulation job ID to query by.

    Returns:
        List of SimulationJobCustomField dicts, or empty list if none found.
    """
    from compartment.cloud_helpers.graphql_queries import CUSTOM_FIELDS_BY_SIM_JOB_QUERY

    all_items = []
    next_token = None

    while True:
        variables = {
            "simulation_job_id": simulation_job_id,
            "limit": 1000,
        }
        if next_token:
            variables["nextToken"] = next_token

        data = _gql_query(job_params, CUSTOM_FIELDS_BY_SIM_JOB_QUERY, variables)
        result = data.get("data", {}).get(
            "simulationJobCustomFieldBySimulationJobId", {}
        )
        items = result.get("items", [])
        all_items.extend(items)

        next_token = result.get("nextToken")
        if not next_token:
            break

    logger.info(
        f"Fetched {len(all_items)} SimulationJobCustomField records for job {simulation_job_id}"
    )
    return all_items


def get_simulation_job_demographic_groups(
    job_params: dict, simulation_job_id: str
) -> list:
    """
    Fetch all SimulationJobDemographicGroup records for a simulation job.

    Handles pagination via nextToken to ensure all records are returned.
    Each item includes the linked DemographicGroup template (name,
    display_name, metadata).

    Args:
        job_params: Dict with GRAPHQL_APIKEY and GRAPHQL_ENDPOINT.
        simulation_job_id: The simulation job ID to query by.

    Returns:
        List of SimulationJobDemographicGroup dicts, or empty list if none found.
    """
    from compartment.cloud_helpers.graphql_queries import (
        DEMOGRAPHIC_GROUPS_BY_SIM_JOB_QUERY,
    )

    all_items = []
    next_token = None

    while True:
        variables = {
            "simulation_job_id": simulation_job_id,
            "limit": 1000,
        }
        if next_token:
            variables["nextToken"] = next_token

        data = _gql_query(job_params, DEMOGRAPHIC_GROUPS_BY_SIM_JOB_QUERY, variables)
        result = data.get("data", {}).get(
            "simulationJobDemographicGroupBySimulationJobId", {}
        )
        items = result.get("items", [])
        all_items.extend(items)

        next_token = result.get("nextToken")
        if not next_token:
            break

    logger.info(
        f"Fetched {len(all_items)} SimulationJobDemographicGroup records for job {simulation_job_id}"
    )
    return all_items


def get_admin_unit_references(job_params: dict, admin_unit_ids: list) -> dict:
    """
    Batch-fetch AdminUnit reference records by ID using searchAdminUnits.

    Args:
        job_params: Dict with GRAPHQL_APIKEY and GRAPHQL_ENDPOINT.
        admin_unit_ids: List of AdminUnit ID strings to fetch.

    Returns:
        Dict keyed by AdminUnit ID mapping to the AdminUnit record dict.
    """
    from compartment.cloud_helpers.graphql_queries import SEARCH_ADMIN_UNITS_QUERY

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

    # Key by ID for O(1) lookup
    refs = {item["id"]: item for item in items if item.get("id")}

    logger.info(
        f"Fetched {len(refs)} AdminUnit references for {len(admin_unit_ids)} requested IDs"
    )
    return refs
