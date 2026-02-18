"""Helpers which reduce duplication between Batch programs."""
from os import environ

def get_simulation_params() -> dict:
    simulation_params = {
        "SIMULATION_JOB_ID": environ[("SIMULATION_JOB_ID")],
        "GRAPHQL_APIKEY": environ[("GRAPHQL_APIKEY")],
        "GRAPHQL_ENDPOINT": environ[("GRAPHQL_ENDPOINT")],
        "ENVIRONMENT": environ.get("ENVIRONMENT", None),
    }
    return simulation_params


def transform_normalized_interventions(normalized_interventions: list) -> list:
    """
    Transforms SimulationJobIntervention records to the legacy intervention format.

    Mapping:
    - Intervention.name (uppercase) → id (lowercase)
    - Intervention.display_name → label
    - Adds type: "INTERVENTION"
    - FieldConfigs.items → variance_params (field_key → field_name)

    Args:
        normalized_interventions: List of SimulationJobIntervention records from the new join table

    Returns:
        List of interventions in the legacy format expected by the validation models
    """
    legacy_interventions = []

    for intervention in normalized_interventions:
        # Skip if no linked Intervention lookup record
        intervention_lookup = intervention.get("Intervention")
        if not intervention_lookup:
            continue

        # Transform FieldConfigs to variance_params
        variance_params = []
        field_configs = intervention.get("FieldConfigs", {})
        field_config_items = field_configs.get("items", []) if field_configs else []

        for fc in field_config_items:
            if fc.get("has_variance"):
                variance_params.append(
                    {
                        "has_variance": fc.get("has_variance", False),
                        "distribution_type": fc.get("distribution_type", "UNIFORM"),
                        "field_name": fc.get("field_key"),  # field_key → field_name
                        "min": fc.get("min"),
                        "max": fc.get("max"),
                    }
                )

        # Build the legacy intervention format
        legacy_intervention = {
            "id": intervention_lookup.get(
                "name", ""
            ).lower(),  # SOCIAL_ISOLATION → social_isolation
            "type": "INTERVENTION",
            "label": intervention_lookup.get("display_name", ""),
            "adherence_min": intervention.get("adherence_min"),
            "adherence_max": intervention.get("adherence_max"),
            "transmission_percentage": intervention.get("transmission_percentage"),
            "start_date": intervention.get("start_date"),
            "end_date": intervention.get("end_date"),
            "start_threshold": intervention.get("start_threshold"),
            "end_threshold": intervention.get("end_threshold"),
            "start_threshold_node_id": intervention.get("start_threshold_node_id"),
            "end_threshold_node_id": intervention.get("end_threshold_node_id"),
            "hour_reduction": intervention.get("hour_reduction"),
            "variance_params": variance_params,
        }

        legacy_interventions.append(legacy_intervention)

    return legacy_interventions


def transform_simulation_job_admin_units(
    sim_job_admin_units: list,
    admin_unit_refs: dict,
) -> list:
    """
    Transforms SimulationJobAdminUnit records to the legacy case_file.admin_zones format.

    Joins each SimulationJobAdminUnit with its corresponding AdminUnit reference
    record to fill in geographic fields (center_lat, center_lon, admin_level, etc.)
    that the compartment batch model requires.

    This will be optimized in the future to be a proper table relationship
    """
    legacy_admin_zones = []

    for unit in sim_job_admin_units:
        admin_unit_id = unit.get("admin_unit_id")
        ref = admin_unit_refs.get(admin_unit_id, {})

        legacy_admin_zones.append(
            {
                "id": admin_unit_id,
                "admin_code": ref.get("admin_code"),
                "admin_iso_code": ref.get("admin_iso_code"),
                "admin_level": ref.get("admin_level"),
                "center_lat": ref.get("center_lat"),
                "center_lon": ref.get("center_lon"),
                "viz_name": ref.get("viz_name") or unit.get("name"),
                "name": unit.get("name"),
                "population": unit.get("population", 0),
                "osm_id": ref.get("osm_id"),
                "infected_population": unit.get("infected_population", 0),
                "seroprevalence": unit.get("seroprevalence"),
                "temp_min": unit.get("temp_min", 15),
                "temp_max": unit.get("temp_max", 30),
                "temp_mean": unit.get("temp_mean", 25),
            }
        )

    return legacy_admin_zones