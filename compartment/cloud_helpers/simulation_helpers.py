"""Helpers which reduce duplication between Batch programs."""
from os import environ

def get_simulation_params(simulation_job_id:str)->dict:

    simulation_params = {
        "SIMULATION_JOB_ID": simulation_job_id,
        "GRAPHQL_APIKEY": environ[("GRAPHQL_APIKEY")],
        "GRAPHQL_ENDPOINT": environ[("GRAPHQL_ENDPOINT")],
        "ENVIRONMENT": environ.get("ENVIRONMENT", None)
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
        intervention_lookup = intervention.get('Intervention')
        if not intervention_lookup:
            continue
            
        # Transform FieldConfigs to variance_params
        variance_params = []
        field_configs = intervention.get('FieldConfigs', {})
        field_config_items = field_configs.get('items', []) if field_configs else []
        
        for fc in field_config_items:
            if fc.get('has_variance'):
                variance_params.append({
                    'has_variance': fc.get('has_variance', False),
                    'distribution_type': fc.get('distribution_type', 'UNIFORM'),
                    'field_name': fc.get('field_key'),  # field_key → field_name
                    'min': fc.get('min'),
                    'max': fc.get('max'),
                })
        
        # Build the legacy intervention format
        legacy_intervention = {
            'id': intervention_lookup.get('name', '').lower(),  # SOCIAL_ISOLATION → social_isolation
            'type': 'INTERVENTION',
            'label': intervention_lookup.get('display_name', ''),
            'adherence_min': intervention.get('adherence_min'),
            'adherence_max': intervention.get('adherence_max'),
            'transmission_percentage': intervention.get('transmission_percentage'),
            'start_date': intervention.get('start_date'),
            'end_date': intervention.get('end_date'),
            'start_threshold': intervention.get('start_threshold'),
            'end_threshold': intervention.get('end_threshold'),
            'start_threshold_node_id': intervention.get('start_threshold_node_id'),
            'end_threshold_node_id': intervention.get('end_threshold_node_id'),
            'hour_reduction': intervention.get('hour_reduction'),
            'variance_params': variance_params,
        }
        
        legacy_interventions.append(legacy_intervention)
    
    return legacy_interventions