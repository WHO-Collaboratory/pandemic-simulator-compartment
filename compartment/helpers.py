from datetime import datetime, timedelta
from logging import basicConfig, StreamHandler, INFO
import geopy.distance
import json
import numpy as np
import pandas as pd
import uuid
import random
import scipy.stats as stats

# --------------------------------------------------
# Helper Functions: Config Loading
# --------------------------------------------------

def load_config(file_path):
    """
    Load the JSON configuration file.
    :param file_path: Path to the JSON config file.
    :return: Parsed configuration dictionary.
    """
    with open(file_path, 'r') as file:
        return json.load(file)

def setup_logging():
    """Sets up logging in an AWS Batch friendly format."""
    basicConfig(
        format="[%(levelname)s] %(name)s: %(message)s",
        level=INFO,
        handlers=[StreamHandler()]
    )
# --------------------------------------------------
# Helper Functions: Model Output
# --------------------------------------------------
# Compartment Groupings are used to condense the output of the model for compartment deltas
dengue_compartment_grouping = {
    "SV": ["SV"],
    "EV": ["EV1", "EV2", "EV3", "EV4"],
    "IV": ["IV1", "IV2", "IV3", "IV4"],
    "S": ["S0"],
    "E": ["E1", "E2", "E3", "E4"],
    "I": ["I1", "I2", "I3", "I4"],
    "C": ["C1", "C2", "C3", "C4"],
    "Snot": ["Snot1", "Snot2", "Snot3", "Snot4"],
    "E2": ["E12", "E13", "E14", "E21", "E23", "E24", "E31", "E32", "E34", "E41", "E42", "E43"],
    "I2": ["I12", "I13", "I14", "I21", "I23", "I24", "I31", "I32", "I34", "I41", "I42", "I43"],
    "H": ["H1", "H2", "H3", "H4"],
    "R": ["R1", "R2", "R3", "R4"]
}

covid_compartment_grouping = {
    "S": ["S"],
    "E": ["E"],
    "I": ["I"],
    "H": ["H"],
    "D": ["D"],
    "R": ["R"]
}

edge_to_variable = {
    "susceptible->infected": "beta",
    "susceptible->exposed": "beta",
    "infected->recovered": "gamma",
    "exposed->infected": "theta",
    "infected->hospitalized": "zeta",
    "infected->deceased": "delta",
    "hospitalized->recovered": "eta",
    "hospitalized->deceased": "epsilon"
}

def transform_interventions(data):
    """
    Take interventions dictionary, unpack, and transform for frontend
    """
    transformed = []

    for intervention_id, details in data.items():
        # Remove keys that should not be written to gql
        filtered = {
            k: v
            for k, v in details.items()
            if k not in ("start_date_ordinal", "end_date_ordinal")
        }
        transformed.append({"id": intervention_id, **filtered})

    return transformed

def compute_parent_admin_total(data, payload, unique_id, parent_unique_id, num_timesteps, step):
    #num_timesteps = len(data[0]["time_series"])
    start_date = datetime.strptime(data[0]["time_series"][0]["date"], "%Y-%m-%d")
    time_series = []
    parent_admin_info = next((payload["data"]["getSimulationJob"][key] for key in ["AdminUnit2", "AdminUnit1", "AdminUnit0"] if payload["data"]["getSimulationJob"].get(key)), None)
    results = {
        "id": parent_unique_id,
        "simulation_job_result_id": unique_id,
        "admin_zone_id": parent_admin_info["id"],
        "admin_unit_id": parent_admin_info["id"],
        "owner": payload['data']['getSimulationJob']['owner']
    }

    for t in range(num_timesteps):
        timestep_total = {"date": (start_date + timedelta(days=t*step)).strftime("%Y-%m-%d")}
        for zone in data:
            zone_timestep = zone["time_series"][t]
            for compartment, age_groups in zone_timestep.items():
                if compartment == "date":
                    continue  # ignore date
                if compartment not in timestep_total:
                    timestep_total[compartment] = {age_group: 0.0 for age_group in age_groups}
                for age_group, value in age_groups.items():
                    timestep_total[compartment][age_group] += value

        time_series.append(timestep_total)
    results["time_series"] = time_series

    return results

def compute_jax_compartment_deltas(population_matrix, disease_type, n_regions, compartment_list):
    compartment_deltas = {}

    if disease_type == "VECTOR_BORNE":
        keys = ["S", "E", "I", "C", "Snot", "E2", "I2", "H", "R"]
        compartment_deltas = {k: 0.0 for k in keys}
        for i in range(n_regions):
            df = pd.DataFrame(population_matrix[:,:,i], columns=compartment_list)
            compartment_deltas["S"] += float(df["S0"].iloc[-1])
            compartment_deltas["E"] += float(df["E_total"].iloc[-1])
            compartment_deltas["I"] += float(df["I_total"].iloc[-1])
            compartment_deltas["C"] += float(df["C_total"].iloc[-1])
            compartment_deltas["Snot"] += float(df["Snot_total"].iloc[-1])
            compartment_deltas["E2"] += float(df["E2_total"].iloc[-1])
            compartment_deltas["I2"] += float(df["I2_total"].iloc[-1])
            compartment_deltas["H"] += float(df["H_total"].iloc[-1])
            compartment_deltas["R"] += float(df["R_total"].iloc[-1])
    else:
        # Only report on compartments that actually exist
        # models may range anywhere between SIR and full SEIHDR
        if population_matrix.ndim == 4:
            # Sum across the age axis (axis=2) so we get back to (T, C, R)
            population_matrix = population_matrix.sum(axis=2)

        desired_keys = ["S", "E", "I", "H", "D", "R"]
        alt_cols = {"E": "E_total", "I": "I_total", "H": "H_total"}

        present = [k for k in desired_keys if (k in compartment_list) or (alt_cols.get(k) in compartment_list)]

        # snapshot = last day, shape (compartments, regions)
        final_step = population_matrix[-1]  # still (comp, regions)

        # Build quick lookup from comp name -> row index in final_step
        comp_idx = {c: idx for idx, c in enumerate(compartment_list)}

        compartment_deltas = {}
        for key in present:
            # Prefer cumulative *_total columns when they exist (E, I, H)
            if key in alt_cols and alt_cols[key] in comp_idx:
                col_name = alt_cols[key]
            else:
                col_name = key
            idx = comp_idx[col_name]
            compartment_deltas[key] = float(final_step[idx, :].sum())
    return compartment_deltas

def compute_multi_run_compartment_deltas(population_matrix, disease_type, n_regions, compartment_list):
    """
    Calculate the average compartment deltas over all simulations
    """
    all_deltas = []
    for sim in population_matrix:
        deltas = compute_jax_compartment_deltas(
            sim,
            disease_type,
            n_regions,
            compartment_list
        )
        all_deltas.append(deltas)
    # Get unique compartments
    compartments = all_deltas[0].keys()
    # calculate average for each compartment
    avg_deltas = {
        comp: np.mean([d[comp] for d in all_deltas])
        for comp in compartments
    }
    return avg_deltas

def create_jax_intervention_results(population_matrix: np.ndarray, intervention_dict: dict, compartment_list: list, 
                                    start_date: datetime, disease_type: str, n_timesteps: int, step: int):
    """
    Generate a list of {{id, trigger_date, trigger_type, active}} events
    Since we cant log statuses during intervention, we need to recompute them
    post-simulation
    """

    events = []
    if disease_type == "VECTOR_BORNE":
        infective_comps = [
            "I1", "I2", "I3", "I4",
            "I12", "I13", "I14",
            "I21", "I23", "I24",
            "I31", "I32", "I34",
            "I41", "I42", "I43",
        ]
        compartment_list = compartment_list[9:-8] # remove vectors and cumulative compartments
        infective_idx = [compartment_list.index(c) for c in infective_comps if c in compartment_list]
    else:
        # collapse 4d age strat matrix
        if population_matrix.ndim == 4:
            population_matrix = population_matrix.sum(axis=2)
        compartment_list = [s for s in compartment_list if "_total" not in s]
        infective_idx = [compartment_list.index("I")]
        
    # Track ON/OFF status for each intervention across timesteps
    status = {name: False for name in intervention_dict.keys()}

    for day in range(n_timesteps):
        current_date = start_date + timedelta(days=day)
        current_ordinal = current_date.toordinal()
        idx = day // step  # index into population_matrix

        if disease_type == "VECTOR_BORNE":
            humans_only = population_matrix[idx][9:-8, :] # remove vectors and cumulative compartments
        else:
            humans_only = population_matrix[idx] # already humans only

        infective_sum = humans_only[infective_idx].sum()
        total_pop = humans_only.sum()
        prop_inf = infective_sum / total_pop if total_pop > 0 else 0.0

        for name, cfg in intervention_dict.items():
            was_active = status[name]

            # date-based rules
            start_ord = cfg.get("start_date_ordinal")
            end_ord   = cfg.get("end_date_ordinal")

            if start_ord is not None and (current_ordinal == start_ord) and not was_active:
                status[name] = True
                events.append({
                    "id": name,
                    "trigger_date": current_date.strftime("%Y-%m-%d"),
                    "trigger_type": "DATE",
                    "active": True,
                })

            if end_ord is not None and (current_ordinal == end_ord) and was_active:
                status[name] = False
                events.append({
                    "id": name,
                    "trigger_date": current_date.strftime("%Y-%m-%d"),
                    "trigger_type": "DATE",
                    "active": False,
                })

            # threshold-based rules
            start_th = cfg.get("start_threshold")
            end_th   = cfg.get("end_threshold")

            if (start_th is not None) and (prop_inf >= start_th) and not status[name]:
                status[name] = True
                events.append({
                    "id": name,
                    "trigger_date": current_date.strftime("%Y-%m-%d"),
                    "trigger_type": "THRESHOLD",
                    "active": True,
                })

            if (end_th is not None) and (prop_inf <= end_th) and status[name]:
                status[name] = False
                events.append({
                    "id": name,
                    "trigger_date": current_date.strftime("%Y-%m-%d"),
                    "trigger_type": "THRESHOLD",
                    "active": False,
                })

    return events

def format_jax_output(intervention_dict, simulation_metadata, population_matrix, compartment_list, 
                      n_regions, start_date, n_timesteps, demographics, disease_type, step):
    """ Im hoping this replaces the mess we have above """
    unique_id = str(uuid.uuid4()) # Generate unique id for gql
    parent_unique_id = str(uuid.uuid4()) 
    
    # get results before transforming interventions for post-simulation
    intervention_results = create_jax_intervention_results(population_matrix, intervention_dict, compartment_list, start_date, disease_type, n_timesteps, step)
    intervention_dict = transform_interventions(intervention_dict)
    
    formatted_data = {
        "id": unique_id,
        "parent_time_series_id": parent_unique_id,
        "simulation_job_id": simulation_metadata['id'],
        "simulation_type": simulation_metadata['simulation_type'],
        "owner": simulation_metadata['owner'],
        "start_date": simulation_metadata['start_date'],
        "end_date": simulation_metadata['end_date'],
        "time_steps": simulation_metadata['time_steps'],
        "interventions": intervention_dict,
        "intervention_results": intervention_results, 
        "admin_zones": []
    }
    admin_zones_payload = payload['data']['getSimulationJob']['case_file']['admin_zones']

    dates = [
        (start_date + timedelta(days=i*step)).strftime("%Y-%m-%d")
        for i in range(population_matrix.shape[0])
    ]

    if disease_type == "VECTOR_BORNE":
        # create dictoinary mapping of compartments to generalized compartments for df groupby
        col2grp = {c:grp for grp, cols in dengue_compartment_grouping.items() for c in cols}

        zero_ages = dict.fromkeys(list(demographics.keys()), 0)
        
        for i in range(n_regions):
            # build a DataFrame for each region
            df = pd.DataFrame(population_matrix[:,:,i], index=dates, columns=compartment_list)
            df.index.name = "date"
            # group by dengue compartment mapping
            # transpose avoids FutureWarning: DataFrame.groupby with axis=1 is deprecated. Do `frame.T.groupby(...)` without axis instead.
            df_grp = df.T.groupby(col2grp).sum().T

            # nest each compartment with age groups
            df_nested = df_grp.map(lambda v: {**zero_ages, "age_all": float(v)})
            df_nested = df_nested.reset_index()

            formatted_data["admin_zones"].append({
                    "simulation_job_result_id": unique_id,
                    "owner": payload['data']['getSimulationJob']['owner'],
                    "admin_zone_id": admin_zones_payload[i].get('id', None),
                    "admin_unit_id": admin_zones_payload[i].get('id', None),
                    "time_series": df_nested.to_dict("records")
                })
    else:
        age_labels = list(demographics.keys())
        regions = [admin_zones_payload[i].get('id', None) for i in range(n_regions)]

        # remove totals from compartment_list
        master_list = ["S", "E", "I", "H", "D", "R"]
        #compartments = [x for x in compartment_list if x in master_list]

        index = pd.MultiIndex.from_product(
            [dates, compartment_list, age_labels, regions],
            names=["date", "compartment", "age_group", "region"]
        )

        df = pd.DataFrame({"value": population_matrix.ravel()},index=index).reset_index()
        df = df[df["compartment"].isin(master_list)]
        
        for region, df_r in df.groupby("region"):
            time_series = []
            for date, df_rd in df_r.groupby("date"):
                # pivot to get compartments × ages in a compact DataFrame
                piv = df_rd.pivot(index="compartment",columns="age_group",values="value")

                # add the column that totals across ages
                piv["age_all"] = piv.sum(axis=1)

                # build the nested structure: {compartment: {age: value, ...}}
                rec = {"date": date}
                rec.update(piv.to_dict(orient="index"))
                time_series.append(rec)

            # --- append to admin_zones so overall structure matches dengue path
            formatted_data["admin_zones"].append({
                "simulation_job_result_id": unique_id,
                "owner": payload['data']['getSimulationJob']['owner'],
                "admin_zone_id": region,
                "admin_unit_id": region,
                "time_series": time_series,
            })

    formatted_data["compartment_deltas"] = compute_jax_compartment_deltas(population_matrix, disease_type,  n_regions, compartment_list)
    formatted_data["parent_admin_total"] = compute_parent_admin_total(formatted_data['admin_zones'], payload, unique_id, parent_unique_id, population_matrix.shape[0], step)
    return formatted_data

def format_uncertainty_output(means_child, lower_child, upper_child,
                              means_parent, lower_parent, upper_parent,
                              simulation_metadata,
                              compartment_list,
                              admin_units,
                              start_date,
                              n_timesteps,
                              step,
                              avg_compartment_deltas):
    
    unique_id = str(uuid.uuid4()) # Generate unique id for gql
    parent_unique_id = str(uuid.uuid4()) 
    
    formatted_data = {
        "id": unique_id,
        "parent_time_series_id": parent_unique_id,
        "simulation_job_id": simulation_metadata['id'],
        "simulation_type": simulation_metadata['simulation_type'],
        "owner": simulation_metadata['owner'],
        "start_date": simulation_metadata['start_date'],
        "end_date": simulation_metadata['end_date'],
        "time_steps": simulation_metadata['time_steps'],
        "admin_zones": [],
        "compartment_deltas": avg_compartment_deltas,
        "parent_admin_total": []
    }
    
    base_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    admin_zones_payload = payload['data']['getSimulationJob']['case_file']['admin_zones']

    # number of timesteps in the output
    n_outputs_child = means_child.shape[0]
    n_outputs_parent = means_parent.shape[0]

    # Format child admin zones
    for zone_idx, zone in enumerate(admin_units):
        zone_obj = {
            "simulation_job_result_id": unique_id,
            "owner": payload['data']['getSimulationJob']['owner'],
            "admin_zone_id": admin_zones_payload[zone_idx].get('id', None),
            "admin_unit_id": admin_zones_payload[zone_idx].get('id', None),
            "time_series": []
        }

        for t in range(n_outputs_child):
            date = base_date + timedelta(days=step * t)
            record = {"date": date.isoformat()}

            # embed each compartment name as its own key
            for c_idx, comp_name in enumerate(compartment_list):
                record[comp_name] = {
                    "mean":  float(means_child[t, c_idx, zone_idx]),
                    "lower": float(lower_child[t, c_idx, zone_idx]),
                    "upper": float(upper_child[t, c_idx, zone_idx]),
                }

            zone_obj["time_series"].append(record)

        formatted_data["admin_zones"].append(zone_obj)

    # Parent admin total (total population) output
    parent_time_series = []
    for t in range(n_outputs_parent):
        date = base_date + timedelta(days=step * t)
        record = {"date": date.isoformat()}
        for c_idx, comp_name in enumerate(compartment_list):
            record[comp_name] = {
                "mean":  float(means_parent[t, c_idx]),
                "lower": float(lower_parent[t, c_idx]),
                "upper": float(upper_parent[t, c_idx]),
            }
        parent_time_series.append(record)

    parent_admin_info = next((payload["data"]["getSimulationJob"][key] for key in ["AdminUnit2", "AdminUnit1", "AdminUnit0"] if payload["data"]["getSimulationJob"].get(key)), None)
    formatted_data["parent_admin_total"] = {
        "id": parent_unique_id,
        "simulation_job_result_id": unique_id,
        "admin_zone_id": parent_admin_info["id"],
        "admin_unit_id": parent_admin_info["id"],
        "owner": payload['data']['getSimulationJob']['owner'],
        "time_series": parent_time_series
        }

    return formatted_data

# --------------------------------------------------
# Helper Functions: Payload formatting
# --------------------------------------------------
def create_initial_population_matrix(case_file, compartment_list):
    """ Using case file, create initial pop matrix for model """
    column_mapping = {value: index for index, value in enumerate(compartment_list)}
    initial_population = np.zeros((len(case_file), len(compartment_list)))

    for i, case in enumerate(case_file):
        infected = round(case['infected_population'] / 100 * case['population'],2)
        susceptible = case['population'] - infected
        initial_population[i, column_mapping['S']] = susceptible
        initial_population[i, column_mapping['I']] = infected

    return initial_population

def create_compartment_list(disease_nodes):
    """ Map disease nodes to compartment abbreviations """
    node_to_compartment = {
        "susceptible": "S",
        "exposed": "E",
        "infected": "I",
        "hospitalized": "H",
        "deceased": "D",
        "recovered": "R"
    }
    master_order = ["S", "E", "I", "H", "D", "R"]
    compartments = [node_to_compartment[node['id']] for node in disease_nodes if node['id'] in node_to_compartment]

    return sorted(compartments, key=lambda x: master_order.index(x))


def create_transmission_dict(transmission_edges):
    """ Map transmission edges to transmission rate variables """
    transmission_dict = {}
    for edge in transmission_edges:
        variable = edge_to_variable.get(edge['id'])
        if variable:
            transmission_rate = edge['data'].get('transmission_rate', 0)  # Default to 0 if not present
            transmission_dict[variable] = transmission_rate
    
    return transmission_dict

def build_uncertainty_params(transmission_edges: list, interventions: list):
    uncertainty_params = []
    
    # Process transmission edges
    for edge in transmission_edges:
        edge_id = edge.get("id")
        data = edge.get("data", {})
        variance_params = data.get("variance_params")

        if variance_params and variance_params.get("has_variance"):
            param_name = edge_to_variable.get(edge_id)
            if param_name:
                uncertainty_params.append({
                    "param": param_name,
                    "dist": variance_params.get("distribution_type", "uniform"),
                    "min": variance_params.get("min", 0),
                    "max": variance_params.get("max", 0)
                })

    # Process interventions
    for intervention in interventions:
        for var_param in intervention.get("variance_params", []):
            if var_param.get("has_variance"):
                field_name = var_param.get("field_name")
                if field_name:
                    uncertainty_params.append({
                        "param": f"intervention.{intervention['id']}.{field_name}",
                        "dist": var_param.get("distribution_type", "uniform"),
                        "min": var_param.get("min", 0) / 100,
                        "max": var_param.get("max", 0) / 100
                    })

    return uncertainty_params

def extract_admin_units(case_file):
    return [case['name'] for case in case_file]

def create_intervention_dict(intervention_nodes, start_date):
    intervention_dict = {}
    for intervention in intervention_nodes:
        # convert to ordinal to support jax timestep interventions
        if intervention.get("start_date") is not None and intervention.get("start_date") != "":
            start_date_ordinal = datetime.strptime(intervention.get("start_date"), "%Y-%m-%d").date().toordinal()
        else:
            start_date_ordinal = None
            intervention['start_date'] = None
        if intervention.get("end_date") is not None and intervention.get("end_date") != "":
            end_date_ordinal = datetime.strptime(intervention.get("end_date"), "%Y-%m-%d").date().toordinal()
        else:
            end_date_ordinal = None
            intervention['end_date'] = None

        if intervention.get("start_date") is None and intervention.get("end_date") is None and intervention.get("start_threshold") is None and intervention.get("end_threshold") is None:
            intervention['start_date'] = start_date
            start_date_ordinal = datetime.strptime(start_date, "%Y-%m-%d").date().toordinal()

        temp = {
            intervention.get("id"): {
                "start_threshold": intervention.get('start_threshold') / 100 if intervention.get('start_threshold') is not None else None,
                "end_threshold": intervention.get('end_threshold') / 100 if intervention.get('end_threshold') is not None else None,
                "start_date": intervention.get('start_date', None),
                "start_date_ordinal": start_date_ordinal,
                "end_date": intervention.get('end_date', None),
                "end_date_ordinal": end_date_ordinal,
                "adherence_min": intervention.get('adherence_min') / 100 if intervention.get('adherence_min') is not None else None,
                "transmission_percentage": intervention.get('transmission_percentage') / 100 if intervention.get('transmission_percentage') is not None else 0.05,
                "start_threshold_node_id": intervention.get('start_threshold_node_id', None),
                "end_threshold_node_id": intervention.get('end_threshold_node_id', None)
            }
        }
        intervention_dict.update(temp)

    return intervention_dict

def get_hemisphere(payload):
    """Crawls to request area's lattitude and returns hemisphere."""
    parent_admin_info = next(
        payload["data"]["getSimulationJob"][key]
        for key in ["AdminUnit2", "AdminUnit1", "AdminUnit0"]
        if payload["data"]["getSimulationJob"][key] is not None)
    center_lat = parent_admin_info['center_lat']
    if center_lat >= 0:
        return "North"
    else:
        return "South"

def get_temperature(case_file, default_min=0, default_max=38, default_mean=30):
    # Apply first admin zone temperature to all admin zones
    case_file = case_file[0]

    return {
        "temp_min": case_file.get("temp_min", default_min) if case_file.get("temp_min") is not None else default_min,
        "temp_max": case_file.get("temp_max", default_max) if case_file.get("temp_max") is not None else default_max,
        "temp_mean": case_file.get("temp_mean", default_mean) if case_file.get("temp_mean") is not None else default_mean,
    }

def get_travel_volume(travel_volume, case_file, default_leaving=0.2, default_returning=0.1):
    if len(case_file) == 1:
        # if 1 sub-admin zone, default no travel
        leaving = float(0)
        returning = float(0)
    else:
        if travel_volume:
            # Assigns defaults and checks for None values
            leaving =  travel_volume.get("leaving", default_leaving) / 100 if travel_volume.get("leaving") is not None else default_leaving
            returning = travel_volume.get("returning", default_returning) / 100 if travel_volume.get("returning") is not None else default_returning
        else:
            # Handle travel_volume: None case
            leaving = default_leaving,
            returning = default_returning
    return {
        "leaving": leaving,
        "returning": returning
    }

def get_simulation_step_size(n_timesteps):
    import math
    return max(math.ceil(n_timesteps / 365), 1)

def prepare_covid_initial_state(initial_population, age_transmission, demographics=None):
    comp_by_zone = initial_population.T  # shape (4, n_admin_zones)

    if demographics:
        weights = np.array(list(demographics.values()), dtype=float) / 100.0
    else:
        weights = np.array([1.0]) 
        age_transmission = np.array([1.0])

    # Broadcast multiply → (4, 1, n_admin) * (1, n_age, 1) → (4, n_age, n_admin)
    age_strat = comp_by_zone[:, None, :] * weights[None, :, None]

    return age_strat, age_transmission

def clean_payload(payload):
    # Extract components from payload
    admin_zones = payload['data']['getSimulationJob']['case_file']['admin_zones']
    demographics = payload['data']['getSimulationJob']['case_file']['demographics']
    time_steps = payload['data']['getSimulationJob']['time_steps']
    start_date = datetime.strptime(payload['data']['getSimulationJob']['start_date'], "%Y-%m-%d").date()
    disease_nodes = payload['data']['getSimulationJob']['Disease']['disease_nodes']
    transmission_edges = payload['data']['getSimulationJob']['Disease']['transmission_edges']
    interventions = payload['data']['getSimulationJob']['interventions']
    disease_type = payload['data']['getSimulationJob']['Disease']['disease_type'] #TODO remove from payload since intrinsic to model.
    # Handle travel_volume: None case
    travel_volume = (payload.get('data', {}).get('getSimulationJob', {}).get('travel_volume', None))
    travel_rates = get_travel_volume(travel_volume, admin_zones)
    
    # Create each component 
    if disease_type == "VECTOR_BORNE":
        run_mode = payload['data']['getSimulationJob']['run_mode']
        compartment_list = ['SV', 'EV1', 'EV2', 'EV3', 'EV4', 'IV1', 'IV2', 'IV3', 'IV4', 
                            'S0', 'E1', 'E2', 'E3', 'E4', 'I1', 'I2', 'I3', 'I4', 
                            'C1', 'C2', 'C3', 'C4', 'Snot1', 'Snot2', 'Snot3', 'Snot4', 
                            'E12', 'E13', 'E14', 'E21', 'E23', 'E24', 'E31', 'E32', 'E34', 'E41', 'E42', 'E43', 
                            'I12', 'I13', 'I14', 'I21', 'I23', 'I24', 'I31', 'I32', 'I34', 'I41', 'I42', 'I43', 
                            'H1', 'H2', 'H3', 'H4', 'R1', 'R2', 'R3', 'R4']
        initial_population = get_dengue_initial_population(admin_zones, compartment_list, run_mode)

    else:
        compartment_list = create_compartment_list(disease_nodes)
        initial_population = create_initial_population_matrix(admin_zones, compartment_list)

    transmission_dict = create_transmission_dict(transmission_edges)
    admin_units = extract_admin_units(admin_zones)
    intervention_dict = create_intervention_dict(interventions, payload['data']['getSimulationJob']['start_date'])

    simulation_metadata = {
        "id": payload['data']['getSimulationJob']['id'],
        "simulation_type": payload['data']['getSimulationJob']['simulation_type'],
        "owner": payload['data']['getSimulationJob']['owner'],
        "start_date": payload['data']['getSimulationJob']['start_date'],
        "end_date": payload['data']['getSimulationJob']['end_date'],
        "time_steps": payload['data']['getSimulationJob']['time_steps'],
    }
    
    # Create final cleaned payload
    cleaned_payload = {
        "initial_population": initial_population,
        "compartment_list": compartment_list,
        "transmission_dict": transmission_dict,
        "admin_units": admin_units,
        "demographics": demographics,
        "time_steps": time_steps,
        "start_date": start_date,
        "intervention_dict": intervention_dict,
        "travel_matrix": get_gravity_model_travel_matrix(admin_zones, travel_rates),
        "hemisphere": get_hemisphere(payload),
        "temperature": get_temperature(admin_zones),
        "travel_volume": travel_rates,
        "simulation_metadata": simulation_metadata
    }
    
    return cleaned_payload

# --------------------------------------------------
# Helper Functions: Gravity Model
# --------------------------------------------------

def get_admin_zone_df(case_file):
    ll_pop = pd.DataFrame(case_file, columns=["id", "center_lat", "center_lon", "population"])
    ll_pop['lat_long'] = list(zip(ll_pop.center_lat, ll_pop.center_lon))
    ll_pop.drop(columns=['center_lat', 'center_lon'], inplace=True)

    #Create df with Cartesian product for calculations between locations
    cross_df = ll_pop.merge(ll_pop, how='cross', suffixes=['_origin', '_destination'])  
    return cross_df


def gravity_model(df, mass_origin_col, mass_dest_col, distance_col, k=1):
    """
    Calculates the gravity model for a given dataframe.

    Args:
        df: pandas dataframe with the required columns
        origin_col: name of the column containing origin identifiers
        destination_col: name of the column containing destination identifiers
        mass_origin_col: name of the column containing origin mass (e.g., population, GDP)
        mass_dest_col: name of the column containing destination mass
        distance_col: name of the column containing distance between origin and destination
        k: constant of proportionality (optional, defaults to 1)

    Returns:
        pandas dataframe with an additional column containing the gravity model results
    """

    df['gravity'] = k * df[mass_origin_col] * df[mass_dest_col] / df[distance_col]**2
    return df

def create_travel_matrix(input_df, sigma):
    '''Data transformations including measuring distance, applying gravity model and pivoting the df'''
    #Calculating distances between cities
    input_df['distance_km'] = input_df.apply(lambda x: geopy.distance.geodesic(x.lat_long_origin, x.lat_long_destination).km, axis=1)
    input_df = gravity_model(input_df, 'population_origin', 'population_destination', 'distance_km')
    input_df['gravity'] = input_df['gravity'].replace(np.inf, 0)
    # Calculate a rate, e.g., flow per origin
    input_df["gravity_rate"] = input_df["gravity"] / input_df.groupby("id_origin")["gravity"].transform("sum")
    pivot_df = pd.pivot_table(input_df, 
                            index='id_origin', 
                            columns='id_destination', 
                            values='gravity_rate', 
                            aggfunc='sum')
    travel_matrix = pivot_df.T
    travel_matrix = travel_matrix.fillna(0) + np.diag(1-travel_matrix.sum(axis=1))

    travel_matrix = travel_matrix * sigma
    np.fill_diagonal(travel_matrix.values, 1 - sigma)
    return travel_matrix.to_numpy()

def get_gravity_model_travel_matrix(case_file, travel_rates):
    df = get_admin_zone_df(case_file)
    sigma = travel_rates.get("leaving")
    return create_travel_matrix(df, sigma)

# --------------------------------------------------
# Helper Functions: Latin Hypercube Sampling
# --------------------------------------------------
def LHS_uniform(low, high, num_runs):
    ''' Randomly samples values for each parameter n_runs times and scales values
        num_runs: int, number of times to run the simulation
        low: float, minimum value for scaling
        high: float, maximum value for scaling
        returns: 1D array of scaled uniform values
    '''
    vals = stats.qmc.LatinHypercube(1).random(num_runs)
    scaled_vals = stats.qmc.scale(vals, low, high).reshape(num_runs)
    return scaled_vals

def LHS_normal(mean, std, uniform_samples):
    '''Scales randomly sampled uniform values to a normal distribution
        mean: float, mean value of distribution
        std: float, standard deviation of normal distribution
        num_runs: int, number of times to run the simulation
        returns: 1D array of scaled uniform values in shape of normal distribution
    '''
    return stats.norm.ppf(uniform_samples, loc=mean, scale=std)


def LHS_triangular(min, probability_mode, max, uniform_samples):
    '''Scales randomly sampled uniform values to a triangular distribution
        min: float, minimum value of distribution
        probability_mode: float, shape parameter for the triangular distribution. Represents the mode of the distribution in its standardized form, 
            and must be between 0 and 1 (inclusive). Defines the peak of the triangular distribution relative to its base.
        max: float, maximum value of distribution
        num_runs: int, number of times to run the simulation
        returns: 1D array of scaled uniform values in shape of triangular distribution
    '''
    return stats.triang.ppf(uniform_samples, loc = min, c=probability_mode, scale = max-min)

def LHS_beta(alpha, beta, uniform_samples):
    '''Scales randomly sampled uniform values to a beta distribution
        alpha: float, exponent variable, power function of the variable x
        beta: float, complement of the variable x (1-x)
        returns: 1D array of scaled uniform values in shape of beta distribution
    '''
    return stats.beta.ppf(uniform_samples, alpha, beta)

#NOTE: loc parameter in scipy.stats.lognorm.ppf defaults to 0, which corresponds to the standard log-normal distribution. 
# If your distribution is shifted, you may need to adjust this parameter. 
def LHS_lognormal(mean, sigma, uniform_samples):
    '''Scales randomly sampled uniform values to a lognormal distribution
        sigma: float, shape parameter of the lognormal distribution
        mean: float, used to scale the distribution
        returns: 1D array of scaled uniform values in shape of lognormal distribution
    '''
    return stats.lognorm.ppf(uniform_samples, sigma, scale=np.exp(mean))

def generate_LHS_samples(num_runs, param_configs):
    """
    Generate samples based on specified distributions and parameters
   
    Args:
        num_runs (int): Number of samples to generate
        param_configs (list): List of dicts containing:
            - param (str): parameter name
            - dist (str): distribution type ('uniform', 'normal', 'triangular', 'beta', 'lognormal')
            - additional keys per distribution:
                - uniform: 'min', 'max'
                - normal: 'mean', 'std'
                - triangular: 'min', 'probability_mode', 'max'
                - beta: 'alpha', 'beta'
                - lognormal: 'mean', 'sigma'
    Returns:
        dict: mapping parameter names to lists of samples
    """
    results = {}
    for cfg in param_configs:
        name = cfg['param']
        dist = cfg['dist'].lower()

        if dist == 'uniform':
            low, high = cfg['min'], cfg['max']
            samples = LHS_uniform(low, high, num_runs)
        else:
            base = LHS_uniform(0, 1, num_runs)
            if dist == 'normal':
                samples = LHS_normal(cfg['mean'], cfg['std'], base)
            elif dist == 'triangular':
                samples = LHS_triangular(cfg['min'], cfg['probability_mode'], cfg['max'], base)
            elif dist == 'beta':
                samples = LHS_beta(cfg['alpha'], cfg['beta'], base)
            elif dist == 'lognormal':
                samples = LHS_lognormal(cfg['mean'], cfg['sigma'], base)
            else:
                raise ValueError(f"Unsupported distribution type: {cfg['dist']}")
        results[name] = samples.tolist()

    param_list = []
    for i in range(num_runs):
        entry = {p: results[p][i] for p in results}
        param_list.append(entry)
    return param_list

# --------------------------------------------------
# Helper Functions: Dengue Misc
# --------------------------------------------------

def get_dengue_initial_population(case_file, compartment_list, run_mode, vector_population=0):
    """
    Creates inital population matrix off of REQUIRED frontend entries:
      - MOSQUITO: [SV]
      - HUMAN: [S0, Snot1, Snot2, Snot3, Snot4]
    run_mode: STOCHASTIC or DETERMINISTIC
    """
    column_mapping = {value: index for index, value in enumerate(compartment_list)}
    initial_population = np.zeros((len(case_file), len(compartment_list)))

    for i, case in enumerate(case_file):
        # Assign all vectors to SV
        initial_population[i, column_mapping['SV']] = vector_population

        # Human population
        population = case['population']
        seroprevalence = case['seroprevalence'] if case['seroprevalence'] is not None else 0
        infected_population = case['infected_population'] if case['infected_population'] is not None else 0

        if run_mode == "STOCHASTIC":
            # Generate 4 random numbers for 4 serotypes
            weights = [random.random()  for _ in range(4)]
            weight_sum = sum(weights)
            Snot1_pct, Snot2_pct, Snot3_pct, Snot4_pct = tuple(w / weight_sum * seroprevalence for w in weights)
            I1_pct, I2_pct, I3_pct, I4_pct = tuple(w / weight_sum * infected_population for w in weights)
        else:
            # DETERMINISTIC - equally distribute
            Snot1_pct, Snot2_pct, Snot3_pct, Snot4_pct = (seroprevalence / 4,) * 4
            I1_pct, I2_pct, I3_pct, I4_pct = (infected_population / 4,) * 4

        # Snot1-4
        Snot1 = round(Snot1_pct / 100 * population, 2)
        Snot2 = round(Snot2_pct / 100 * population, 2)
        Snot3 = round(Snot3_pct / 100 * population, 2)
        Snot4 = round(Snot4_pct / 100 * population, 2)
        susceptible = population - Snot1 - Snot2 - Snot3 - Snot4

        initial_population[i, column_mapping['Snot1']] = Snot1
        initial_population[i, column_mapping['Snot2']] = Snot2
        initial_population[i, column_mapping['Snot3']] = Snot3
        initial_population[i, column_mapping['Snot4']] = Snot4

        # I1-4
        I1 = round(I1_pct / 100 * population, 2)
        I2 = round(I2_pct / 100 * population, 2)
        I3 = round(I3_pct / 100 * population, 2)
        I4 = round(I4_pct / 100 * population, 2)
        susceptible = susceptible - I1 - I2 - I3 - I4

        initial_population[i, column_mapping['S0']] = susceptible
        initial_population[i, column_mapping['I1']] = I1
        initial_population[i, column_mapping['I2']] = I2
        initial_population[i, column_mapping['I3']] = I3
        initial_population[i, column_mapping['I4']] = I4
    
    return initial_population