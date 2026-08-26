from datetime import datetime, timedelta, date
import logging
from logging import basicConfig, StreamHandler, INFO
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import geopy.distance
import json
import numpy as np
import pandas as pd
import uuid
import random
import scipy.stats as stats
import os
import sys

logger = logging.getLogger(__name__)

# --------------------------------------------------
# Helper Functions: Config Loading
# --------------------------------------------------


def _legacy_edge_metadata(config_data: dict) -> dict[tuple[str, str], dict]:
    """Return schema metadata keyed by legacy edge source/target.

    Older generated configs did not include ``variable_name`` or
    ``value_type`` on their embedded ``Disease.transmission_edges``.  Resolve
    the model lazily so custom edges can be upgraded without relying on the
    small ``edge_to_variable`` compatibility table.
    """
    disease = config_data.get("Disease") or {}
    artifact = config_data.get("ModelArtifact") or {}
    model_identifier = artifact.get("model_key") or disease.get("disease_type")
    if not model_identifier:
        return {}

    try:
        from compartment.registry import resolve

        model_class = resolve(model_identifier)
        if model_class is None:
            return {}
        schema = model_class._build_parameter_schema()
    except Exception:
        logger.debug(
            "Could not resolve parameter schema while upgrading legacy config",
            exc_info=True,
        )
        return {}

    return {
        (edge.source.casefold(), edge.target.casefold()): {
            "disease_param": edge.variable_name,
            "value_type": edge.to_dict()["value_type"],
        }
        for edge in schema.transmission_edges
    }


def _legacy_field_config(
    variance: dict | None,
    *,
    field_key: str,
    disease_param: str | None = None,
) -> dict:
    """Translate one legacy variance declaration to a FieldConfig item."""
    variance = variance or {}
    item = {
        "field_key": field_key,
        "has_variance": bool(variance.get("has_variance", False)),
        "distribution_type": str(
            variance.get("distribution_type", "UNIFORM")
        ).upper(),
    }
    if disease_param:
        item["disease_param"] = disease_param.upper()
    for bound in ("min", "max"):
        if variance.get(bound) is not None:
            item[bound] = variance[bound]
    return item


def _normalize_legacy_config(config_data: dict) -> None:
    """Upgrade legacy generated sections in a SimulationJob config in place."""
    disease = config_data.get("Disease")
    if isinstance(disease, dict) and "transmission_edges" in disease:
        legacy_edges = disease.pop("transmission_edges") or []
        if "TransmissionEdges" not in config_data:
            schema_edges = _legacy_edge_metadata(config_data)
            normalized_edges = []
            for edge in legacy_edges:
                source = edge.get("source", "")
                target = edge.get("target", "")
                data = edge.get("data") or {}
                variance = data.get("variance_params") or {}
                schema_edge = schema_edges.get(
                    (str(source).casefold(), str(target).casefold()),
                    {},
                )
                disease_param = (
                    variance.get("field_name")
                    or schema_edge.get("disease_param")
                    or edge_to_variable.get(f"{source}->{target}")
                )
                lookup = {"source": source, "target": target}
                value_type = schema_edge.get("value_type") or edge.get("value_type")
                if value_type:
                    lookup["value_type"] = value_type
                normalized_edges.append(
                    {
                        "transmission_edge": lookup,
                        "value": data.get(
                            "transmission_rate",
                            edge.get("transmission_rate", 0),
                        ),
                        "FieldConfigs": {
                            "items": [
                                _legacy_field_config(
                                    variance,
                                    field_key="value",
                                    disease_param=disease_param,
                                )
                            ]
                        },
                    }
                )
            config_data["TransmissionEdges"] = {"items": normalized_edges}

    if "interventions" in config_data:
        legacy_interventions = config_data.pop("interventions") or []
        if "Interventions" not in config_data:
            normalized_interventions = []
            for intervention in legacy_interventions:
                intervention = dict(intervention)
                intervention_id = intervention.pop("id", None) or intervention.pop(
                    "name", None
                )
                variance_params = intervention.pop("variance_params", []) or []
                item = {
                    "Intervention": {
                        "name": intervention_id,
                        "display_name": intervention.pop(
                            "display_name",
                            str(intervention_id).replace("_", " ").title(),
                        ),
                    },
                    **intervention,
                }
                if variance_params:
                    item["FieldConfigs"] = {
                        "items": [
                            _legacy_field_config(
                                variance,
                                field_key=variance.get("field_name", ""),
                            )
                            for variance in variance_params
                        ]
                    }
                normalized_interventions.append(item)
            config_data["Interventions"] = {"items": normalized_interventions}


def load_config_from_json(config_path: str) -> dict:
    """Load simulation config from a local JSON file.

    Handles convenience shortcuts so users skip cloud-only boilerplate:
    - Upgrades legacy Disease.transmission_edges and lowercase interventions
    - Wraps top-level admin_zones into case_file
    - Adds default demographics if missing
    """
    with open(config_path, "r") as f:
        config_data = json.load(f)

    # Upgrade both short-form files and already-wrapped GraphQL-shaped files.
    is_wrapped = (
        "data" in config_data and "getSimulationJob" in config_data["data"]
    )
    job_config = (
        config_data["data"]["getSimulationJob"] if is_wrapped else config_data
    )
    _normalize_legacy_config(job_config)

    if is_wrapped:
        return config_data

    # Wrap top-level admin_zones into case_file
    if "admin_zones" in job_config and "case_file" not in job_config:
        job_config["case_file"] = {"admin_zones": job_config.pop("admin_zones")}

    # Move top-level demographics into case_file
    if "case_file" in job_config:
        if "demographics" not in job_config["case_file"]:
            job_config["case_file"]["demographics"] = job_config.pop(
                "demographics",
                {},
            )

    return {"data": {"getSimulationJob": job_config}}


def write_results_to_local(results: list, output_path: str):
    """Write simulation results to a local JSON file."""
    # Create directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)


def setup_logging():
    """Sets up logging in an AWS Lambda/CloudWatch friendly format."""
    root_logger = logging.getLogger()

    # Remove all existing handlers to ensure clean configuration
    # This is important for AWS Lambda where handlers might already exist
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create a stream handler that writes to stdout (required for CloudWatch)
    handler = StreamHandler(sys.stdout)
    handler.setLevel(INFO)

    # Create formatter
    formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)

    # Configure root logger
    root_logger.setLevel(INFO)
    root_logger.addHandler(handler)


# --------------------------------------------------
# Helper Functions: Model Output
# --------------------------------------------------


def get_compartment_delta_grouping(model_class, compartment_list):
    """
    Get compartment grouping for delta calculations.

    If model has COMPARTMENT_DELTA_GROUPING, use it.
    Otherwise, generate default 1:1 mapping from compartment_list (excluding cumulative _total columns).
    """
    if model_class and hasattr(model_class, "COMPARTMENT_DELTA_GROUPING"):
        return model_class.COMPARTMENT_DELTA_GROUPING

    # Default: 1:1 mapping (each compartment groups to itself)
    # Exclude cumulative (_total) columns as they're only used internally
    return {comp: [comp] for comp in compartment_list if not comp.endswith("_total")}


edge_to_variable = {
    "susceptible->infected": "beta",
    "susceptible->exposed": "beta",
    "infected->recovered": "gamma",
    "exposed->infected": "theta",
    "infected->hospitalized": "zeta",
    "infected->deceased": "delta",
    "hospitalized->recovered": "eta",
    "hospitalized->deceased": "epsilon",
    "recovered->susceptible": "omega",
}


def convert_dates(obj):
    """Recursively convert date, datetime, and ndarray objects for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_dates(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_dates(v) for v in obj]
    elif isinstance(obj, (date, datetime)):
        return obj.isoformat()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


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


def compute_parent_admin_total(data, payload, unique_id, parent_unique_id):
    num_timesteps = len(data[0]["time_series"])
    time_series = []
    parent_admin_info = payload.get("AdminUnit")
    results = {
        "id": parent_unique_id,
        "simulation_job_result_id": unique_id,
        "admin_zone_id": parent_admin_info["id"],
        "admin_unit_id": parent_admin_info["id"],
        "owner": payload["owner"],
    }

    for t in range(num_timesteps):
        # Reuse the child zones' date labels instead of recomputing them from a
        # fixed step. The output grid appends the exact end date when the
        # sampling step does not divide the duration evenly, so ``t * step``
        # would overshoot the requested end date on downsampled runs.
        timestep_total = {"date": data[0]["time_series"][t]["date"]}
        for zone in data:
            zone_timestep = zone["time_series"][t]
            for compartment, age_groups in zone_timestep.items():
                if compartment == "date":
                    continue  # ignore date
                # Skip cumulative columns (defensive check)
                if compartment.endswith("_total"):
                    continue
                if compartment not in timestep_total:
                    timestep_total[compartment] = {
                        age_group: 0.0 for age_group in age_groups
                    }
                for age_group, value in age_groups.items():
                    timestep_total[compartment][age_group] += value

        time_series.append(timestep_total)
    results["time_series"] = time_series

    return results


def compute_jax_compartment_deltas(
    population_matrix, disease_type, n_regions, compartment_list, model_class=None
):
    """
    Compute compartment deltas using model's COMPARTMENT_DELTA_GROUPING if available,
    otherwise defaults to 1:1 mapping.

    Prefers cumulative (_total) columns when they exist, matching original behavior.
    """
    compartment_deltas = {}

    # Get compartment grouping from model or generate default
    grouping = get_compartment_delta_grouping(model_class, compartment_list)

    # Handle age-stratified models (4D arrays)
    if population_matrix.ndim == 4:
        # Sum across the age axis (axis=2) so we get back to (T, C, R)
        population_matrix = population_matrix.sum(axis=2)

    # snapshot = last day, shape (compartments, regions)
    final_step = population_matrix[-1]  # (comp, regions)

    # Build quick lookup from comp name -> row index in final_step
    comp_idx = {c: idx for idx, c in enumerate(compartment_list)}

    # Compute delta for each grouped compartment
    for group_name, comp_names in grouping.items():
        # Prefer cumulative column if it exists (e.g., E_total instead of summing E1, E2, E3, E4)
        cumulative_col = f"{group_name}_total"

        if cumulative_col in comp_idx:
            # Use the cumulative column directly
            idx = comp_idx[cumulative_col]
            compartment_deltas[group_name] = float(final_step[idx, :].sum())
        else:
            # Sum individual compartments in the group
            group_total = 0.0
            for comp_name in comp_names:
                if comp_name in comp_idx:
                    idx = comp_idx[comp_name]
                    group_total += float(final_step[idx, :].sum())
            compartment_deltas[group_name] = group_total

    return compartment_deltas


def compute_multi_run_compartment_deltas(
    population_matrix, disease_type, n_regions, compartment_list, model_class=None
):
    """
    Calculate the median compartment deltas over all simulations, plus
    lower/upper simulation-based interval bounds on the same per-run deltas.
    """
    all_deltas = []
    for sim in population_matrix:
        deltas = compute_jax_compartment_deltas(
            sim, disease_type, n_regions, compartment_list, model_class
        )
        all_deltas.append(deltas)
    # Get unique compartments
    compartments = all_deltas[0].keys()
    lower_q = 2.5
    upper_q = 97.5
    # Median + CI bounds for each compartment, matching the multi-run time-series
    # result shape.
    return {
        comp: {
            "median": float(np.median([d[comp] for d in all_deltas])),
            "lower": float(np.percentile([d[comp] for d in all_deltas], lower_q)),
            "upper": float(np.percentile([d[comp] for d in all_deltas], upper_q)),
        }
        for comp in compartments
    }



def create_jax_intervention_results(
    population_matrix: np.ndarray,
    intervention_dict: dict,
    compartment_list: list,
    start_date: datetime,
    disease_type: str,
    n_timesteps: int,
    model_class=None,
):
    """
    Generate a list of {{id, trigger_date, trigger_type, active}} events
    Since we cant log statuses during intervention, we need to recompute them
    post-simulation
    """

    events = []
    if disease_type == "VECTOR_BORNE":
        infective_comps = [
            "I1",
            "I2",
            "I3",
            "I4",
            "I12",
            "I13",
            "I14",
            "I21",
            "I23",
            "I24",
            "I31",
            "I32",
            "I34",
            "I41",
            "I42",
            "I43",
        ]
        compartment_list = compartment_list[
            9:-8
        ]  # remove vectors and cumulative compartments
        infective_idx = [
            compartment_list.index(c) for c in infective_comps if c in compartment_list
        ]
    elif disease_type == "VECTOR_BORNE_2STRAIN":
        infective_comps = ["I1", "I2", "I12", "I21"]
        infective_idx = [
            compartment_list.index(c) for c in infective_comps if c in compartment_list
        ]
    else:
        # collapse 4d age strat matrix
        if population_matrix.ndim == 4:
            population_matrix = population_matrix.sum(axis=2)
        compartment_list = [s for s in compartment_list if "_total" not in s]

        # Use schema-declared infective compartments when available;
        # fall back to "I" for legacy models.
        if model_class is None:
            # Backward compatibility for direct callers. The normal formatter
            # passes the exact model class so shared disease types stay valid.
            from compartment.registry import resolve

            model_class = resolve(disease_type)
        if model_class and hasattr(model_class, "COMPARTMENTS") and hasattr(model_class.COMPARTMENTS, "infective_ids"):
            infective_comps = list(model_class.COMPARTMENTS.infective_ids)
            infective_idx = [
                compartment_list.index(c) for c in infective_comps if c in compartment_list
            ]
        elif "I" in compartment_list:
            infective_idx = [compartment_list.index("I")]
        else:
            # No infective compartment found — use empty list
            infective_idx = []

    # Track ON/OFF status for each intervention across timesteps
    status = {name: False for name in intervention_dict.keys()}

    time_points = get_simulation_time_points(n_timesteps)
    if len(time_points) != population_matrix.shape[0]:
        raise ValueError(
            "Population matrix time axis does not match the simulation time grid: "
            f"{population_matrix.shape[0]} values for {len(time_points)} time points"
        )

    for day in range(n_timesteps + 1):
        current_date = start_date + timedelta(days=day)
        current_ordinal = current_date.toordinal()
        # Threshold reconstruction uses the most recent solver output on or
        # before this calendar day. This also maps the exact final day to the
        # appended endpoint when the regular sampling step does not divide the
        # duration evenly.
        idx = int(np.searchsorted(time_points, day, side="right") - 1)

        if disease_type == "VECTOR_BORNE":
            humans_only = population_matrix[idx][
                9:-8, :
            ]  # remove vectors and cumulative compartments
        elif disease_type == "VECTOR_BORNE_2STRAIN":
            humans_only = population_matrix[idx]  # no vectors in 2-strain model
        else:
            humans_only = population_matrix[idx]  # already humans only

        infective_sum = humans_only[infective_idx].sum()
        total_pop = humans_only.sum()
        prop_inf = infective_sum / total_pop if total_pop > 0 else 0.0

        for name, cfg in intervention_dict.items():
            was_active = status[name]

            # date-based rules
            start_ord = cfg.get("start_date_ordinal")
            end_ord = cfg.get("end_date_ordinal")

            if (
                start_ord is not None
                and (current_ordinal == start_ord)
                and not was_active
            ):
                status[name] = True
                events.append(
                    {
                        "id": name,
                        "trigger_date": current_date.strftime("%Y-%m-%d"),
                        "trigger_type": "DATE",
                        "active": True,
                    }
                )

            if end_ord is not None and (current_ordinal == end_ord) and was_active:
                status[name] = False
                events.append(
                    {
                        "id": name,
                        "trigger_date": current_date.strftime("%Y-%m-%d"),
                        "trigger_type": "DATE",
                        "active": False,
                    }
                )

            # threshold-based rules
            start_th = cfg.get("start_threshold")
            end_th = cfg.get("end_threshold")

            if (start_th is not None) and (prop_inf >= start_th) and not status[name]:
                status[name] = True
                events.append(
                    {
                        "id": name,
                        "trigger_date": current_date.strftime("%Y-%m-%d"),
                        "trigger_type": "THRESHOLD",
                        "active": True,
                    }
                )

            if (end_th is not None) and (prop_inf <= end_th) and status[name]:
                status[name] = False
                events.append(
                    {
                        "id": name,
                        "trigger_date": current_date.strftime("%Y-%m-%d"),
                        "trigger_type": "THRESHOLD",
                        "active": False,
                    }
                )

    return events


def create_date_based_intervention_results(
    intervention_dict: dict,
    start_date,
    n_timesteps: int,
):
    """Return one shared set of date-trigger events for a multi-run result.

    Date schedules do not vary between uncertainty or stochastic trajectories,
    so they can be reported once on the aggregate result. Threshold events are
    intentionally excluded because their trigger dates can differ per run.
    """
    if isinstance(start_date, datetime):
        simulation_start = start_date.date()
    elif isinstance(start_date, date):
        simulation_start = start_date
    else:
        simulation_start = datetime.strptime(str(start_date), "%Y-%m-%d").date()

    simulation_end_ordinal = simulation_start.toordinal() + n_timesteps
    events = []

    for name, cfg in intervention_dict.items():
        start_ordinal = cfg.get("start_date_ordinal")
        end_ordinal = cfg.get("end_date_ordinal")

        if (
            start_ordinal is not None
            and simulation_start.toordinal() <= start_ordinal <= simulation_end_ordinal
        ):
            events.append(
                {
                    "id": name,
                    "trigger_date": date.fromordinal(start_ordinal).isoformat(),
                    "trigger_type": "DATE",
                    "active": True,
                }
            )

            # Match deterministic event semantics: an end event is only
            # emitted after the intervention has activated in the simulation.
            if (
                end_ordinal is not None
                and start_ordinal < end_ordinal <= simulation_end_ordinal
            ):
                events.append(
                    {
                        "id": name,
                        "trigger_date": date.fromordinal(end_ordinal).isoformat(),
                        "trigger_type": "DATE",
                        "active": False,
                    }
                )

    return sorted(events, key=lambda event: event["trigger_date"])


def format_jax_output(
    intervention_dict,
    payload,
    population_matrix,
    compartment_list,
    n_regions,
    start_date,
    n_timesteps,
    demographics,
    disease_type,
    model_class=None,
):
    """Im hoping this replaces the mess we have above"""
    unique_id = str(uuid.uuid4())  # Generate unique id for gql
    parent_unique_id = str(uuid.uuid4())

    # get results before transforming interventions for post-simulation
    intervention_results = create_jax_intervention_results(
        population_matrix,
        intervention_dict,
        compartment_list,
        start_date,
        disease_type,
        n_timesteps,
        model_class=model_class,
    )
    intervention_dict = transform_interventions(intervention_dict)

    formatted_data = {
        "id": unique_id,
        "parent_time_series_id": parent_unique_id,
        "simulation_job_id": payload["id"],
        "simulation_type": payload["simulation_type"],
        "owner": payload["owner"],
        "start_date": payload["start_date"],
        "end_date": payload["end_date"],
        "time_steps": payload["time_steps"],
        "interventions": intervention_dict,
        "intervention_results": intervention_results,
        "admin_zones": [],
    }
    admin_zones_payload = payload["case_file"]["admin_zones"]

    time_points = get_simulation_time_points(n_timesteps)
    if len(time_points) != population_matrix.shape[0]:
        raise ValueError(
            "Population matrix time axis does not match the simulation time grid: "
            f"{population_matrix.shape[0]} values for {len(time_points)} time points"
        )
    dates = [
        (start_date + timedelta(days=float(offset))).strftime("%Y-%m-%d")
        for offset in time_points
    ]

    if population_matrix.ndim == 3:
        # Get compartment grouping from model or generate default
        grouping = get_compartment_delta_grouping(model_class, compartment_list)
        # Create dictionary mapping of compartments to generalized compartments for df groupby
        col2grp = {c: grp for grp, cols in grouping.items() for c in cols}

        # Ensure all compartments (including cumulative columns) are mapped
        for comp in compartment_list:
            if comp not in col2grp:
                col2grp[comp] = comp

        zero_ages = dict.fromkeys(list(demographics.keys()), 0)

        for i in range(n_regions):
            # build a DataFrame for each region
            df = pd.DataFrame(
                population_matrix[:, :, i], index=dates, columns=compartment_list
            )
            df.index.name = "date"
            # group by dengue compartment mapping
            # transpose avoids FutureWarning: DataFrame.groupby with axis=1 is deprecated. Do `frame.T.groupby(...)` without axis instead.
            df_grp = df.T.groupby(col2grp).sum().T

            # Remove cumulative columns (_total) from time_series output
            df_grp = df_grp[
                [col for col in df_grp.columns if not col.endswith("_total")]
            ]

            # nest each compartment with age groups
            df_nested = df_grp.map(lambda v: {**zero_ages, "age_all": float(v)})
            df_nested = df_nested.reset_index()

            formatted_data["admin_zones"].append(
                {
                    "simulation_job_result_id": unique_id,
                    "owner": payload["owner"],
                    "admin_zone_id": admin_zones_payload[i].get("id", None),
                    "admin_unit_id": admin_zones_payload[i].get("id", None),
                    "time_series": df_nested.to_dict("records"),
                }
            )
    elif population_matrix.ndim == 4:
        formatted_data["admin_zones"] = fast_format_jax_output_demographic(
            population_matrix,
            compartment_list,
            demographics,
            admin_zones_payload,
            n_regions,
            n_timesteps,
            unique_id,
            payload,
        )
    else:
        raise ValueError(
            f"Unsupported population matrix dimension: {population_matrix.ndim}"
        )

    formatted_data["compartment_deltas"] = compute_jax_compartment_deltas(
        population_matrix, disease_type, n_regions, compartment_list, model_class
    )
    formatted_data["parent_admin_total"] = compute_parent_admin_total(
        formatted_data["admin_zones"],
        payload,
        unique_id,
        parent_unique_id,
    )
    return formatted_data


def fast_format_jax_output_demographic(
    population_matrix,
    compartment_list,
    demographics,
    admin_zones_payload,
    n_regions,
    n_timesteps,
    unique_id,
    payload,
):
    # Build index arrays - only include base compartments (not cumulative _total columns)
    master_list = [c for c in compartment_list if not c.endswith("_total")]
    age_labels = list(demographics.keys())
    time_points = get_simulation_time_points(n_timesteps)
    if len(time_points) != population_matrix.shape[0]:
        raise ValueError(
            "Population matrix time axis does not match the simulation time grid: "
            f"{population_matrix.shape[0]} values for {len(time_points)} time points"
        )
    dates = [
        (
            payload["start_date"] + timedelta(days=float(offset))
        ).strftime("%Y-%m-%d")
        for offset in time_points
    ]
    regions = [admin_zones_payload[i].get("id", None) for i in range(n_regions)]
    index = pd.MultiIndex.from_product(
        [dates, compartment_list, age_labels, regions],
        names=["date", "compartment", "age_group", "region"],
    )

    # Flatten the population_matrix for DataFrame
    df = pd.DataFrame({"value": population_matrix.ravel()}, index=index).reset_index()
    # Filter to only include compartments in master_list (base + cumulative)
    df = df[df["compartment"].isin(master_list)]

    # Pivot ALL data at once: grouped by region, then by date
    df_piv = df.pivot_table(
        index=["region", "date", "compartment"],
        columns="age_group",
        values="value",
        fill_value=0,
    )

    # Compute 'age_all' for all (region,date,compartment) groups
    df_piv["age_all"] = df_piv.sum(axis=1)

    # Unstack to region, then date for efficient extraction
    all_regions = []
    for region, group in df_piv.groupby(level="region"):
        # group is indexed by (region, date, compartment)
        time_series = []
        for date, group_date in group.groupby(level="date"):
            # group_date is (region, date, compartment) x [ages]
            # We need compartment as keys, age dicts as values
            rec = {"date": date}
            for comp, row in group_date.droplevel(["region", "date"]).iterrows():
                rec[comp] = row.to_dict()
            time_series.append(rec)
        all_regions.append(
            {
                "simulation_job_result_id": unique_id,
                "owner": payload["owner"],
                "admin_zone_id": region,
                "admin_unit_id": region,
                "time_series": time_series,
            }
        )

    return all_regions


def format_uncertainty_output(
    medians_child,
    lower_child,
    upper_child,
    medians_parent,
    lower_parent,
    upper_parent,
    payload,
    compartment_list,
    admin_units,
    start_date,
    n_timesteps,
    compartment_deltas,
    intervention_dict=None,
):

    unique_id = str(uuid.uuid4())  # Generate unique id for gql
    parent_unique_id = str(uuid.uuid4())
    base_date = datetime.strptime(start_date, "%Y-%m-%d").date()

    formatted_data = {
        "id": unique_id,
        "parent_time_series_id": parent_unique_id,
        "simulation_job_id": payload["id"],
        "simulation_type": payload["simulation_type"],
        "owner": payload["owner"],
        "start_date": payload["start_date"],
        "end_date": payload["end_date"],
        "time_steps": payload["time_steps"],
        "interventions": transform_interventions(intervention_dict or {}),
        "intervention_results": create_date_based_intervention_results(
            intervention_dict or {}, base_date, n_timesteps
        ),
        "admin_zones": [],
        "compartment_deltas": compartment_deltas,
        "parent_admin_total": [],
    }

    admin_zones_payload = payload["case_file"]["admin_zones"]

    # Filter out cumulative (_total) columns from time_series output
    display_compartments = [
        (idx, comp)
        for idx, comp in enumerate(compartment_list)
        if not comp.endswith("_total")
    ]

    # number of timesteps in the output
    n_outputs_child = medians_child.shape[0]
    n_outputs_parent = medians_parent.shape[0]
    time_points = get_simulation_time_points(n_timesteps)
    if n_outputs_child != len(time_points) or n_outputs_parent != len(time_points):
        raise ValueError(
            "Multi-run output time axes do not match the simulation time grid: "
            f"child={n_outputs_child}, parent={n_outputs_parent}, "
            f"time_points={len(time_points)}"
        )

    # Format child admin zones
    for zone_idx, zone in enumerate(admin_units):
        zone_obj = {
            "simulation_job_result_id": unique_id,
            "owner": payload["owner"],
            "admin_zone_id": admin_zones_payload[zone_idx].get("id", None),
            "admin_unit_id": admin_zones_payload[zone_idx].get("id", None),
            "time_series": [],
        }

        for t, offset in enumerate(time_points):
            date = base_date + timedelta(days=float(offset))
            record = {"date": date.isoformat()}

            # embed each compartment name as its own key (exclude cumulative columns)
            for c_idx, comp_name in display_compartments:
                record[comp_name] = {
                    "median": float(medians_child[t, c_idx, zone_idx]),
                    "lower": float(lower_child[t, c_idx, zone_idx]),
                    "upper": float(upper_child[t, c_idx, zone_idx]),
                }

            zone_obj["time_series"].append(record)

        formatted_data["admin_zones"].append(zone_obj)

    # Parent admin total (total population) output
    parent_time_series = []
    for t, offset in enumerate(time_points):
        date = base_date + timedelta(days=float(offset))
        record = {"date": date.isoformat()}
        for c_idx, comp_name in display_compartments:
            record[comp_name] = {
                "median": float(medians_parent[t, c_idx]),
                "lower": float(lower_parent[t, c_idx]),
                "upper": float(upper_parent[t, c_idx]),
            }
        parent_time_series.append(record)

    parent_admin_info = payload.get("AdminUnit")
    formatted_data["parent_admin_total"] = {
        "id": parent_unique_id,
        "simulation_job_result_id": unique_id,
        "admin_zone_id": parent_admin_info["id"],
        "admin_unit_id": parent_admin_info["id"],
        "owner": payload["owner"],
        "time_series": parent_time_series,
    }

    return formatted_data


# --------------------------------------------------
# Helper Functions: Payload formatting
# --------------------------------------------------
def create_initial_population_matrix(case_file, compartment_list):
    """Using case file, create initial pop matrix for model"""
    column_mapping = {value: index for index, value in enumerate(compartment_list)}
    initial_population = np.zeros((len(case_file), len(compartment_list)))

    for i, case in enumerate(case_file):
        infected = round(case["infected_population"] / 100 * case["population"], 2)
        susceptible = case["population"] - infected
        initial_population[i, column_mapping["S"]] = susceptible
        initial_population[i, column_mapping["I"]] = infected

    return initial_population




def create_transmission_dict(transmission_edge_items):
    """Map normalized TransmissionEdges.items to transmission rate variables.

    Values are passed through in their native units so ``_load_transmission_params``
    can apply ``_to_rate()`` exactly once:

    - DAYS:       value is raw days (e.g. 5.0) -> stored as-is; _to_rate() inverts
    - PERCENTAGE: value is whole-number percent (e.g. 4.0) -> stored as-is; _to_rate() divides by 100
    - RATE:       passed through as-is

    Args:
        transmission_edge_items: List of dicts from TransmissionEdges.items[]
    """
    transmission_dict = {}
    for edge in transmission_edge_items:
        lookup = edge.get("transmission_edge", {})
        source = lookup.get("source", "")
        target = lookup.get("target", "")
        edge_id = f"{source}->{target}"
        value_type = lookup.get("value_type", "RATE")

        # Get variable name from FieldConfigs.disease_param if available
        field_configs = edge.get("FieldConfigs", {})
        field_config_items = field_configs.get("items", []) if field_configs else []

        variable = None
        if field_config_items:
            disease_param = field_config_items[0].get("disease_param", "")
            if disease_param:
                variable = disease_param.lower()

        # Fall back to edge_to_variable mapping
        if not variable:
            variable = edge_to_variable.get(edge_id)

        if variable:
            value = edge.get("value", 0)


            # When multiple edges map to the same param (e.g. S->I and S->E
            # both map to beta), prefer the non-zero value so the active
            # path wins over the inactive one.
            if variable not in transmission_dict or value != 0:
                transmission_dict[variable] = value

    return transmission_dict


def build_uncertainty_params(transmission_edge_items: list, intervention_items: list):
    """Build parameter uncertainty from normalized TransmissionEdges.items
    and Interventions.items.

    Args:
        transmission_edge_items: List of dicts from TransmissionEdges.items[]
        intervention_items: List of dicts from Interventions.items[]
    """
    uncertainty_params = []

    if transmission_edge_items:
        for edge in transmission_edge_items:
            field_configs = edge.get("FieldConfigs", {})
            field_config_items = field_configs.get("items", []) if field_configs else []

            for fc in field_config_items:
                if fc.get("has_variance"):
                    param_name = fc.get("disease_param", "").lower()
                    if not param_name:
                        # Fall back to edge_to_variable mapping
                        lookup = edge.get("transmission_edge", {})
                        edge_id = f"{lookup.get('source', '')}->{lookup.get('target', '')}"
                        param_name = edge_to_variable.get(edge_id)
                    if param_name:
                        deterministic_value = edge.get("value", 0)
                        min_value = fc.get("min")
                        max_value = fc.get("max")
                        missing_bounds = [
                            bound
                            for bound, value in (("min", min_value), ("max", max_value))
                            if value is None
                        ]
                        if missing_bounds:
                            logger.warning(
                                "Uncertainty parameter '%s' does not provide %s; "
                                "using its deterministic value (%s) for the missing "
                                "%s.",
                                param_name,
                                " or ".join(missing_bounds),
                                deterministic_value,
                                "bound" if len(missing_bounds) == 1 else "bounds",
                            )
                        uncertainty_params.append(
                            {
                                "param": param_name,
                                "dist": fc.get("distribution_type", "uniform"),
                                "min": min_value
                                if min_value is not None
                                else deterministic_value,
                                "max": max_value
                                if max_value is not None
                                else deterministic_value,
                            }
                        )

    if intervention_items:
        for item in intervention_items:
            intervention_lookup = item.get("Intervention", {})
            intervention_id = intervention_lookup.get("name", "").lower()
            field_configs = item.get("FieldConfigs", {})
            field_config_items = field_configs.get("items", []) if field_configs else []

            for fc in field_config_items:
                if fc.get("has_variance"):
                    field_name = fc.get("field_key")
                    if field_name:
                        uncertainty_params.append(
                            {
                                "param": f"intervention.{intervention_id}.{field_name}",
                                "dist": fc.get("distribution_type", "uniform"),
                                "min": fc.get("min", 0) / 100,
                                "max": fc.get("max", 0) / 100,
                            }
                        )

    return uncertainty_params


def _section_items(section):
    """Return the .items list from a config section that may be either a dict
    or a Pydantic model (or None)."""
    if not section:
        return []
    if isinstance(section, dict):
        return section.get("items", [])
    return getattr(section, "items", [])


def extract_disease_variance_params(disease_section):
    """Normalize disease-parameter variance configs from a local config's
    Disease section into the {param, dist, min, max} shape used for LHS.

    Mirrors the cloud path (gql.get_simulation_job), which derives the same
    shape from SimulationJobCustomField FieldConfig records. Local configs
    instead declare variance inline as Disease.variance_params[].

    Args:
        disease_section: the Disease dict from a loaded local config.
    """
    if not disease_section:
        return []
    return [
        {"dist": vp.get("dist", "uniform"), **vp}
        for vp in disease_section.get("variance_params", [])
    ]


def collect_uncertainty_params(cleaned_config, disease_param_field_configs=None):
    """Gather every parameter variance/uncertainty for a validated config into
    a single flat list suitable for generate_LHS_samples.

    Sources, all merged:
      - TransmissionEdges.items[].FieldConfigs with has_variance
      - Interventions.items[].FieldConfigs with has_variance
      - disease_param_field_configs (disease custom fields, already normalized
        by extract_disease_variance_params for local or gql for cloud)

    Args:
        cleaned_config: the validated Pydantic config.
        disease_param_field_configs: pre-extracted disease variance configs.
    """
    transmission_edge_items = _section_items(
        getattr(cleaned_config, "TransmissionEdges", None)
    )
    intervention_items = _section_items(
        getattr(cleaned_config, "Interventions", None)
    )

    params = build_uncertainty_params(
        as_dict_list(transmission_edge_items),
        as_dict_list(intervention_items),
    )
    if disease_param_field_configs:
        params.extend(disease_param_field_configs)
    return params


def resolve_run_mode(model_class, uncertainty_params):
    """Determine the effective run mode entirely from the model and its params.

    The run_mode field from the frontend config is intentionally ignored — the
    model class is the authoritative source for STOCHASTIC, and relying on the
    frontend value would create edge cases (e.g. UNCERTAINTY with no variance
    params running 30 identical deterministic trajectories).

    Priority order:
    1. STOCHASTIC — model class declares ``STOCHASTIC = True``.  Always runs 30
       trajectories.  If variance parameters are also present they are spread
       across those same 30 runs rather than adding additional runs.
    2. UNCERTAINTY — any variance parameter is declared on an edge, intervention,
       or disease param.
    3. DETERMINISTIC — otherwise.
    """
    if getattr(model_class, "STOCHASTIC", False):
        return "STOCHASTIC"
    if uncertainty_params:
        return "UNCERTAINTY"
    return "DETERMINISTIC"


def extract_admin_units(case_file):
    return [case["name"] for case in case_file]


def _date_to_ordinal(val):
    """Convert a date string or date/datetime object to an ordinal int."""
    if isinstance(val, (date, datetime)):
        return val.toordinal()
    return datetime.strptime(val, "%Y-%m-%d").date().toordinal()


def create_intervention_dict(intervention_items, start_date):
    """Create intervention dict from normalized Interventions.items.

    Args:
        intervention_items: List of dicts from Interventions.items[]
            Each has Intervention (lookup with name), adherence_min,
            transmission_percentage, start_date, end_date, etc.
        start_date: Simulation start date (fallback if no dates/thresholds set).
    """
    intervention_dict = {}
    for item in intervention_items:
        intervention_lookup = item.get("Intervention", {})
        intervention_id = intervention_lookup.get("name", "").lower()

        # convert to ordinal to support jax timestep interventions
        item_start_date = item.get("start_date")
        if item_start_date is not None and item_start_date != "":
            start_date_ordinal = _date_to_ordinal(item_start_date)
        else:
            start_date_ordinal = None
            item_start_date = None

        item_end_date = item.get("end_date")
        if item_end_date is not None and item_end_date != "":
            end_date_ordinal = _date_to_ordinal(item_end_date)
        else:
            end_date_ordinal = None
            item_end_date = None

        if (
            item_start_date is None
            and item_end_date is None
            and item.get("start_threshold") is None
            and item.get("end_threshold") is None
        ):
            item_start_date = start_date
            start_date_ordinal = _date_to_ordinal(start_date)

        intervention_dict[intervention_id] = {
            "start_threshold": item.get("start_threshold") / 100
            if item.get("start_threshold") is not None
            else None,
            "end_threshold": item.get("end_threshold") / 100
            if item.get("end_threshold") is not None
            else None,
            "start_date": item_start_date,
            "start_date_ordinal": start_date_ordinal,
            "end_date": item_end_date,
            "end_date_ordinal": end_date_ordinal,
            "adherence_min": item.get("adherence_min") / 100
            if item.get("adherence_min") is not None
            else None,
            "transmission_percentage": item.get("transmission_percentage") / 100
            if item.get("transmission_percentage") is not None
            else 0.05,
            "start_threshold_node_id": item.get("start_threshold_node_id", None),
            "end_threshold_node_id": item.get("end_threshold_node_id", None),
        }

    return intervention_dict


def get_hemisphere(admin_unit):
    """
    Determine hemisphere using the selected AdminUnit.
    """
    center_lat = admin_unit["center_lat"]  # already validated
    return "North" if center_lat >= 0 else "South"


def has_age_stratification(demographics):
    """
    Check if demographics represent actual age stratification.
    Returns False if demographics is None or represents a single age group.
    """
    if demographics is None:
        return False

    # If it's a dict-like object with age groups
    if hasattr(demographics, "keys"):
        age_keys = list(demographics.keys())
        # More than one age group means stratification
        return len(age_keys) > 1

    return False


def get_demographics_or_default(case_file_dict):
    """
    Get demographics from case_file, or return a simple default.
    Returns a dict with either age groups or just 'age_all'.
    """
    if "demographics" in case_file_dict:
        demographics = case_file_dict["demographics"]
        if has_age_stratification(demographics):
            return demographics

    # No stratification - return simple single-age group
    return {"age_all": 100.0}


def get_temperature(case_file, default_min=0, default_max=38, default_mean=30):
    # Apply first admin zone temperature to all admin zones
    case_file = case_file[0]

    return {
        "temp_min": case_file.get("temp_min", default_min)
        if case_file.get("temp_min") is not None
        else default_min,
        "temp_max": case_file.get("temp_max", default_max)
        if case_file.get("temp_max") is not None
        else default_max,
        "temp_mean": case_file.get("temp_mean", default_mean)
        if case_file.get("temp_mean") is not None
        else default_mean,
    }


def get_simulation_step_size(n_timesteps):
    import math

    return max(math.ceil(n_timesteps / 365), 1)


def get_simulation_time_points(n_timesteps):
    """Return solver/output day offsets including the exact end date.

    ``n_timesteps`` is the elapsed simulation duration in days, not the number
    of output rows. The initial state is reported at day zero and the final
    state at ``n_timesteps``, producing 366 daily observations for a 365-day
    simulation. For downsampled longer simulations, the exact endpoint is
    appended when the regular step does not divide the duration evenly.
    """
    duration = float(n_timesteps)
    step = get_simulation_step_size(n_timesteps)
    time_points = np.arange(0.0, duration, step, dtype=float)

    if time_points.size == 0 or not np.isclose(time_points[-1], duration):
        time_points = np.append(time_points, duration)

    return time_points


def prepare_covid_initial_state(
    initial_population, age_transmission, demographics=None
):
    comp_by_zone = initial_population.T  # shape (4, n_admin_zones)

    if demographics:
        weights = np.array(list(demographics.values()), dtype=float) / 100.0
    else:
        weights = np.array([1.0])
        age_transmission = np.array([1.0])

    # Broadcast multiply → (4, 1, n_admin) * (1, n_age, 1) → (4, n_age, n_admin)
    age_strat = comp_by_zone[:, None, :] * weights[None, :, None]

    return age_strat, age_transmission


# --------------------------------------------------
# Helper Functions: Gravity Model
# --------------------------------------------------


def get_admin_zone_df(case_file):
    ll_pop = pd.DataFrame(
        case_file, columns=["id", "center_lat", "center_lon", "population"]
    )
    ll_pop["lat_long"] = list(zip(ll_pop.center_lat, ll_pop.center_lon))
    ll_pop.drop(columns=["center_lat", "center_lon"], inplace=True)

    # Create df with Cartesian product for calculations between locations
    cross_df = ll_pop.merge(ll_pop, how="cross", suffixes=["_origin", "_destination"])
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

    df["gravity"] = k * df[mass_origin_col] * df[mass_dest_col] / df[distance_col] ** 2
    return df


def create_travel_matrix(input_df, sigma, zone_order=None):
    """
    Measure distances, apply the gravity model, and pivot into a travel matrix.

    ``T[i, j]`` is the fraction of zone *i*'s population present in zone *j*.
    Each row sums to 1: ``sigma`` is distributed across the off-diagonal by
    gravity weight, and the diagonal holds the stay-home remainder.

    Args:
        input_df: Cross-joined origin/destination frame from
            :func:`get_admin_zone_df`.
        sigma: Fraction of each zone's population away from home per day.
        zone_order: Zone ids in population-matrix column order. Required for
            correctness whenever ids aren't already sorted — ``pivot_table``
            sorts its labels, which would otherwise permute the matrix
            relative to the population matrix.
    """
    # Calculating distances between cities
    input_df["distance_km"] = input_df.apply(
        lambda x: geopy.distance.geodesic(x.lat_long_origin, x.lat_long_destination).km,
        axis=1,
    )
    input_df = gravity_model(
        input_df, "population_origin", "population_destination", "distance_km"
    )
    # Self-pairs have distance 0, so gravity is infinite; so is any pair of
    # zones sharing a centroid. Drop them — the diagonal is set explicitly below.
    input_df["gravity"] = input_df["gravity"].replace([np.inf, -np.inf], 0)

    # Share of each origin's outbound trips going to each destination.
    row_totals = input_df.groupby("id_origin")["gravity"].transform("sum")
    input_df["gravity_rate"] = np.where(
        row_totals > 0, input_df["gravity"] / row_totals.replace(0, 1), 0.0
    )

    pivot_df = pd.pivot_table(
        input_df,
        index="id_origin",
        columns="id_destination",
        values="gravity_rate",
        aggfunc="sum",
    )
    if zone_order is not None:
        pivot_df = pivot_df.reindex(index=zone_order, columns=zone_order)

    travel_matrix = pivot_df.fillna(0).to_numpy(dtype=float)
    np.fill_diagonal(travel_matrix, 0.0)

    # Scale each row's off-diagonal mass to sigma. A row with no reachable
    # destination (every pair collapsed to zero gravity) keeps its whole
    # population at home rather than losing sigma of it.
    off_diagonal_totals = travel_matrix.sum(axis=1, keepdims=True)
    travel_matrix = np.divide(
        travel_matrix * sigma,
        off_diagonal_totals,
        out=np.zeros_like(travel_matrix),
        where=off_diagonal_totals > 0,
    )
    np.fill_diagonal(travel_matrix, 1.0 - travel_matrix.sum(axis=1))
    return travel_matrix


def get_gravity_model_travel_matrix(admin_zones, sigma):
    """
    Create a travel matrix using the inverse-square gravity model.

    Call this from a model's ``build_travel_matrix()`` with the model's own
    outbound travel rate. ``sigma`` is a **fraction** (0-1), not a percentage —
    convert PERCENTAGE-typed parameters first (see ``Model._to_rate``).

    Returns the identity matrix when ``sigma`` is 0 or None, and ``[[1.0]]``
    for a single zone.

    Args:
        admin_zones: Admin-zone dicts with ``id``, ``center_lat``,
            ``center_lon`` and ``population``.
        sigma: Fraction of each zone's population away from home per day.
    """
    n_regions = len(admin_zones)

    if n_regions == 1:
        # Single region - no travel needed
        return np.array([[1.0]])

    if not sigma:
        # No travel rate specified - return identity
        return np.eye(n_regions)

    # Key zones positionally rather than by their own ids, so the matrix
    # rows/columns always line up with the population matrix columns
    # (real zone ids are UUIDs, and pivot_table sorts its labels).
    zone_order = list(range(n_regions))
    df = get_admin_zone_df(
        [{**zone, "id": i} for i, zone in enumerate(admin_zones)]
    )
    return create_travel_matrix(df, sigma, zone_order=zone_order)


# --------------------------------------------------
# Helper Functions: Latin Hypercube Sampling
# --------------------------------------------------
def LHS_uniform(low, high, num_runs):
    """Randomly samples values for each parameter n_runs times and scales values
    num_runs: int, number of times to run the simulation
    low: float, minimum value for scaling
    high: float, maximum value for scaling
    returns: 1D array of scaled uniform values
    """
    vals = stats.qmc.LatinHypercube(1).random(num_runs)
    scaled_vals = stats.qmc.scale(vals, low, high).reshape(num_runs)
    return scaled_vals


def LHS_normal(mean, std, uniform_samples):
    """Scales randomly sampled uniform values to a normal distribution
    mean: float, mean value of distribution
    std: float, standard deviation of normal distribution
    num_runs: int, number of times to run the simulation
    returns: 1D array of scaled uniform values in shape of normal distribution
    """
    return stats.norm.ppf(uniform_samples, loc=mean, scale=std)


def LHS_triangular(min, probability_mode, max, uniform_samples):
    """Scales randomly sampled uniform values to a triangular distribution
    min: float, minimum value of distribution
    probability_mode: float, shape parameter for the triangular distribution. Represents the mode of the distribution in its standardized form,
        and must be between 0 and 1 (inclusive). Defines the peak of the triangular distribution relative to its base.
    max: float, maximum value of distribution
    num_runs: int, number of times to run the simulation
    returns: 1D array of scaled uniform values in shape of triangular distribution
    """
    return stats.triang.ppf(
        uniform_samples, loc=min, c=probability_mode, scale=max - min
    )


def LHS_beta(alpha, beta, uniform_samples):
    """Scales randomly sampled uniform values to a beta distribution
    alpha: float, exponent variable, power function of the variable x
    beta: float, complement of the variable x (1-x)
    returns: 1D array of scaled uniform values in shape of beta distribution
    """
    return stats.beta.ppf(uniform_samples, alpha, beta)


# NOTE: loc parameter in scipy.stats.lognorm.ppf defaults to 0, which corresponds to the standard log-normal distribution.
# If your distribution is shifted, you may need to adjust this parameter.
def LHS_lognormal(mean, sigma, uniform_samples):
    """Scales randomly sampled uniform values to a lognormal distribution
    sigma: float, shape parameter of the lognormal distribution
    mean: float, used to scale the distribution
    returns: 1D array of scaled uniform values in shape of lognormal distribution
    """
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
        name = cfg["param"]
        dist = cfg["dist"].lower()

        if dist == "uniform":
            low, high = cfg["min"], cfg["max"]
            samples = LHS_uniform(low, high, num_runs)
        else:
            base = LHS_uniform(0, 1, num_runs)
            if dist == "normal":
                samples = LHS_normal(cfg["mean"], cfg["std"], base)
            elif dist == "triangular":
                samples = LHS_triangular(
                    cfg["min"], cfg["probability_mode"], cfg["max"], base
                )
            elif dist == "beta":
                samples = LHS_beta(cfg["alpha"], cfg["beta"], base)
            elif dist == "lognormal":
                samples = LHS_lognormal(cfg["mean"], cfg["sigma"], base)
            else:
                raise ValueError(f"Unsupported distribution type: {cfg['dist']}")
        results[name] = samples.tolist()

    param_list = []
    for i in range(num_runs):
        entry = {p: results[p][i] for p in results}
        param_list.append(entry)
    return param_list


# --------------------------------------------------
# Helper Functions: Misc
# --------------------------------------------------


def get_executor_class():
    """Get the appropriate executor class, falling back to ThreadPoolExecutor if multiprocessing fails."""
    try:
        with ThreadPoolExecutor(max_workers=1) as test_executor:
            pass
        return ThreadPoolExecutor
    except (OSError, RuntimeError, ValueError):
        return ThreadPoolExecutor


def as_dict_list(obj):
    """Normalize a Pydantic model or list of models/dicts to list of dicts.

    Handles cases where obj might be:
    - List of Pydantic models (call model_dump() on each)
    - List of dicts (return as-is)
    - Single Pydantic model (call model_dump() and wrap in list)
    - Single dict (wrap in list)
    - None (return empty list)
    """
    if obj is None:
        return []

    # Handle list/sequence
    if isinstance(obj, (list, tuple)):
        result = []
        for item in obj:
            if hasattr(item, "model_dump"):
                result.append(item.model_dump())
            elif isinstance(item, dict):
                result.append(item)
            else:
                result.append(item)
        return result

    # Handle single item
    if hasattr(obj, "model_dump"):
        return [obj.model_dump()]
    elif isinstance(obj, dict):
        return [obj]

    return []
