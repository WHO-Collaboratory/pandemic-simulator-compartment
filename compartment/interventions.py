import logging
from compartment.helpers import setup_logging
import jax.numpy as jnp

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

# --------------------------------------------------
# JAX-compatible interventions
# --------------------------------------------------

def jax_prop_intervention(intervention_dict,
                                   prop_infectives: float,
                                   rates: dict,
                                   intervention_statuses: dict,
                                   travel_matrix: jnp.ndarray):
    # private helper that returns (new_rate, new_status)
    def _update_one(comp_rate, status, cfg):
        start_th = cfg.get("start_threshold")
        end_th   = cfg.get("end_threshold")
        adh      = cfg["adherence_min"]
        reduc    = cfg["transmission_percentage"]

        if start_th is None:
            turn_on = jnp.bool_(False)
        else:
            turn_on = jnp.logical_and(prop_infectives >= start_th,
                                      jnp.logical_not(status))

        if end_th is None:
            turn_off = jnp.bool_(False)
        else:
            turn_off = jnp.logical_and(prop_infectives <= end_th,
                                       status)
            
        new_status = jnp.where(turn_on,
                               True,
                               jnp.where(turn_off, False, status))

        reduced_rate = comp_rate * (1.0 - adh * reduc)
        new_rate     = jnp.where(new_status, reduced_rate, comp_rate)
        return new_rate, new_status

    # Update the physical intervention if defined
    if "physical" in intervention_dict:
        new_b, new_flag = _update_one(
            rates["b_V_T"],
            intervention_statuses["physical"],
            intervention_dict["physical"]
        )
        rates = {**rates, "b_V_T": new_b} 
        intervention_statuses = {**intervention_statuses,
                                 "physical": new_flag}
        
    # Update the chemical intervention if defined
    if "chemical" in intervention_dict:
        new_s, new_flag = _update_one(
            rates["s_V_T"],
            intervention_statuses["chemical"],
            intervention_dict["chemical"]
        )
        rates = {**rates, "s_V_T": new_s}
        intervention_statuses = {**intervention_statuses,
                                 "chemical": new_flag}

    # COVID interventions
    for name in ("social_isolation", "vaccination", "mask_wearing"):
        if name in intervention_dict and "beta" in rates:
            new_beta, new_flag = _update_one(
                rates["beta"],
                intervention_statuses.get(name, False),
                intervention_dict[name]
            )
            rates = {**rates, "beta": new_beta}
            intervention_statuses = {**intervention_statuses, name: new_flag}

    # COVID lock down
    if "lock_down" in intervention_dict:
        cfg = intervention_dict["lock_down"]
        # reuse _update_one to compute the on/off status - rate is irrelevant
        _, new_flag = _update_one(
            0.0, 
            intervention_statuses.get("lock_down", False),
            cfg
        )

        new_flag_scalar = jnp.any(new_flag)
        travel_matrix = jnp.where(new_flag_scalar,
                                  jnp.eye(travel_matrix.shape[0]),
                                  travel_matrix)
        intervention_statuses = {**intervention_statuses, "lock_down": new_flag_scalar}

    return rates, intervention_statuses, travel_matrix


def jax_timestep_intervention(intervention_dict: dict,
                              current_ordinal_day: int,
                              rates: dict,
                              intervention_statuses: dict,
                              travel_matrix: jnp.ndarray):
    # private helper that returns (new_rate, new_status)
    def _update_date(comp_rate, status, cfg, current_ordinal_day):
        # Retrieve the intervention date bounds and parameters
        start_date_ordinal = cfg.get("start_date_ordinal")
        end_date_ordinal = cfg.get("end_date_ordinal")
        adh = cfg["adherence_min"]
        reduc = cfg["transmission_percentage"]

        # If no start date is provided we cant activate the intervention
        if start_date_ordinal is None:
            return comp_rate, status

        if end_date_ordinal is None:
            in_window = current_ordinal_day >= start_date_ordinal
        else:
            in_window = jnp.logical_and(current_ordinal_day >= start_date_ordinal,
                                        current_ordinal_day <= end_date_ordinal)

        new_status = jnp.where(in_window, True, status)
        new_rate   = jnp.where(in_window,
                               comp_rate * (1 - adh * reduc),
                               comp_rate)
        return new_rate, new_status

    # Update Dengue physical intervention if defined
    if "physical" in intervention_dict:
        new_b, new_flag = _update_date(rates["b_V_T"],
                                       intervention_statuses["physical"],
                                       intervention_dict["physical"],
                                       current_ordinal_day)
        rates = {**rates, "b_V_T": new_b}
        intervention_statuses = {**intervention_statuses, "physical": new_flag}

    # Update Dengue chemical intervention if defined
    if "chemical" in intervention_dict:
        new_s, new_flag = _update_date(rates["s_V_T"],
                                       intervention_statuses["chemical"],
                                       intervention_dict["chemical"],
                                       current_ordinal_day)
        rates = {**rates, "s_V_T": new_s}
        intervention_statuses = {**intervention_statuses, "chemical": new_flag}

    # upate SOCIAL_ISOLATION, VACCINATION, MASK_WEARING for covid model
    for name in ("social_isolation", "vaccination", "mask_wearing"):
        if name in intervention_dict and "beta" in rates:
            new_beta, new_flag = _update_date(
                rates["beta"],
                intervention_statuses.get(name, False),
                intervention_dict[name],
                current_ordinal_day,
            )
            rates = {**rates, "beta": new_beta}
            intervention_statuses = {**intervention_statuses, name: new_flag}

    # COVID lock down
    if "lock_down" in intervention_dict:
      _, new_flag = _update_date(
          0.0,
          intervention_statuses.get("lock_down", False),
          intervention_dict["lock_down"],
          current_ordinal_day
      )
  
      travel_matrix = jnp.where(new_flag,
                                jnp.eye(travel_matrix.shape[0]),
                                travel_matrix)
  
      intervention_statuses = {**intervention_statuses, "lock_down": new_flag}

    return rates, intervention_statuses, travel_matrix
