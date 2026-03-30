"""
Runtime domain classes for disease models.

These classes bridge the gap between schema-time definitions
(:mod:`compartment.parameters`) and the ODE solver.  They hold
config-derived values and provide JAX-safe methods that the model
calls during ``derivative()``.

- :class:`TransmissionEdge` — schema metadata + current rate value,
  with :meth:`compute_flow` for automatic flow computation.
- :class:`Intervention` — schema metadata + config values, with
  JAX-compatible activation checks and rate modification.

Both classes are **standalone** — they hold no back-references to the
model.  The model creates them from the config and calls their methods,
passing context (populations, time, etc.).

These classes are JAX-safe: they use ``jnp.where`` / ``jnp.logical_and``
for branching and never mutate state during the solver run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import jax.numpy as jnp
except ImportError:
    import numpy as jnp  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# TransmissionEdge — runtime edge with compute_flow()
# ---------------------------------------------------------------------------


@dataclass
class TransmissionEdge:
    """
    Runtime transmission edge — schema metadata + current rate value.

    Created from a :class:`TransmissionEdgeDef` and the corresponding
    config rate value.  The model stores a list of these and can call
    :meth:`compute_flow` for automatic derivative computation.

    Attributes:
        source_id: Compartment the flow leaves (e.g. ``"S"``).
        target_id: Compartment the flow enters (e.g. ``"I"``).
        variable_name: Rate key in the params dict (e.g. ``"beta"``).
        rate: Current rate value from config (e.g. ``0.3``).
        frequency_dependent: If ``True``, uses FOI formula.
    """

    source_id: str
    target_id: str
    variable_name: str
    rate: float
    frequency_dependent: bool = False

    def compute_flow(
        self,
        source_pop: Any,
        infective_sum: Any = None,
        N_total: Any = None,
    ) -> Any:
        """
        Compute the flow for this edge.  JAX-safe, no side effects.

        For frequency-dependent edges:
            ``source_pop * rate * infective_sum / (N_total + 1e-10)``

        For standard edges:
            ``rate * source_pop``

        Args:
            source_pop: Population array of the source compartment.
            infective_sum: Sum of all infective compartments (required
                for frequency-dependent edges).
            N_total: Total population (required for frequency-dependent
                edges).

        Returns:
            Flow array (same shape as source_pop).
        """
        if self.frequency_dependent:
            return source_pop * self.rate * infective_sum / (N_total + 1e-10)
        return self.rate * source_pop

    @classmethod
    def from_schema(cls, edge_def, rate: float) -> TransmissionEdge:
        """
        Create a runtime edge from a :class:`TransmissionEdgeDef` and
        the corresponding config rate value.

        Args:
            edge_def: Schema definition from ``define_parameters()``.
            rate: Rate value from the config's ``transmission_dict``.

        Returns:
            A new :class:`TransmissionEdge` instance.
        """
        return cls(
            source_id=edge_def.source_id,
            target_id=edge_def.target_id,
            variable_name=edge_def.variable_name,
            rate=rate,
            frequency_dependent=edge_def.frequency_dependent,
        )


# ---------------------------------------------------------------------------
# Intervention — runtime intervention with JAX-safe activation + apply
# ---------------------------------------------------------------------------


@dataclass
class Intervention:
    """
    Runtime intervention with JAX-compatible activation and rate modification.

    Created from an :class:`InterventionDef` (schema) and the
    corresponding config dict entry (runtime values).  The model
    creates a list of these in ``__init__`` and calls their methods
    in ``derivative()``.

    The activation logic replicates exactly the ``_update_one`` and
    ``_update_date`` helpers from ``interventions.py``, using
    ``jnp.where`` and ``jnp.logical_and`` for JAX traceability.

    Attributes:
        id: Intervention identifier (e.g. ``"mask_wearing"``).
        target_rates: Variable names of edges this modifies (e.g. ``["beta"]``).
        modifies_travel: If ``True``, replaces travel matrix with identity
            when active (lockdown behavior).
        adherence: Population adherence fraction (0.0 - 1.0).
        transmission_reduction: Transmission reduction fraction (0.0 - 1.0).
        start_date_ordinal: Ordinal day the intervention starts (or ``None``).
        end_date_ordinal: Ordinal day the intervention ends (or ``None``).
        start_threshold: Infection proportion that triggers activation (or ``None``).
        end_threshold: Infection proportion that triggers deactivation (or ``None``).
    """

    id: str
    target_rates: list[str]
    modifies_travel: bool
    adherence: float
    transmission_reduction: float
    start_date_ordinal: int | None
    end_date_ordinal: int | None
    start_threshold: float | None
    end_threshold: float | None

    def check_date_activation(
        self,
        current_day_ordinal: Any,
        current_status: Any,
    ) -> tuple[Any, Any]:
        """
        Compute intervention activation based on date window.

        Replicates the ``_update_date`` helper from ``interventions.py``.
        Returns a tuple of ``(apply_flag, new_status)`` where:

        - ``apply_flag``: Whether to apply rate reduction right now.
          This is ``in_window`` (True only during the date window).
          **Rate reduction is not persistent** — once outside the
          window, rates bounce back.
        - ``new_status``: Persisted status flag (True if ever activated,
          stays True even after window ends).

        If no ``start_date_ordinal`` is set, returns
        ``(current_status, current_status)`` unchanged (no date trigger).

        Args:
            current_day_ordinal: Current simulation day as ordinal.
            current_status: Current activation status (JAX boolean).

        Returns:
            Tuple of (apply_flag, new_status).
        """
        if self.start_date_ordinal is None:
            return current_status, current_status

        if self.end_date_ordinal is None:
            in_window = current_day_ordinal >= self.start_date_ordinal
        else:
            in_window = jnp.logical_and(
                current_day_ordinal >= self.start_date_ordinal,
                current_day_ordinal <= self.end_date_ordinal,
            )

        new_status = jnp.where(in_window, True, current_status)
        return in_window, new_status

    def check_threshold_activation(
        self,
        prop_infective: Any,
        current_status: Any,
    ) -> Any:
        """
        Compute intervention status based on infection threshold.

        Replicates the ``_update_one`` helper from ``interventions.py``.
        Returns the new status (JAX-traced boolean).

        If no ``start_threshold`` is set, returns *current_status*
        unchanged (no threshold trigger configured).

        Args:
            prop_infective: Proportion of population that is infective (scalar).
            current_status: Current activation status (JAX boolean).

        Returns:
            New activation status (JAX boolean).
        """
        if self.start_threshold is None:
            turn_on = jnp.bool_(False)
        else:
            turn_on = jnp.logical_and(
                prop_infective >= self.start_threshold,
                jnp.logical_not(current_status),
            )

        if self.end_threshold is None:
            turn_off = jnp.bool_(False)
        else:
            turn_off = jnp.logical_and(
                prop_infective <= self.end_threshold,
                current_status,
            )

        return jnp.where(
            turn_on,
            True,
            jnp.where(turn_off, False, current_status),
        )

    def apply_to_rates(
        self,
        rates: dict[str, Any],
        active: Any,
    ) -> dict[str, Any]:
        """
        Apply transmission reduction to target rates.

        For each rate in ``target_rates``, computes:
        ``jnp.where(active, rate * (1 - adherence * reduction), rate)``

        This matches the formula in ``interventions.py``'s ``_update_one``
        and ``_update_date`` helpers.

        Args:
            rates: Dict of rate variable names → current values.
            active: JAX boolean indicating whether the intervention is active.

        Returns:
            New rates dict with target rates conditionally reduced.
        """
        for rate_name in self.target_rates:
            if rate_name in rates:
                comp_rate = rates[rate_name]
                reduced = comp_rate * (
                    1.0 - self.adherence * self.transmission_reduction
                )
                rates = {**rates, rate_name: jnp.where(active, reduced, comp_rate)}
        return rates

    def apply_to_travel(
        self,
        travel_matrix: Any,
        active: Any,
    ) -> Any:
        """
        Replace travel matrix with identity when active (lockdown).

        Only has an effect if ``modifies_travel`` is ``True``.

        Args:
            travel_matrix: Current travel matrix (JAX array).
            active: JAX boolean indicating whether the intervention is active.

        Returns:
            Travel matrix (identity if active and modifies_travel, else unchanged).
        """
        if not self.modifies_travel:
            return travel_matrix
        active_scalar = jnp.any(active)
        return jnp.where(
            active_scalar,
            jnp.eye(travel_matrix.shape[0]),
            travel_matrix,
        )

    @classmethod
    def from_config(
        cls,
        intervention_def,
        cfg: dict[str, Any],
    ) -> Intervention:
        """
        Create a runtime intervention from a schema definition and
        the corresponding entry in the config's ``intervention_dict``.

        The config dict has already been processed by
        ``create_intervention_dict()`` (values divided by 100, dates
        converted to ordinals).

        Args:
            intervention_def: Schema definition from ``define_parameters()``.
            cfg: Config dict entry for this intervention, e.g.::

                {
                    "adherence_min": 0.3,
                    "transmission_percentage": 0.35,
                    "start_date_ordinal": 738156,
                    "end_date_ordinal": None,
                    "start_threshold": 0.05,
                    "end_threshold": 0.01,
                }

        Returns:
            A new :class:`Intervention` instance.
        """
        return cls(
            id=intervention_def.id,
            target_rates=intervention_def.target_rates,
            modifies_travel=intervention_def.modifies_travel,
            adherence=cfg.get("adherence_min") or 0.0,
            transmission_reduction=cfg.get("transmission_percentage") or 0.0,
            start_date_ordinal=cfg.get("start_date_ordinal"),
            end_date_ordinal=cfg.get("end_date_ordinal"),
            start_threshold=cfg.get("start_threshold"),
            end_threshold=cfg.get("end_threshold"),
        )
