from __future__ import annotations
from typing import Any, Dict, List, Callable
from pydantic import BaseModel, Field, ConfigDict

from compartment.validation.simulation_config import SimulationConfig
from compartment.helpers import (
    create_initial_population_matrix,
    create_transmission_dict,
    extract_admin_units,
    create_intervention_dict,
    get_hemisphere,
    get_temperature,
)


class ProcessedSimulation(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    # Original config data
    config: Dict[str, Any]  # The validated config as a dict

    # Computed fields (added by post-processing)
    compartment_list: List[str] = Field(default_factory=list)
    initial_population: Any = Field(default=None)
    transmission_dict: Dict = Field(default_factory=dict)
    admin_units: List = Field(default_factory=list)
    intervention_dict: Dict = Field(default_factory=dict)
    hemisphere: str = Field(default="")
    temperature: Dict = Field(default_factory=dict)

    # Support dict-like access for backward compatibility with Model classes
    def __getitem__(self, key):
        """Allow dict-style access: config["key"]"""
        if hasattr(self, key):
            value = getattr(self, key)
            # Convert Pydantic models/lists to dicts for nested access
            if hasattr(value, "model_dump"):
                return value.model_dump()
            return value
        return self.config.get(key)

    def get(self, key, default=None):
        """Allow dict.get() style access"""
        if hasattr(self, key):
            value = getattr(self, key)
            # Convert Pydantic models/lists to dicts for nested access
            if hasattr(value, "model_dump"):
                return value.model_dump()
            return value
        return self.config.get(key, default)

    def __getattr__(self, name):
        """
        Allow attribute access for fields in config dict.
        This enables config.time_steps even when time_steps isn't a direct field.
        """
        # Avoid infinite recursion by checking __dict__ first
        if name in ("config", "model_config", "model_fields"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        # Check if it's in the config dict
        if "config" in self.__dict__ and name in self.config:
            return self.config[name]

        # Not found
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )


class ValidationPostProcessor:
    # Registry of custom processors: disease_type -> processor_function
    _processors: Dict[str, Callable[[SimulationConfig], ProcessedSimulation]] = {}

    @classmethod
    def register_processor(
        cls,
        disease_type: str,
        processor_func: Callable[[SimulationConfig], ProcessedSimulation],
    ) -> None:
        cls._processors[disease_type] = processor_func

    @staticmethod
    def process(
        config: SimulationConfig, model_class: type | None = None
    ) -> ProcessedSimulation:
        disease_type = config.Disease.disease_type

        # Check for custom registered processor (rare - only if you need special logic)
        if disease_type in ValidationPostProcessor._processors:
            return ValidationPostProcessor._processors[disease_type](config)

        # Use smart default processor (works for all diseases automatically!)
        return ValidationPostProcessor._process_default(
            config, model_class=model_class
        )

    @staticmethod
    def _process_default(
        config: SimulationConfig, model_class: type | None = None
    ) -> ProcessedSimulation:
        # Convert to dicts for helper functions
        disease_dict = config.Disease.model_dump()
        disease_type = config.Disease.disease_type
        admin_zones_dicts = [z.model_dump() for z in config.case_file.admin_zones] if config.case_file else []
        admin_unit_dict = config.AdminUnit.model_dump() if config.AdminUnit else None

        # Extract normalized TransmissionEdges and Interventions
        transmission_edge_items = []
        if config.TransmissionEdges:
            transmission_edge_items = [
                e.model_dump() for e in config.TransmissionEdges.items
            ]

        intervention_items = []
        if config.Interventions:
            intervention_items = [
                i.model_dump() for i in config.Interventions.items
            ]

        if model_class is None:
            # Backward compatibility for callers that invoke the post-processor
            # directly. This shortcut works only for an unambiguous disease type;
            # the normal validation path passes the exact model class instead.
            from compartment.registry import resolve

            model_class = resolve(disease_type)

        # === COMPARTMENT LIST ===
        # All models define their compartments via COMPARTMENT_LIST on the class.
        # The config's compartment_list is a fallback for models that don't.
        if model_class and hasattr(model_class, "COMPARTMENT_LIST"):
            compartment_list = model_class.COMPARTMENT_LIST
        elif disease_dict.get("compartment_list"):
            compartment_list = disease_dict["compartment_list"]
        else:
            raise ValueError(
                "'compartment_list' must be provided in Disease config for models "
                "without a fixed COMPARTMENT_LIST."
            )

        # === INITIAL POPULATION (use Model class methods) ===
        if model_class:
            # Let the Model compute its own initial population
            initial_population = model_class.get_initial_population(
                admin_zones=admin_zones_dicts, compartment_list=compartment_list
            )
        else:
            # Fallback to default implementation for unregistered disease types
            initial_population = create_initial_population_matrix(
                admin_zones_dicts, compartment_list
            )

        # === TRANSMISSION DICT (from normalized TransmissionEdges) ===
        transmission_dict = {}
        if transmission_edge_items:
            transmission_dict = create_transmission_dict(transmission_edge_items)

        # === COMMON DERIVED FIELDS (only compute if data available) ===
        admin_units = extract_admin_units(admin_zones_dicts)
        intervention_dict = (
            create_intervention_dict(intervention_items, config.start_date)
            if intervention_items
            else {}
        )
        # NOTE: the travel matrix is *not* built here. Each model declares its
        # own mobility parameters (conventionally ``travel_sigma``) as custom
        # fields and builds its own matrix in ``Model.build_travel_matrix()``,
        # which the framework invokes before ``prepare_initial_state()``.
        hemisphere = get_hemisphere(admin_unit_dict) if admin_unit_dict else ""
        temperature = get_temperature(admin_zones_dicts) if admin_zones_dicts else {}

        return ProcessedSimulation(
            config=config.model_dump(),
            compartment_list=compartment_list,
            initial_population=initial_population,
            transmission_dict=transmission_dict,
            admin_units=admin_units,
            intervention_dict=intervention_dict,
            hemisphere=hemisphere,
            temperature=temperature,
        )
