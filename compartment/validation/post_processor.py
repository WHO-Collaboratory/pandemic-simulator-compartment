"""
Automatic post-processing for simulation configurations.

This module automatically processes validated configs - NO manual processors needed!

The processor intelligently detects what fields exist in your disease config and
processes them automatically:
    - Compartment lists (from disease_nodes OR compartment_list)
    - Initial population (standard OR dengue-specific)
    - Transmission dicts (if transmission_edges exist)
    - Admin units, interventions, travel, hemisphere, temperature (if data available)

The processing happens AFTER validation, keeping concerns separated:
    1. Validate config (SimulationConfig)
    2. Auto post-process (ValidationPostProcessor)
    3. Run simulation

Benefits:
- ✅ Fully automatic - no processor creation needed
- ✅ Handles all disease types (respiratory, vector-borne, etc.)
- ✅ Intelligently detects and processes available fields
- ✅ Only computes what's needed (no wasted processing)

IMPORTANT FOR USERS:
- You DON'T need to create a post-processor!
- The system automatically detects and processes your disease config
- Only register a custom processor if you have very unique requirements (rare!)
"""
from __future__ import annotations
from typing import Any, Dict, List, Callable, Optional
from pydantic import BaseModel, Field, ConfigDict

from compartment.validation.simulation_config import SimulationConfig
from compartment.helpers import (
    create_initial_population_matrix,
    create_compartment_list,
    create_transmission_dict,
    extract_admin_units,
    create_intervention_dict,
    get_gravity_model_travel_matrix,
    get_hemisphere,
    get_temperature,
    get_dengue_initial_population,
    create_dengue_compartment_list,
)


class ProcessedSimulation(BaseModel):
    """
    Result of post-processing a SimulationConfig.
    
    Contains all the computed fields that are needed for running the simulation,
    plus the original config data.
    
    This object can be used directly by Model classes - it provides both
    dict-like access (config["key"]) and attribute access (config.key).
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    # Original config data
    config: Dict[str, Any]  # The validated config as a dict
    
    # Computed fields (added by post-processing)
    compartment_list: List[str] = Field(default_factory=list)
    initial_population: Any = Field(default=None)
    transmission_dict: Dict = Field(default_factory=dict)
    admin_units: List = Field(default_factory=list)
    intervention_dict: Dict = Field(default_factory=dict)
    travel_matrix: Any = Field(default=None)
    hemisphere: str = Field(default="")
    temperature: Dict = Field(default_factory=dict)
    
    # Support dict-like access for backward compatibility with Model classes
    def __getitem__(self, key):
        """Allow dict-style access: config["key"]"""
        if hasattr(self, key):
            value = getattr(self, key)
            # Convert Pydantic models/lists to dicts for nested access
            if hasattr(value, 'model_dump'):
                return value.model_dump()
            return value
        return self.config.get(key)
    
    def get(self, key, default=None):
        """Allow dict.get() style access"""
        if hasattr(self, key):
            value = getattr(self, key)
            # Convert Pydantic models/lists to dicts for nested access
            if hasattr(value, 'model_dump'):
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
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        # Check if it's in the config dict
        if "config" in self.__dict__ and name in self.config:
            return self.config[name]
        
        # Not found
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class ValidationPostProcessor:
    """
    Handles post-processing of validated simulation configs.
    
    This class provides:
    1. A DEFAULT processor that works for most diseases (no custom code needed!)
    2. A registry for custom processors (only if you need special logic)
    3. Built-in processors for existing diseases (backward compatibility)
    
    Usage (Simple - works for most diseases):
        config = SimulationConfig[MyDiseaseConfig].model_validate(data)
        processed = ValidationPostProcessor.process(config)
        # Uses default processor automatically!
    
    Usage (Custom - only if you need special logic):
        # Define custom processor
        def my_custom_processor(config):
            # Your custom logic here
            return ProcessedSimulation(...)
        
        # Register it
        ValidationPostProcessor.register_processor("MY_DISEASE", my_custom_processor)
        
        # Now process uses your custom logic
        processed = ValidationPostProcessor.process(config)
    """
    
    # Registry of custom processors: disease_type -> processor_function
    _processors: Dict[str, Callable[[SimulationConfig], ProcessedSimulation]] = {}
    
    @classmethod
    def register_processor(
        cls, 
        disease_type: str, 
        processor_func: Callable[[SimulationConfig], ProcessedSimulation]
    ) -> None:
        """
        Register a custom processor for a specific disease type.
        
        Only needed if your disease has unique post-processing requirements
        that the default processor can't handle.
        
        Args:
            disease_type: The disease type string (e.g., "MY_DISEASE")
            processor_func: Function that takes SimulationConfig and returns ProcessedSimulation
        
        Example:
            def process_my_disease(config: SimulationConfig) -> ProcessedSimulation:
                # Custom logic here
                return ProcessedSimulation(...)
            
            SimulationPostProcessor.register_processor("MY_DISEASE", process_my_disease)
        """
        cls._processors[disease_type] = processor_func
    
    @staticmethod
    def process(config: SimulationConfig) -> ProcessedSimulation:
        """
        Automatically post-process a validated config into a ProcessedSimulation.
        
        The processor intelligently detects what fields exist and processes them
        accordingly. No manual post-processor creation needed!
        
        Tries processors in this order:
        1. Custom processor (if registered for this disease_type) - rare, only for special cases
        2. Automatic smart processor (detects and processes all available fields)
        
        Args:
            config: Validated SimulationConfig instance
            
        Returns:
            ProcessedSimulation with all computed fields
        """
        disease_type = config.Disease.disease_type
        
        # Check for custom registered processor (rare - only if you need special logic)
        if disease_type in ValidationPostProcessor._processors:
            return ValidationPostProcessor._processors[disease_type](config)
        
        # Use smart default processor (works for all diseases automatically!)
        return ValidationPostProcessor._process_default(config)
    
    @staticmethod
    def _process_default(config: SimulationConfig) -> ProcessedSimulation:
        """
        Smart default processor that automatically handles ALL disease types.
        
        Automatically detects and processes:
        - Vector-borne diseases (uses dengue-specific functions when disease_type is VECTOR_BORNE)
        - Respiratory diseases (standard processing)
        - Any other disease type (generic processing)
        
        Intelligently handles:
        - Compartment extraction from disease_nodes OR compartment_list
        - Initial population (dengue-specific or standard)
        - Transmission dict (only if transmission_edges exist)
        - Admin units, interventions, travel, hemisphere, temperature (only if data available)
        
        No manual processor creation needed - just works!
        """
        # Convert to dicts for helper functions
        disease_dict = config.Disease.model_dump()
        disease_type = config.Disease.disease_type
        admin_zones_dicts = [z.model_dump() for z in config.case_file.admin_zones]
        interventions_list = (
            [i.model_dump() for i in config.interventions] if config.interventions else []
        )
        admin0_dict = config.AdminUnit0.model_dump() if config.AdminUnit0 else None
        admin1_dict = config.AdminUnit1.model_dump() if config.AdminUnit1 else None
        admin2_dict = config.AdminUnit2.model_dump() if config.AdminUnit2 else None
        travel_volume_dict = config.travel_volume.model_dump() if config.travel_volume else None
        
        # === COMPARTMENT LIST (automatically detect source) ===
        if disease_type == "VECTOR_BORNE":
            # Vector-borne diseases like dengue use special compartment list
            compartment_list = create_dengue_compartment_list(disease_type)
        elif disease_dict.get('compartment_list'):
            # Explicit compartment list provided
            compartment_list = disease_dict['compartment_list']
        elif disease_dict.get('disease_nodes'):
            # Extract from disease graph structure
            compartment_list = create_compartment_list(disease_dict['disease_nodes'])
        else:
            raise ValueError(
                "Either 'disease_nodes' or 'compartment_list' must be provided in Disease config. "
                "If your disease doesn't have compartments, register a custom processor."
            )
        
        # === INITIAL POPULATION (automatically use appropriate function) ===
        if disease_type == "VECTOR_BORNE":
            # Vector-borne diseases like dengue use special initialization
            initial_population = get_dengue_initial_population(
                admin_zones_dicts, 
                compartment_list, 
                config.run_mode
            )
        else:
            # Standard initialization for all other diseases
            initial_population = create_initial_population_matrix(admin_zones_dicts, compartment_list)
        
        # === TRANSMISSION DICT (only if transmission_edges exist) ===
        transmission_dict = {}
        if disease_dict.get('transmission_edges'):
            try:
                transmission_dict = create_transmission_dict(disease_dict['transmission_edges'])
            except (KeyError, AttributeError):
                # If the helper can't process transmission edges (e.g., missing 'id' field
                # or not in predefined mapping), create a simple dict
                for edge in disease_dict['transmission_edges']:
                    # Create a key from source->target
                    key = f"{edge.get('source', '')}_{edge.get('target', '')}"
                    rate = edge.get('data', {}).get('transmission_rate', edge.get('transmission_rate', 0))
                    transmission_dict[key] = rate
        
        # === COMMON DERIVED FIELDS (only compute if data available) ===
        admin_units = extract_admin_units(admin_zones_dicts)
        intervention_dict = create_intervention_dict(interventions_list, config.start_date) if interventions_list else {}
        travel_matrix = get_gravity_model_travel_matrix(admin_zones_dicts, travel_volume_dict) if travel_volume_dict else None
        hemisphere = get_hemisphere(admin2_dict, admin1_dict, admin0_dict) if (admin2_dict or admin1_dict or admin0_dict) else ""
        temperature = get_temperature(admin_zones_dicts) if admin_zones_dicts else {}
        
        return ProcessedSimulation(
            config=config.model_dump(),
            compartment_list=compartment_list,
            initial_population=initial_population,
            transmission_dict=transmission_dict,
            admin_units=admin_units,
            intervention_dict=intervention_dict,
            travel_matrix=travel_matrix,
            hemisphere=hemisphere,
            temperature=temperature,
        )
