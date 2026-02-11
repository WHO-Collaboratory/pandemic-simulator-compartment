"""Abstract base class for all models"""
from abc import ABC, abstractmethod
import numpy as np

class Model(ABC):
    def __init__(self, config: dict):
        self.config = config
    
    @property
    @abstractmethod
    def disease_type(self):
        pass

    @classmethod
    def get_initial_population(cls, admin_zones, compartment_list, **kwargs):
        """
        Compute initial population distribution for this disease model.
        
        Default implementation for simple S-I models.
        Override in subclasses for disease-specific initialization logic.
        
        Args:
            admin_zones: List of admin zone dicts with population data
            compartment_list: List of compartment names
            **kwargs: Model-specific parameters (run_mode, etc.)
        
        Returns:
            numpy array of shape (n_zones, n_compartments)
        """
        column_mapping = {value: index for index, value in enumerate(compartment_list)}
        initial_population = np.zeros((len(admin_zones), len(compartment_list)))

        for i, zone in enumerate(admin_zones):
            infected = round(zone['infected_population'] / 100 * zone['population'], 2)
            susceptible = zone['population'] - infected
            initial_population[i, column_mapping['S']] = susceptible
            initial_population[i, column_mapping['I']] = infected

        return initial_population

    def prepare_initial_state(self):
        pass

    def get_params(self):
        pass

    def derivative(self, y, t, p):
        pass