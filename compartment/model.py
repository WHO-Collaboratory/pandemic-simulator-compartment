"""Abstract base class for all models"""
from abc import ABC,abstractmethod
class Model(ABC):
    def __init__(self, config:dict):
        self.config = config
    
    @property
    @abstractmethod
    def disease_type(self):
        pass

    def prepare_initial_state(self):
        pass

    def get_params(self):
        pass

    def derivative(self, y, t, p):
        pass