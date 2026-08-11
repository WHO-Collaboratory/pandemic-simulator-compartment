"""
Fixed-compartment COVID variant subclasses.

Each class is a thin configuration wrapper over CovidJaxModel. The equation
logic is shared; only the schema (and therefore the artifact and COMPARTMENT_LIST)
differs between variants. None of these expose flexible compartment selection.

CovidJaxModel itself is COVID_SEIHDR (the full model).
"""

from compartment.models.example_disease_parameter_uncertainty_model.model import ExampleDiseaseParameterUncertaintyModel