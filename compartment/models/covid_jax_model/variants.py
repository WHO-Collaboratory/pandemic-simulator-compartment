"""
Fixed-compartment COVID variant subclasses.

Each class is a thin configuration wrapper over CovidJaxModel. The derivative
logic is shared; only the schema (and therefore the artifact and COMPARTMENT_LIST)
differs between variants. None of these expose flexible compartment selection.

CovidJaxModel itself is COVID_SEIHDR (the full model).
"""

from compartment.models.covid_jax_model.model import CovidJaxModel

# Parameters for the S→I beta edge used in variants without an E compartment.
# Mirrors the S→E beta declaration in CovidJaxModel.define_parameters().
_BETA_SI = dict(
    source="susceptible",
    target="infected",
    variable_name="beta",
    frequency_dependent=True,
    label="Transmission Rate (S->I)",
    description="Rate at which susceptible individuals become infected through contact with infected individuals",
    default=0.25,
    min_value=0.01,
    max_value=2.0,
    default_min=0.2,
    default_max=0.3,
    unit="per day",
)


class CovidSEIRModel(CovidJaxModel):

    DISEASE_TYPE = "COVID_SEIR"
    DISEASE_LABEL = "Novel Respiratory (SEIR)"
    DISEASE_DESCRIPTION = "An SEIR compartmental model for novel respiratory diseases with age-stratified transmission"

    @classmethod
    def define_parameters(cls, schema):
        super().define_parameters(schema)
        schema.remove_compartment("H")  # removes zeta(I→H), epsilon(H→D), eta(H→R)
        schema.remove_compartment("D")  # removes delta(I→D)


class CovidSIHRModel(CovidJaxModel):

    DISEASE_TYPE = "COVID_SIHR"
    DISEASE_LABEL = "Novel Respiratory (SIHR)"
    DISEASE_DESCRIPTION = "An SIHR compartmental model for novel respiratory diseases with age-stratified transmission"

    @classmethod
    def define_parameters(cls, schema):
        super().define_parameters(schema)
        schema.remove_compartment("E")  # removes beta(S→E), theta(E→I)
        schema.remove_compartment("D")  # removes delta(I→D), epsilon(H→D)
        schema.add_transmission_edge(**_BETA_SI)


class CovidSIDRModel(CovidJaxModel):

    DISEASE_TYPE = "COVID_SIDR"
    DISEASE_LABEL = "Novel Respiratory (SIDR)"
    DISEASE_DESCRIPTION = "An SIDR compartmental model for novel respiratory diseases with age-stratified transmission"

    @classmethod
    def define_parameters(cls, schema):
        super().define_parameters(schema)
        schema.remove_compartment("E")  # removes beta(S→E), theta(E→I)
        schema.remove_compartment("H")  # removes zeta(I→H), epsilon(H→D), eta(H→R)
        schema.add_transmission_edge(**_BETA_SI)


class CovidSEIHRModel(CovidJaxModel):

    DISEASE_TYPE = "COVID_SEIHR"
    DISEASE_LABEL = "Novel Respiratory (SEIHR)"
    DISEASE_DESCRIPTION = "An SEIHR compartmental model for novel respiratory diseases with age-stratified transmission"

    @classmethod
    def define_parameters(cls, schema):
        super().define_parameters(schema)
        schema.remove_compartment("D")  # removes delta(I→D), epsilon(H→D)


class CovidSEIDRModel(CovidJaxModel):

    DISEASE_TYPE = "COVID_SEIDR"
    DISEASE_LABEL = "Novel Respiratory (SEIDR)"
    DISEASE_DESCRIPTION = "An SEIDR compartmental model for novel respiratory diseases with age-stratified transmission"

    @classmethod
    def define_parameters(cls, schema):
        super().define_parameters(schema)
        schema.remove_compartment("H")  # removes zeta(I→H), epsilon(H→D), eta(H→R)


class CovidSIHDRModel(CovidJaxModel):

    DISEASE_TYPE = "COVID_SIHDR"
    DISEASE_LABEL = "Novel Respiratory (SIHDR)"
    DISEASE_DESCRIPTION = "An SIHDR compartmental model for novel respiratory diseases with age-stratified transmission"

    @classmethod
    def define_parameters(cls, schema):
        super().define_parameters(schema)
        schema.remove_compartment("E")  # removes beta(S→E), theta(E→I)
        schema.add_transmission_edge(**_BETA_SI)


class CovidSIRModel(CovidJaxModel):

    DISEASE_TYPE = "COVID_SIR"
    DISEASE_LABEL = "Novel Respiratory (SIR)"
    DISEASE_DESCRIPTION = "An SIR compartmental model for novel respiratory diseases with age-stratified transmission"

    @classmethod
    def define_parameters(cls, schema):
        super().define_parameters(schema)
        schema.remove_compartment("E")  # removes beta(S→E), theta(E→I)
        schema.remove_compartment("H")  # removes zeta(I→H), epsilon(H→D), eta(H→R)
        schema.remove_compartment("D")  # removes delta(I→D)
        schema.add_transmission_edge(**_BETA_SI)
