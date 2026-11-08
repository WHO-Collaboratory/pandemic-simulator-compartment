import logging

import jax
import jax.numpy as jnp
import numpy as np

from compartment.model import Model, ValueType

logger = logging.getLogger(__name__)

# Options of the zone field "initial_water_mode" (how B(0) is set).
WATER_BALANCED = "Balanced with starting cases"
WATER_AS_ENTERED = "As entered"


def _keep_intervention_fields(schema, intervention_id, fields):
    """Show only ``fields`` for one intervention, relabelled.

    ``fields`` maps a generic field name to (label, description, required) or to
    (label, description, required, overrides), where overrides sets other attributes of
    the field (unit, value_type, default, min_value, max_value). The framework gives every
    intervention the same generic fields; this model uses only some of them, sometimes
    for another quantity, so the others are removed from the form and the kept ones
    relabelled. There is no public API for this yet, so it edits the builder's
    intervention list and leaves the generic fields in place (with a warning) if that
    structure changes.
    """
    defs = getattr(schema, "_interventions", None)
    intv = next((d for d in defs or [] if getattr(d, "id", None) == intervention_id), None)
    if intv is None or not hasattr(intv, "parameters"):
        logger.warning("Could not trim fields of intervention %s; generic fields kept.", intervention_id)
        return
    kept = []
    for param in intv.parameters:
        if param.name in fields:
            spec = fields[param.name]
            param.label, param.description, param.required = spec[:3]
            for attr, value in (spec[3] if len(spec) > 3 else {}).items():
                setattr(param, attr, value)
            kept.append(param)
    intv.parameters = kept


class CholeraSvirbOcvMaityModel(Model):
    """SVIRB cholera model built on Maity et al. (2023), Bull Math Biol 85:55, Eq. 1.

    Lusaka, Zambia, 2017-18. Maity et al. extend Posny et al. (2015) with waning natural
    and vaccine-induced immunity; their only vaccination is a constant rate phi. Added
    here, not in the paper: a reactive OCV campaign, with two extra compartments that keep
    track of vaccinated people after infection (VI) and recovery (VR); the paper's
    constant controls as dated interventions, with sanitation split into a water route
    (rho_e) and a contact route (rho_h); the per-week reading of the fitted rates; the
    fitted N and I(0); and the balanced B(0). u(t) = doses given on day t and T_p = time
    to protection, so u(t - T_p) is the doses taking effect on day t. Per admin zone::

        lambda = (1 - rho_e) beta_e B / (k + B) + (1 - rho_h) beta_h (I + VI)
        dS/dt  = mu N - lambda S - phi S - u(t - T_p) S / (S + R) + alpha V + omega (R + VR) - mu S
        dV/dt  = phi S + u(t - T_p) S / (S + R) - sigma lambda V - (alpha + mu) V
        dI/dt  = lambda S - (tau + gamma + mu) I
        dVI/dt = sigma lambda V - (tau + gamma + mu) VI
        dR/dt  = (tau + gamma) I - u(t - T_p) R / (S + R) - (omega + mu) R
        dVR/dt = (tau + gamma) VI + u(t - T_p) R / (S + R) - (omega + mu) VR
        dB/dt  = xi (I + VI) - (delta + nu) B

    I + VI is the paper's I and R + VR the paper's R, so with phi = 0.05, no OCV
    campaign, rho_e = rho_h and the paper's controls over the whole run this is exactly
    Eq. 1.
    beta_e, beta_h and nu are the published wave-1 fitted means (Table 2, Poisson),
    read per week and entered per day (divided by 7); the sanitation efficacy rho (Table 2) enters through the two sanitation
    interventions; the other values are Table 1, except phi, which defaults to 0 because
    the dated OCV campaign replaces it.

    Interventions (the paper's controls are constant from day 0; here each is dated):

    * **Treatment** (tau): infected people also stop being infectious at
      1 / time to treatment. Acts throughout, also in the control run; dates only
      restrict it.
    * **Safe water & sanitation** (rho_e) and **hygiene & contact** (rho_h): beta_e
      and beta_h x (1 - adherence x transmission reduction). Preset: the paper's
      rho = 16% on both routes from 28 Dec 2017 (the documented strengthening of the
      response), not from day 0 as in the paper.
    * **Bacteria removal** (nu): inside its dates bacteria are also removed from the
      water at the model parameter nu (paper Table 2: 1.8, read per week); outside them only natural
      death delta acts. Preset: the whole run.
    * **Reactive OCV campaign** replaces the paper's constant phi: u(t) doses per day,
      shared between S and R in proportion to their size (V, I, VI, VR are not
      vaccinated); S -> V, R -> VR, after the time to protection. Coverage c, daily
      speed v and dates are the intervention's fields; vaccine efficacy, time to
      protection and duration of protection are model parameters ("OCV vaccine: ...").
      The end date chooses the mode: with an end date, c N doses spread evenly over
      end - start days (the end date itself is not a vaccination day) and the speed is
      ignored; without one, v N doses per day from the start date until c N doses are
      given.

    Cumulative totals (``<X>_total``) count everything that enters compartment X:
    I_total + VI_total are all new cases, VI_total the breakthrough cases among them.

    No mobility: with several admin zones each zone is simulated on its own (own
    population, water and OCV doses); zones share the parameters and intervention dates
    but never infect each other. ``build_travel_matrix()`` is not overridden and the
    travel matrix is not used.
    """

    # One series per compartment. Listing them keeps the hand-declared VI_total and
    # VR_total out of the results view; each delta is then the compartment's cumulative
    # inflow (<X>_total), except B, which is its final concentration.
    COMPARTMENT_DELTA_GROUPING = {c: [c] for c in ("S", "V", "I", "VI", "R", "VR", "B")}

    @classmethod
    def define_parameters(cls, schema):
        """Declare compartments, Table 1-2 parameters and the five interventions.

        Args:
            schema (ParameterSchemaBuilder): Builder that collects the model's info, metadata,
                compartments, parameters, admin-zone fields and interventions.
        """
        schema.set_model_info(
            disease_type="cholera_svirb_ocv_maity",
            label="Cholera – SVIRB, reactive OCV (extends Maity et al. 2023)",
            description=(
                "SVIRB cholera model for estimating the effect of a reactive OCV campaign, safe water, "
                "hygiene, bacteria removal and treatment. Built on Maity et al. (2023, Bull Math Biol): "
                "their two-route transmission structure and published fitted values for Lusaka 2017-18 (Table 2, read per week). "
                "The reactive OCV campaign is not in the paper (which has only a constant vaccination "
                "rate) and was added for this model."
            ),
        )
        schema.set_model_metadata(
            authors=[
                {
                    "name": "Biplab Maity, Bapi Saha, Indrajit Ghosh, Joydev Chattopadhyay (source model)",
                    "affiliation": (
                        "Indian Statistical Institute, Kolkata; Government College of Engineering and "
                        "Textile Technology; University of Georgia"
                    ),
                },
                {
                    "name": "Yingsi Lai (implementation; reactive OCV campaign extension)",
                    "affiliation": "WHO Hub for Pandemic and Epidemic Intelligence",
                },
            ],
            license="Apache-2.0",
            model_type=(
                "Compartmental – SVIRB with a water reservoir (B), deterministic ODE; two transmission "
                "routes; leaky vaccine with breakthrough (VI) and vaccinated-recovered (VR) compartments"
            ),
            diseases=["Cholera"],
            transmission_routes=["Waterborne", "Fecal-oral"],
            questions_answered=[
                "How do reactive OCV campaigns of different timing, speed and coverage compare, under "
                "the paper's transmission and control values?",
                "What if the campaign had started earlier, vaccinated faster or reached more people?",
                "How much do safe water, hygiene and bacteria removal contribute, route by route?",
                "How many cases occur among vaccinated people (breakthrough infections)?",
            ],
            key_assumptions=[
                "Constant population; births equal natural deaths; no cholera deaths.",
                "Treatment acts throughout, as in the paper; keeping it on in the control run is this "
                "model's choice. It ends "
                "infectiousness for every infected person at 1 / time to treatment (default 3 days, "
                "Miller Neilan 2010), although antibiotics go mainly to severe cases, so its effect "
                "is overstated.",
                "Sanitation (default 16% on both routes) and bacteria removal (nu) are interventions "
                "set to the paper's values. Bacteria removal acts over the whole run, as in the paper. "
                "In the preset, safe water and hygiene & contact act from 28 Dec 2017, not from day 0 as "
                "in the paper: the paper's 16% is read as the level after the documented strengthening "
                "of the response in late December (Statutory Instrument No. 79 of 2017 in force from "
                "28 Dec; street-vending ban and market closures from 29-30 Dec; higher chlorination of "
                "the municipal water), with no such reduction before. Without them (as in the control "
                "run) there is no sanitation at all and epidemics are very large.",
                "The reactive OCV campaign is an addition to the paper (which has only a constant "
                "vaccination rate phi). OCV doses go to unvaccinated people without symptoms (S and R) in proportion to "
                "their size; only doses given to S give protection (leaky vaccine; efficacy, time to "
                "protection and duration of protection are model parameters, defaults 67%, 7 days "
                "and 2.5 years).",
                "One homogeneous population per zone: OCV coverage is spread over everyone, although "
                "the 2018 campaign targeted four sub-districts.",
                "Fitted rates are read per week and entered per day (Table 2 value / 7), a reading made "
                "for this model: nu per week is needed to reproduce the ratio of the paper's Table 2 R_e "
                "and R_h. For beta_e and beta_h the case data cannot decide (N absorbs the factor 7); "
                "read per day, with N and I(0) refitted and the same 98.5% coverage, they fit as well "
                "(RMSE 80.3 vs 81.3) but need N of about 193,000, fewer than the 1,319,276 people "
                "vaccinated, so they are read per week too, although the paper's R0 (at phi = 0.05 and "
                "an unreported N) would need N of about 3.9 M with the per-week reading and 0.56 M "
                "with the per-day one, and so points the other way.",
                "Time to protection (7 days) follows Pasetto 2018; other studies differ (Azman 2015: "
                "immediate; Chao 2011: 50% of maximum efficacy at 10 days). A zero delay is not "
                "plausible, although with it the fit improves (RMSE 60.4 vs 81.3, N and I(0) refitted), "
                "because it lets the campaign stand in for the decline that began before protection.",
                "Data dates: the 15 weekly counts are Sunday-Saturday weeks ending 28 Oct 2017 ... 3 Feb "
                "2018 (the labels of the paper's Fig. 2a are week-ending dates; checked against WHO AFRO "
                "situation reports), so day 0 is 22 Oct 2017.",
                "B(0) follows I(0) by default (zone field 'Initial V. cholerae in water' = 'Balanced "
                "with starting cases': xi I(0) / (delta + nu)); 'As entered' uses the zone's value.",
            ],
            applicability=(
                "Urban cholera outbreaks with both waterborne and person-to-person spread, in one "
                "well-mixed population (or several independent zones), over one epidemic wave; "
                "comparing reactive OCV campaigns (timing, speed, coverage, vaccine) against a "
                "background of water, hygiene and treatment measures. Parameter defaults are fitted "
                "to Lusaka 2017-18 and should be re-examined for other settings."
            ),
            constraints=(
                "Fitted values come from one wave in one city (15 weekly counts); N and I(0) are not "
                "reported in the paper and were fitted here; B(0) is not observed. One OCV campaign "
                "per run (the framework keeps one intervention per type). Interventions are switched "
                "by dates only, not by case thresholds; an end date before the start date makes an "
                "intervention inactive. No movement between zones. Uncertainty runs over the "
                "parameter ranges are a near-threshold sensitivity check (cases vary from hundreds to "
                "about 100,000), not the uncertainty of the fit."
            ),
            biases=(
                "All infections are compared with reported cases (no reporting fraction), so true "
                "infections may be underestimated. Spreading the campaign over the whole population "
                "rather than the four targeted sub-districts dilutes its local effect; the fitted N "
                "gives a coverage of 98.5%, which only a homogeneous model needs. The decline still comes "
                "about a week late: reported cases fell from 642 (7-13 Jan) to 340 (14-20 Jan), although "
                "protection only began on Wednesday 17 Jan and built up until 26 Jan (about 10% of the "
                "doses a day), while the model stays at 518 that week. The "
                "start date and size of the sanitation step are assumptions, and they decide how much "
                "of the decline the OCV campaign explains: cases averted by the campaign by 4 Feb 2018 "
                "are about 560 to 1,440 under the alternatives in model.md (preset: about 1,020), and "
                "more if the run is longer, since without OCV the epidemic is still rising on 4 Feb; "
                "the effect of "
                "the 2018 campaign is not identified by the data. Treating every case at the "
                "same rate overstates the effect of treatment. Removing bacteria removal (on from day 0) "
                "gives a very large epidemic because transmission was fitted with it present."
            ),
            citations=[
                "Maity B, Saha B, Ghosh I, Chattopadhyay J (2023) Model-based estimation of expected "
                "time to cholera extinction in Lusaka, Zambia. Bull Math Biol 85:55. "
                "doi:10.1007/s11538-023-01149-0",
                "Posny D, Wang J, Mukandavire Z, Modnak C (2015) Analyzing transmission dynamics of "
                "cholera with public health interventions. Math Biosci 264:38-53.",
                "Miller Neilan RL, Schaefer E, Gaff H, Fister KR, Lenhart S (2010) Modeling optimal "
                "intervention strategies for cholera. Bull Math Biol 72(8):2004-2018.",
                "Azman AS, Luquero FJ, Ciglenecki I, Grais RF, Sack DA, Lessler J (2015) The impact of "
                "a one-dose versus two-dose oral cholera vaccine regimen in outbreak settings: a "
                "modeling study. PLoS Med 12(8):e1001867.",
                "Pasetto D, Finger F, Camacho A, et al. (2018) Near real-time forecasting for cholera "
                "decision making in Haiti after Hurricane Matthew. PLoS Comput Biol 14(5):e1006127.",
                "Lee EC, Chao DL, Lemaitre JC, et al. (2020) Achieving coordinated national immunity and "
                "cholera elimination in Haiti through vaccination: a modelling study. Lancet Glob Health "
                "8(8):e1081-e1089.",
                "Chao DL, Halloran ME, Longini IM Jr (2011) Vaccination strategies for epidemic cholera "
                "in Haiti with implications for the developing world. PNAS 108(17):7081-7085.",
                "Sardar T, Mukhopadhyay S, Bhowmick AR, Chattopadhyay J (2013) An optimal cost "
                "effectiveness study on Zimbabwe cholera seasonal data from 2008-2011. PLoS ONE "
                "8(12):e81231.",
                "Hartley DM, Morris JG Jr, Smith DL (2006) Hyperinfectivity: a critical element in the "
                "ability of V. cholerae to cause epidemics? PLoS Med 3(1):e7.",
                "Mukandavire Z, Liao S, Wang J, Gaff H, Smith DL, Morris JG Jr (2011) Estimating the "
                "reproductive numbers for the 2008-2009 cholera outbreaks in Zimbabwe. PNAS "
                "108(21):8767-8772.",
                "Ferreras E, et al. (2020) Alternative observational designs to estimate the "
                "effectiveness of one dose of oral cholera vaccine in Lusaka, Zambia. Epidemiol Infect "
                "148:e78. doi:10.1017/S095026882000062X",
                "GTFCC Case Management Working Group (2018) Technical note: use of antibiotics for the "
                "treatment and control of cholera.",
                "WHO AFRO (2018) Zambia cholera situation report No. 114, 2 February 2018 (OCV round 1 "
                "closed 20 Jan after 10 days: 1,319,276 people in Chawama, Kanyama, Matero, Chipata and "
                "Lusaka Central Prison; weekly cases 14-20 and 21-27 Jan).",
                "WHO AFRO (2018) WHO supports the immunization of 1 million people against cholera in "
                "Zambia, news release, 10 January 2018 (campaign launched 10 Jan).",
                "WHO AFRO (2018) Zambia cholera situation report No. 115, 3 February 2018 (weekly cases "
                "28 Jan-3 Feb).",
                "WHO (2017) Cholera - Zambia. Disease Outbreak News, 11 December 2017.",
                "Sinyange N, Brunkard JM, Kapata N, et al. (2018) Cholera epidemic - Lusaka, Zambia, "
                "October 2017-May 2018. MMWR Morb Mortal Wkly Rep 67(19):556-559.",
                "Kapata N, Sinyange N, Mazaba ML, et al. (2018) A multisectoral emergency response "
                "approach to a cholera outbreak in Zambia: October 2017-February 2018. J Infect Dis "
                "218(Suppl 3):S181-S183. doi:10.1093/infdis/jiy490",
                "Mazaba ML (2018) Hopeful: towards the tail end of the Zambia cholera 2017/2018 "
                "outbreak? (Editorial). Health Press Zambia Bull 2(5):1-3.",
                "ReliefWeb (2018) Zambia: Cholera Outbreak - Oct 2017 (EP-2017-000178-ZMB), WHO update "
                "of 26 January 2018.",
                "Daily Nation (Zambia) (2017) Lungu bans street vending, closes Soweto market. "
                "30 December 2017.",
                "Reuters (Mfula C) (2018) Zambia declares curfew in Lusaka slum struck by cholera. "
                "January 2018 (curfew in Kanyama from Sunday 7 January)",
                "Republic of Zambia (2017) Public Health (Infected Areas) (Cholera) Regulations, 2017. "
                "Statutory Instrument No. 79 of 2017 (commenced 28 December 2017).",
            ],
            validation=(
                "Equations verified against a SciPy solution of the same equations and against the "
                "paper's Eq. 1. Wave 1, 15 weeks ending 28 Oct 2017 ... 3 Feb 2018 (day 0 = 22 Oct 2017): "
                "paper wave-1 values (Table 2 read per week, / 7), treatment and bacteria removal over "
                "the whole run, sanitation (16% on both routes) from 28 Dec 2017, OCV round 1 (1,319,276 "
                "doses spread evenly over 10 days from 10 Jan 2018, 7-day delay); only N and I(0) "
                "fitted: RMSE 81.3 cases/week (paper's own fit, with constant phi = 0.05: 65.8; with "
                "sanitation from day 0: 118.4). The model peaks in the week ending 13 Jan, one week "
                "after the reported peak. With the paper's own "
                "assumption (phi = 0.05/day from day 0, no campaign, the paper's controls from day 0), "
                "the Table 2 values fixed and N and I(0) refitted, the best RMSE is about 137."
            ),
            not_for=(
                "Estimating the cases averted by the 2018 Lusaka campaign (the preset does not "
                "reproduce the timing of the January decline); reproducing Maity et al. Fig. 2 exactly "
                "(unreported N and initial conditions); the "
                "2018 second wave (flooding from mid-February per Maity 2023, mid-March per MMWR 2018), "
                "which is not modelled; long-term "
                "endemic projections or the paper's stochastic time-to-extinction analysis; settings "
                "where spread between connected zones matters (use a mobility model)."
            ),
        )

        # --- Compartments ---
        schema.add_compartment("S", "Susceptible", "Unvaccinated susceptible humans")
        schema.add_compartment("V", "Vaccinated", "Vaccinated, not infected (partially protected, leaky)")
        schema.add_compartment("I", "Infected", "Unvaccinated infected humans (cases)", infective=True)
        schema.add_compartment("VI", "Infected (vaccinated)", "Vaccinated infected humans (breakthrough cases)", infective=True)
        schema.add_compartment("R", "Recovered", "Unvaccinated recovered humans (temporarily immune)")
        schema.add_compartment("VR", "Recovered (vaccinated)", "Vaccinated recovered or immune humans")
        schema.add_compartment("B", "Bacteria", "V. cholerae concentration in water (cells/mL)")
        # Cumulative inflows for the two compartments that are not edge targets (the
        # framework adds S_total, V_total, I_total and R_total itself). Flows are written
        # by hand in equation(), so these are declared by hand.
        schema.add_compartment("VI_total", "Breakthrough cases total", "Cumulative new infections of vaccinated people (V -> VI)")
        schema.add_compartment("VR_total", "Recovered (vaccinated) total", "Cumulative entries to VR (recoveries of VI and doses to R)")

        # --- Transmission edges (UI arrows) ---
        schema.add_transmission_parameter(
            source="susceptible", target="infected", variable_name="beta_h",
            label="Human-to-human transmission rate (beta_h)",
            description=(
                "Infection per infected person (I + VI), per day. Table 2 (Poisson, wave 1): 1.4e-6 "
                "(95% CI 1.28e-6 to 1.5e-6), read per week, = 2.0e-7 per person per day (= 1.4e-6 / 7). Uncertainty range "
                "1.82e-7 to 2.15e-7 (CI / 7, rounded outward)"
            ),
            default=2.0e-7, default_min=1.82e-7, default_max=2.15e-7, min_value=0.0, max_value=1e-3,
            unit="per person per day",
        )
        schema.add_transmission_parameter(
            source="susceptible", target="vaccinated", variable_name="phi",
            label="Background vaccination rate (phi)",
            description=(
                "Constant vaccination of susceptibles every day (a rate, not an initial share). Paper "
                "(Table 1): 0.05 per day from day 0. Default 0: Lusaka's 2018 vaccination was a dated "
                "campaign (OCV intervention); people vaccinated before the outbreak go in 'Initially "
                "vaccinated'."
            ),
            default=0.0, min_value=0.0, max_value=1.0, unit="per day",
        )
        schema.add_transmission_parameter(
            source="vaccinated", target="susceptible", variable_name="alpha",
            label="OCV vaccine: duration of protection (1/alpha)",
            description=(
                "V -> S. Table 1: 2.5 years. Set together with 'OCV vaccine: efficacy' and 'OCV "
                "vaccine: time to protection' when modelling another vaccine."
            ),
            default=912.5, value_type=ValueType.DAYS, unit="days", min_value=30.0, max_value=7300.0,
        )
        schema.add_transmission_parameter(
            source="infected", target="recovered", variable_name="gamma",
            label="Natural recovery period (1/gamma)", description="Table 1: 5 days (also VI -> VR)",
            default=5.0, value_type=ValueType.DAYS, unit="days", min_value=1.0, max_value=30.0,
        )
        schema.add_transmission_parameter(
            source="recovered", target="susceptible", variable_name="omega",
            label="Duration of natural immunity (1/omega)", description="Table 1: 2 years (R and VR -> S)",
            default=730.0, value_type=ValueType.DAYS, unit="days", min_value=30.0, max_value=7300.0,
        )

        # --- Fitted values (Table 2, Poisson error structure, wave 1) ---
        schema.add_parameter(
            name="beta_e", label="Environment-to-human transmission rate (beta_e)",
            description=(
                "Maximum infection rate from water (B >> k), per day. Table 2: 0.058 (95% CI "
                "0.052-0.071), read per week, = 0.0082857 per day (= 0.058 / 7). Uncertainty range 0.0074-0.0102 per day (CI / 7, rounded outward)."
            ),
            value_type=ValueType.FLOAT, default=0.0082857, default_min=0.0074, default_max=0.0102,
            min_value=0.0, max_value=10.0, unit="per day",
        )
        schema.add_parameter(
            name="nu", label="Bacteria removal: removal rate of V. cholerae (nu)",
            description=(
                "Removal of bacteria from water while the 'Bacteria removal' intervention is active, on "
                "top of natural death, per day. Table 2: 1.8 (95% CI 1.6-2.1), read per week, = 0.257143 "
                "per day (= 1.8 / 7). Uncertainty range 0.23-0.3 per day (the paper's 95% CI 1.6-2.1, "
                "/ 7 = 0.229-0.3, rounded); with it bacteria last 1/(delta + nu) = 3.4 days instead of 30."
            ),
            value_type=ValueType.FLOAT, default=0.257143, default_min=0.23, default_max=0.3,
            min_value=0.0, max_value=50.0, unit="per day",
        )

        # --- Other Table 1 values ---
        schema.add_parameter(
            name="treatment_days", label="Treatment: time to treatment (1/tau)",
            description=(
                "Average days until an infected person is treated and stops being infectious. Table 1: "
                "tau = 1/3 per day (Miller Neilan et al. 2010). With natural recovery (5 days) a case "
                "is infectious ~1.9 days on average. Uncertainty range: tau +/-20%, as the paper's "
                "sensitivity analysis uses for parameters it did not fit (2.5-3.75 days)."
            ),
            value_type=ValueType.FLOAT, default=3.0, default_min=2.5, default_max=3.75,
            min_value=0.1, max_value=60.0, unit="days",
        )
        schema.add_parameter(
            name="xi", label="Shedding rate of V. cholerae (xi)",
            description="Table 1: 10 cells/mL per day per infected person (I and VI)",
            value_type=ValueType.FLOAT, default=10.0, min_value=0.0, max_value=1e4, unit="cells/mL/day/person",
        )
        schema.add_parameter(
            name="bacteria_lifetime", label="Natural lifetime of V. cholerae in water (1/delta)",
            description="Table 1: delta = 1/30 per day",
            value_type=ValueType.FLOAT, default=30.0, min_value=0.1, max_value=365.0, unit="days",
        )
        schema.add_parameter(
            name="half_saturation", label="Half-saturation constant (k)", description="Table 1: 1e6 cells/mL",
            value_type=ValueType.FLOAT, default=1e6, min_value=1.0, max_value=1e10, unit="cells/mL",
        )
        schema.add_parameter(
            name="life_expectancy", label="Life expectancy (1/mu)", description="Table 1: 43.5 years",
            value_type=ValueType.FLOAT, default=43.5, min_value=1.0, max_value=120.0, unit="years",
        )

        # --- OCV vaccine (with 'OCV vaccine: duration of protection', the V -> S arrow) ---
        # Model parameters, not intervention fields: the intervention form has only generic
        # fields, and the framework reads its threshold fields as prevalence thresholds.
        schema.add_parameter(
            name="vaccine_efficacy", label="OCV vaccine: efficacy",
            description=(
                "Leaky vaccine: vaccinated people are infected at (1 - efficacy) x the unvaccinated "
                "rate. Paper (Table 1): 67%. Lusaka 2016, one dose: 88.9% (95% CI 42.7-97.8, matched case-control; "
                "Ferreras et al. 2020)."
            ),
            value_type=ValueType.FLOAT, default=67.0, min_value=0.0, max_value=100.0, unit="%",
        )
        schema.add_parameter(
            name="time_to_protection", label="OCV vaccine: time to protection",
            description=(
                "Days from a dose until the person moves to V or VR; until then they are infected at "
                "the unvaccinated rate. Assumption: 7 days (Pasetto et al. 2018: immunity one week "
                "after administration)."
            ),
            value_type=ValueType.FLOAT, default=7.0, min_value=0.0, max_value=30.0, unit="days",
        )

        # --- Initial state not reported in the paper ---
        schema.add_admin_zone_field(
            name="vaccinated_population", label="Initially vaccinated",
            description="Share of the non-infected population in V at the start: V(0) = share x (N - I(0)) (not reported in the paper)",
            value_type=ValueType.PERCENTAGE, default=0.0, min_value=0.0, max_value=100.0, unit="%",
        )
        schema.add_admin_zone_field(
            name="initial_water_mode", label="Initial V. cholerae in water",
            description=(
                "How B(0) is set in this zone. 'Balanced with starting cases' (default): computed from "
                "the zone's starting cases, B(0) = xi I(0) / (delta + nu), about 34 x the starting cases "
                "with the defaults, so it follows I(0). 'As entered': the value of 'Initial V. cholerae "
                "concentration' is used (e.g. 0 = clean water at the start)."
            ),
            value_type=ValueType.SELECT, default=WATER_BALANCED,
            options=[WATER_BALANCED, WATER_AS_ENTERED],
        )
        schema.add_admin_zone_field(
            name="bacteria_concentration", label="Initial V. cholerae concentration",
            description=(
                "B at the start, used only when 'Initial V. cholerae in water' is 'As entered'. With "
                "'Balanced with starting cases' (default) it is computed from the starting cases "
                "instead (preset: 117.8 for I(0) = 3.4)."
            ),
            value_type=ValueType.FLOAT, default=0.0, min_value=0.0, max_value=1e9, unit="cells/mL",
        )

        # --- Interventions ---
        schema.add_intervention(
            id="treatment", label="Treatment (case management)",
            description=(
                "Rehydration, antibiotics and isolation: infected people stop being infectious at "
                "1 / 'Treatment: time to treatment' on top of natural recovery. As in the paper it acts "
                "throughout the run; it also stays on in the control run (this model's choice). Set dates "
                "only to restrict it."
            ),
            target_rates=[],
        )
        schema.add_intervention(
            id="water_measures", label="Safe water & sanitation (environment-to-human)",
            description=(
                "Clean water supply: chlorination, emergency water tanks, household water treatment. "
                "beta_e x (1 - Adherence x Transmission reduction) = the paper's sanitation efficacy rho "
                "on this route. Paper: 100% x 16% (Table 2: rho = 0.16, 95% CI 0.13-0.22) from day 0; "
                "preset: the same from 28 Dec 2017, standing for the measures of late December to early "
                "January (higher chlorination from late December; emergency water tanks from "
                "December)."
            ),
            target_rates=[], adherence=100.0, transmission_reduction=16.0,   # beta_e is not an edge
        )
        schema.add_intervention(
            id="contact_measures", label="Hygiene & contact measures (human-to-human)",
            description=(
                "Hand hygiene, food safety, safe funerals. beta_h x (1 - Adherence x Transmission "
                "reduction) = the paper's rho on this route. Paper: 100% x 16% from day 0; preset: the "
                "same from 28 Dec 2017, standing for the measures of late December to early January "
                "(SI No. 79 from 28 Dec, banning gatherings of more than five people; street-vending "
                "ban and market closures from 29 Dec; Kanyama curfew from 7 Jan)."
            ),
            target_rates=[], adherence=100.0, transmission_reduction=16.0,   # applied in equation()
        )
        schema.add_intervention(
            id="bacteria_removal", label="Bacteria removal from water (sanitation, nu)",
            description=(
                "Disinfection and cleaning of water sources: inside the dates bacteria are removed at "
                "'Bacteria removal: removal rate' (paper Table 2: 1.8, read per week) on top of natural death, so they "
                "last ~3.4 days instead of 30. Paper / preset: whole run. Without it (as in the control "
                "run) there is no removal at all and epidemics are very large."
            ),
            target_rates=[],
        )
        schema.add_intervention(
            id="ocv_campaign", label="Reactive OCV campaign",
            description=(
                "Two ways to give doses, chosen by the end date. End date filled: target coverage x "
                "population doses spread evenly over (end date - start date) days, the end date itself "
                "not being a vaccination day, and the daily speed is ignored. End date empty: daily "
                "speed x population doses per day from the start date until the target coverage is "
                "reached. Doses go to unvaccinated people without symptoms (S and R) in proportion to "
                "their size. Lusaka round 1: 10-20 Jan 2018, 1,319,276 people = 98.4892% of the fitted N. The vaccine "
                "itself (efficacy, time to protection, duration of protection) is set in the model "
                "parameters 'OCV vaccine: ...'."
            ),
            target_rates=[], adherence=98.4892, transmission_reduction=9.84892,
        )

        _keep_intervention_fields(schema, "treatment", {
            "start_date": ("Treatment start date", "First day of case management (empty = start of the run)", False),
            "end_date": ("Treatment end date", "Last day of case management (empty = end of the run)", False),
        })
        # Adherence and transmission reduction are required: left empty, the framework would
        # use 0 and 5% instead of the paper's 100% x 16%.
        _keep_intervention_fields(schema, "water_measures", {
            "adherence_min": ("Adherence", "Share of the population reached (safe water, treated sources). Paper: 100%", True),
            "transmission_percentage": ("Transmission reduction", "Reduction in environment-to-human transmission (beta_e) among them. Paper: 16%", True),
            "start_date": ("Start date", "First day of the measures (paper: start of the wave; Lusaka preset: 28 Dec 2017)", False),
            "end_date": ("End date", "Last day of the measures", False),
        })
        _keep_intervention_fields(schema, "contact_measures", {
            "adherence_min": ("Adherence", "Share of the population complying. Paper: 100%", True),
            "transmission_percentage": ("Transmission reduction", "Reduction in human-to-human transmission (beta_h) among them. Paper: 16%", True),
            "start_date": ("Start date", "First day of the measures (paper: start of the wave; Lusaka preset: 28 Dec 2017)", False),
            "end_date": ("End date", "Last day of the measures", False),
        })
        _keep_intervention_fields(schema, "bacteria_removal", {
            "start_date": ("Removal start date", "First day of bacteria removal (empty = start of the run)", False),
            "end_date": ("Removal end date", "Last day of bacteria removal (empty = end of the run)", False),
        })
        _keep_intervention_fields(schema, "ocv_campaign", {
            "adherence_min": ("Target coverage", "Doses given by the end of the campaign as a share of the "
                              "population (max 99%). Lusaka round 1: 1,319,276 / 1,339,513 = 98.4892%", True,
                              {"max_value": 99.0}),
            "transmission_percentage": ("Daily vaccination speed (used only if no end date)",
                                        "Doses per day as a share of the population. Ignored when an end date "
                                        "is filled. With no end date the campaign runs at this speed and stops "
                                        "at the target coverage; if left empty the platform uses 5%. "
                                        "Lusaka: 131,928 per day = 9.84892%", False),
            "start_date": ("Vaccination start date", "First day of the campaign (Lusaka round 1: 10 Jan 2018)", True),
            "end_date": ("Vaccination end date (if filled, daily speed is ignored)",
                         "Filled: doses are spread evenly over (end date - start date) days to reach the target "
                         "coverage; the end date itself is not a vaccination day (to include 20 Jan, enter "
                         "21 Jan), and the daily speed is ignored. Empty: vaccinate at the daily speed until "
                         "the target coverage. Lusaka round 1 preset: 20 Jan 2018 (10 days)", False),
        })

    # ------------------------------------------------------------------
    # Initial state
    # ------------------------------------------------------------------
    @classmethod
    def get_initial_population(cls, admin_zones, compartment_list, **kwargs):
        """Seed S, V, I and B from the admin-zone fields (VI, R, VR start at 0).

        B is the entered 'Initial V. cholerae concentration'; for zones whose
        'Initial V. cholerae in water' is 'Balanced with starting cases' it is replaced in
        prepare_initial_state(), which has the model parameters the balance needs.

        Args:
            admin_zones (list[dict]): Zones with population, infected_population (%),
                vaccinated_population (%), initial_water_mode and bacteria_concentration
                (cells/mL).
            compartment_list (list[str]): Compartment ids in state order.
            **kwargs: Passed by the framework; not used.

        Returns:
            np.ndarray: Initial population, shape (zones, compartments).
        """
        col = {c: i for i, c in enumerate(compartment_list)}
        out = np.zeros((len(admin_zones), len(compartment_list)))
        for z, zone in enumerate(admin_zones):
            n = float(zone["population"])
            infected = zone["infected_population"] / 100.0 * n
            vaccinated = float(zone.get("vaccinated_population", 0.0) or 0.0) / 100.0 * (n - infected)
            out[z, col["I"]] = infected
            out[z, col["V"]] = vaccinated
            out[z, col["S"]] = n - infected - vaccinated
            out[z, col["B"]] = float(zone.get("bacteria_concentration", 0.0) or 0.0)
        return out

    def prepare_initial_state(self):
        """Return the (compartments x zones) initial state.

        In each zone whose 'Initial V. cholerae in water' is 'Balanced with starting cases'
        (the default), B(0) is set to xi (I(0) + VI(0)) / (delta + nu), with the model
        parameter nu, so that shedding by the starting cases equals natural death plus
        removal. It does not depend on which interventions a run contains, so scenarios
        start from the same state. With 'As entered', the zone's 'Initial V. cholerae
        concentration' is used.

        Returns:
            jnp.ndarray: Population matrix used as the solver's initial state.
        """
        pm = jnp.asarray(self.population_matrix)
        config = getattr(self, "config", None)
        case_file = config.get("case_file") if hasattr(config, "get") else None
        zones = (case_file or {}).get("admin_zones") or []
        modes = [
            zone.get("initial_water_mode") if hasattr(zone, "get") else getattr(zone, "initial_water_mode", None)
            for zone in zones
        ]
        for mode in modes:
            if mode not in (None, WATER_BALANCED, WATER_AS_ENTERED):
                logger.warning("Unknown initial_water_mode %r; using %r.", mode, WATER_BALANCED)
        balanced_zone = [mode != WATER_AS_ENTERED for mode in modes] or [True] * pm.shape[-1]
        idx = {c: i for i, c in enumerate(self.compartment_list)}
        removal = 1.0 / self.bacteria_lifetime + self.nu
        balanced = self.xi * (pm[idx["I"]] + pm[idx["VI"]]) / removal
        return pm.at[idx["B"]].set(jnp.where(jnp.asarray(balanced_zone), balanced, pm[idx["B"]]))

    # ------------------------------------------------------------------
    # Interventions
    # ------------------------------------------------------------------
    def _intervention(self, intervention_id):
        """The configured intervention, or None (control run / not configured)."""
        if intervention_id not in self.intervention_dict:
            return None
        return next((i for i in self.interventions if i.id == intervention_id), None)

    def _window(self, t, intervention_id, absent=0.0):
        """1 inside an intervention's date window, 0 outside (whole run if no dates).

        ``absent`` when the intervention is not configured (control run). Window
        edges are logistic with a 0.5-day scale (most of the change within about +/- 1
        day; totals are kept) so the adaptive solver does not stall;
        a start on or before day 0 or an end at or after the run end is open. An end date
        before the start date gives an empty window (never active, with a warning).
        """
        intv = self._intervention(intervention_id)
        if intv is None:
            return absent
        if (intv.start_date_ordinal is not None and intv.end_date_ordinal is not None
                and intv.end_date_ordinal < intv.start_date_ordinal):
            warned = self.__dict__.setdefault("_reversed_warned", set())
            if intervention_id not in warned:
                warned.add(intervention_id)
                logger.warning("%s: end date before start date; the intervention is never active.",
                               intervention_id)
            return 0.0                                                     # empty window
        start = 0 if intv.start_date_ordinal is None else intv.start_date_ordinal - self.start_date_ordinal
        end = self.n_timesteps + 1 if intv.end_date_ordinal is None else intv.end_date_ordinal - self.start_date_ordinal + 1
        on = 1.0 if start <= 0 else jax.nn.sigmoid((t - start) / 0.5)
        off = 0.0 if end > self.n_timesteps else jax.nn.sigmoid((t - end) / 0.5)
        return on - off

    def _level(self, t, intervention_id):
        """Adherence x transmission reduction inside the window (rho_e, rho_h)."""
        intv = self._intervention(intervention_id)
        if intv is None:
            return 0.0
        return intv.adherence * intv.transmission_reduction * self._window(t, intervention_id)

    def _ocv_doses(self, t, N):
        """Doses taking effect on day t, u(t - T_p), of the reactive OCV campaign.

        0 in the control run. u(t) is the doses given on day t; T_p is the time to
        protection.

        The campaign is set in the intervention form: target coverage c (the
        "Target coverage" field), daily speed v ("Daily vaccination speed") and dates.
        With an end date: c N doses spread evenly over end - start days (the end date
        is not a vaccination day). Without one: v N doses per day from the start date
        until c N doses are given (c / v days). Both are shifted by the time to
        protection, since a dose moves its recipient to V or VR only then.

        Args:
            t (float): Days since the simulation start date.
            N (jnp.ndarray): Population of each zone.

        Returns:
            jnp.ndarray | float: Doses per day in each zone at time t.
        """
        intv = self._intervention("ocv_campaign")
        if intv is None:
            return 0.0
        if intv.start_date_ordinal is None:
            logger.warning("ocv_campaign needs a start date; ignoring it.")
            return 0.0
        start = intv.start_date_ordinal - self.start_date_ordinal + float(self.time_to_protection)
        cov = min(max(float(intv.adherence), 0.0), 0.99)
        if intv.end_date_ordinal is not None:
            days = float(intv.end_date_ordinal - intv.start_date_ordinal)
            if days < 1.0:
                logger.warning("ocv_campaign end date is not after its start date; ignoring it.")
                return 0.0
            per_day = cov / days
        else:
            per_day = float(intv.transmission_reduction)
            if per_day <= 0.0:
                logger.warning("ocv_campaign without an end date needs a daily vaccination speed; ignoring it.")
                return 0.0
            days = cov / per_day
        window = jax.nn.sigmoid((t - start) / 0.5) - jax.nn.sigmoid((t - start - days) / 0.5)
        return per_day * N * window

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------
    def equation(self, y, t, p):
        """Compute the derivatives for every zone.

        Args:
            y (jnp.ndarray): Compartment values ordered by ``compartment_list``, zones last.
            t (float): Days since the simulation start date.
            p (tuple): Packed transmission-edge parameters.

        Returns:
            jnp.ndarray: Stacked per-compartment derivatives (dy/dt).
        """
        params = self._unpack_params(p)
        st = {c: y[i] for i, c in enumerate(self.compartment_list)}
        S, V, I, VI, R, VR, B = (st[k] for k in ("S", "V", "I", "VI", "R", "VR", "B"))
        N = S + V + I + VI + R + VR

        beta_e = self.beta_e * (1.0 - self._level(t, "water_measures"))        # (1 - rho_e) beta_e
        beta_h = params["beta_h"] * (1.0 - self._level(t, "contact_measures"))  # (1 - rho_h) beta_h
        nu = self.nu * self._window(t, "bacteria_removal")                    # 0 outside / control
        tau = self._window(t, "treatment", absent=1.0) / self.treatment_days        # on in the control run
        phi = params["phi"]

        alpha, gamma, omega = params["alpha"], params["gamma"], params["omega"]
        mu = 1.0 / (self.life_expectancy * 365.0)
        sigma = 1.0 - float(self.vaccine_efficacy) / 100.0
        xi, delta, k = self.xi, 1.0 / self.bacteria_lifetime, self.half_saturation

        # OCV: u(t - T_p) doses taking effect, shared between S and R; per-capita rate capped at 5/day.
        eligible = jnp.maximum(S + R, 1e-9)
        per_capita = jnp.minimum(self._ocv_doses(t, N) / eligible, 5.0)
        vac_s, vac_r = per_capita * S, per_capita * R

        # No mobility: each admin zone has its own population and water, and zones do not
        # mix (the travel matrix is not used). Every array below holds one value per zone.
        shed = I + VI
        lam = beta_e * B / (k + B) + beta_h * shed
        inf_s, inf_v = lam * S, sigma * lam * V
        recover = tau + gamma

        derivs = {c: jnp.zeros_like(S) for c in self.compartment_list}
        derivs["S"] = mu * N - inf_s - phi * S - vac_s + alpha * V + omega * (R + VR) - mu * S
        derivs["V"] = phi * S + vac_s - inf_v - (alpha + mu) * V
        derivs["I"] = inf_s - (recover + mu) * I
        derivs["VI"] = inf_v - (recover + mu) * VI
        derivs["R"] = recover * I - vac_r - (omega + mu) * R
        derivs["VR"] = recover * VI + vac_r - (omega + mu) * VR
        derivs["B"] = xi * shed - (delta + nu) * B
        # Cumulative inflows; all new cases = I_total + VI_total.
        totals = {
            "I_total": inf_s,                              # new unvaccinated cases
            "VI_total": inf_v,                             # new breakthrough cases
            "V_total": phi * S + vac_s,                    # vaccinations of susceptibles
            "R_total": recover * I,                        # recoveries of I
            "VR_total": recover * VI + vac_r,              # recoveries of VI + doses to R
            "S_total": alpha * V + omega * (R + VR),       # waning returns to S (births excluded)
        }
        for name, value in totals.items():
            if name in derivs:
                derivs[name] = value
        return jnp.stack([derivs[c] for c in self.compartment_list])
