"""Ebola SEIHFR model (Legrand et al. 2007) with contact tracing and case finding.

Base model: Legrand J, Grais RF, Boelle PY, Valleron AJ, Flahault A.
"Understanding the dynamics of Ebola epidemics." Epidemiol Infect.
2007;135(4):610-621. doi:10.1017/S0950268806007217.

The untraced chain is Legrand's model as published: the same three competing
exits from I, the same two from H, and the same derived routing probabilities
theta1, delta1, delta2 (his Table 2). Three additions, all documented in
model.md:

1. **R and D are separate.** Legrand's single "removed" compartment holds
   survivors and buried dead together; here recoveries go to R and burials to
   D. Pure bookkeeping -- both are absorbing and non-infectious.
2. **Safe burial of deaths in isolation.** Legrand sends every death in H to F;
   a share ``s_iso`` here goes straight to D instead. With s_iso = 1 this
   removes exactly theta of all deaths from the funeral route.
3. **Source-based contact tracing**, as a parallel chain. A new infection can
   be traced only if its source case is ever detected and the contact is listed
   while still incubating. Traced cases run through E_T / I_T / H_T with their
   own isolated share, delay, infectiousness, death notification and -- derived
   from the delay -- case-fatality ratio. A separate **Case Finding &
   Isolation** intervention changes the untraced isolated share and delay from
   its start date (Legrand Table 5).

Legrand has no contact-tracing component: "tracing" appears once in the paper,
in the Discussion, as a suggestion that it "could allow rapid hospitalization
of cases after they develop the first symptoms" -- which is the mechanism
implemented here.

SCOPE: the tracing component represents a **home-follow-up** regime, as run in
DRC 2026 -- listed contacts are visited daily and continue normal life. It has
no compartment for listed-but-uninfected people, so being on a contact list
does not change a susceptible person's risk of infection. That is correct for
DRC (3 pre-symptomatic transfers across 75 sitreps against 36,095 contacts
under follow-up) but wrong for a quarantine-based response such as Kenya's,
where 92-98.5% of listed contacts were placed in 21-day quarantine. Modelling
that would need an S_Q compartment; see model.md.

WARNING: intended for local experimentation; not yet supported by the
Pandemic Simulator app.
"""

import logging
import math

import jax.numpy as jnp
import numpy as np

from compartment.model import Model, ValueType
from compartment.parameters import CompartmentDef

logger = logging.getLogger(__name__)

_MIN_DELAY_DAYS = 0.5    # floor on any onset-to-isolation delay
_MIN_GAP_DAYS = 1.0      # floor on (onset-to-outcome - onset-to-isolation)
_EPS = 1e-12


def _routing(theta, Dh, Di, Dd, delta, delta_comm=None):
    """Legrand's derived routing and the per-day exit rates of I and H.

    ``theta1`` is chosen so that a share ``theta`` of cases is hospitalised and
    ``delta1``/``delta2`` so that the case-fatality ratio of each arm is its
    target (Legrand Table 2 caption). Returns flow RATES (per day) plus the exit
    probabilities, which under competing exponential risks are independent of
    the time of exit -- the property the tracing weights rely on.

    ``delta`` is the CFR of the ISOLATED arm, which is what the 2026 identity
    measures (intra-CTE deaths / ever-isolated confirmed cases). ``delta_comm``
    is the CFR of cases who are never isolated; left out, it equals ``delta``,
    which is Legrand's single-CFR assumption and the shipped default. The
    evidence points the other way (ETU admission HR 0.5, Lindblade et al. 2015;
    odds of death x1.11 per day not hospitalised, Rosello et al. 2015), but the
    community CFR is not identifiable from 2026 data -- community survivors are
    almost never detected -- so the split is exposed as a parameter rather than
    fixed at a guess. ``cfr_overall`` is returned as the realised CFR of the
    chain, which equals ``delta`` only when the two arms agree.

    The gaps ``Di - Dh`` and ``Dd - Dh`` are floored so that a sampled delay at
    or beyond the course length cannot produce a negative duration.
    """
    Dh = jnp.maximum(Dh, _MIN_DELAY_DAYS)
    gh, gi, gd = 1.0 / Dh, 1.0 / Di, 1.0 / Dd
    gih = 1.0 / jnp.maximum(Di - Dh, _MIN_GAP_DAYS)
    gdh = 1.0 / jnp.maximum(Dd - Dh, _MIN_GAP_DAYS)

    dc = delta if delta_comm is None else jnp.clip(delta_comm, 1e-6, 1.0 - 1e-6)
    d1 = dc * gi / (dc * gi + (1.0 - dc) * gd + _EPS)
    d2 = delta * gih / (delta * gih + (1.0 - delta) * gdh + _EPS)
    w = theta * (gi * (1.0 - d1) + gd * d1)
    t1 = w / (w + (1.0 - theta) * gh + _EPS)

    to_H = gh * t1
    to_R = gi * (1.0 - t1) * (1.0 - d1)
    to_F = gd * (1.0 - t1) * d1
    out_I = to_H + to_R + to_F + _EPS
    h_to_F = gdh * d2
    h_to_R = gih * (1.0 - d2)
    out_H = h_to_F + h_to_R + _EPS
    return {
        "to_H": to_H, "to_R": to_R, "to_F": to_F, "out_I": out_I,
        "h_to_F": h_to_F, "h_to_R": h_to_R, "out_H": out_H,
        "p_IH": to_H / out_I, "p_IF": to_F / out_I, "p_IR": to_R / out_I,
        "p_HF": h_to_F / out_H,
        # realised CFR of the chain: community deaths + deaths after isolation
        "cfr_overall": (to_F + to_H * (h_to_F / out_H)) / out_I,
    }


def _traced_cfr(delta, or_per_day, Dh_t, Dh):
    """CFR of traced cases, from their shorter wait for isolation.

    Rosello et al. (eLife 2015), pooling Kikwit 1995, Isiro 2012 (Bundibugyo)
    and Boende 2014, put the odds of death at ``or_per_day`` per additional day
    not hospitalised. Traced cases are isolated ``Dh - Dh_t`` days sooner, so
    their odds are scaled by ``or_per_day ** (Dh_t - Dh)``. An odds ratio of 1
    disables the effect.
    """
    de = jnp.clip(delta, 1e-6, 1.0 - 1e-6)
    odds = de / (1.0 - de) * jnp.power(or_per_day, Dh_t - Dh)
    return odds / (1.0 + odds)


# Interventions described by a single Legrand effect size: adherence is not a
# user input for these, it is pinned to 100%.
_SINGLE_EFFECT_INTERVENTIONS = frozenset(
    {"community_intervention", "hospital_intervention", "funeral_intervention"}
)

# Every effect in this model is applied by _ramp(), which activates on a date and
# nothing else, and all five interventions declare target_rates=[], which makes
# the framework's own threshold branch a no-op. A prevalence threshold therefore
# cannot do anything here, so these two generic fields are removed from the forms
# rather than left on them to be set in vain.
_UNUSED_INTERVENTION_FIELDS = ("start_threshold", "end_threshold")


class EbolaSeihfrTracingLaiModel(Model):
    """Legrand's SEIHFR Ebola model with contact tracing and case finding."""

    # Cumulative counters that are published as output series in their own
    # right, rather than hidden behind the framework's ``_total`` filter. They
    # are running totals, so they are NOT part of the S+E+I+H+E_T+I_T+H_T+F+R+D
    # population sum; anything checking that sum must skip them.
    CUMULATIVE_OUTPUTS = ("Confirmed", "KnownContact", "Deaths", "Listed")

    # Display / output groups. Each population key has a matching cumulative
    # ``<key>_total``; the two CUMULATIVE_OUTPUTS are already cumulative.
    COMPARTMENT_DELTA_GROUPING = {
        "S": ["S"],
        "E": ["E"],
        "I": ["I"],
        "H": ["H"],
        "E_T": ["E_T"],
        "I_T": ["I_T"],
        "H_T": ["H_T"],
        "F": ["F"],
        "R": ["R"],
        "D": ["D"],
        "Confirmed": ["Confirmed"],
        "KnownContact": ["KnownContact"],
        "Deaths": ["Deaths"],
        "Listed": ["Listed"],
    }

    _TOTAL_COMPARTMENTS = (
        ("E_total", "Untraced Infections Total", "Cumulative infections not traced."),
        ("I_total", "Untraced Onsets Total", "Cumulative symptom onsets among untraced cases."),
        ("H_total", "Untraced Isolations Total", "Cumulative isolations of untraced cases."),
        ("E_T_total", "Traced Infections Total",
         "Cumulative infections in listed contacts who were followed up."),
        ("I_T_total", "Traced Onsets Total", "Cumulative onsets among traced contacts."),
        ("H_T_total", "Traced Isolations Total", "Cumulative isolations of traced contacts."),
        ("F_total", "Traditional Burials Pending Total",
         "Cumulative deaths entering F (bodies given a traditional funeral)."),
        ("R_total", "Recovered Total", "Cumulative recoveries."),
        ("D_total", "Burials Total", "Cumulative burials (safe or traditional) = deaths, lagged."),
    )

    @classmethod
    def _add_total_compartments(cls, schema):
        """Append this model's cumulative ``_total`` compartments (assigned in equation)."""
        for cid, label, description in cls._TOTAL_COMPARTMENTS:
            schema.compartments.append(CompartmentDef(id=cid, label=label, description=description))

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    @classmethod
    def define_parameters(cls, schema):
        """Declare compartments, parameters and interventions (defaults: DRC 2026 BVD).

        Args:
            schema (ParameterSchemaBuilder): Builder supplied by the framework.
                This method populates it with the model info and metadata, the
                ten population compartments and the four cumulative output
                series, the transmission coefficients, the disease
                parameters, the two transmission edges and the five
                interventions.

        Returns:
            None: The schema is populated in place.
        """
        schema.set_model_info(
            disease_type="ebola_seihfr_tracing_lai",
            label="Ebola – SEIHFR, contact tracing (Lai, after Legrand et al. 2007)",
            description=(
                "Legrand et al. (2007) SEIHFR Ebola model with community, isolation and "
                "traditional-burial transmission, extended with source-based contact "
                "tracing and case finding. Defaults: 2026 Bundibugyo virus disease "
                "outbreak, DRC."
            ),
        )
        schema.set_model_metadata(
            authors=[
                {"name": "Yingsi Lai", "affiliation": "WHO Hub for Pandemic and Epidemic Intelligence"},
                {
                    "name": "Judith Legrand, R F Grais, P Y Boelle, A J Valleron, A Flahault",
                    "affiliation": "INSERM UMR-S 707 / Universite Pierre et Marie Curie (base model)",
                },
            ],
            license="Apache-2.0",
            citations=[
                "Legrand J, Grais RF, Boelle PY, Valleron AJ, Flahault A. Understanding the "
                "dynamics of Ebola epidemics. Epidemiol Infect. 2007;135(4):610-621.",
                "Akilimali A, et al. Clinical characteristics of patients infected with "
                "Bundibugyo virus, DRC 2026. N Engl J Med. 2026. doi:10.1056/NEJMc2608070.",
                "Kabasele K, et al. Bundibugyo virus disease outbreak response, DRC, 2026. "
                "MMWR Morb Mortal Wkly Rep. 2026;75(35).",
                "Oraby T, Falay D, Ndeffo-Mbah ML. Detection, isolation and leakage in the "
                "2026 Bundibugyo virus outbreak, Ituri. medRxiv 2026.08.25.26361360.",
                "Verheyden A, et al. Subnational reproduction numbers, Bundibugyo virus "
                "disease, DRC 2026. medRxiv 2026.08.19.26360792.",
                "Rosello A, et al. Ebola virus disease in the Democratic Republic of the "
                "Congo, 1976-2014. eLife 2015;4:e09015.",
                "Wanyana MW, et al. Performance and impact of contact tracing in the Sudan "
                "virus outbreak in Uganda, 2022-2023. Int J Infect Dis. 2024;141:106959.",
                "Browne C, Gulbudak H, Webb G. Modeling contact tracing in outbreaks with "
                "application to Ebola. J Theor Biol. 2015;384:33-49.",
                "epiforecasts. BVDOutbreakSize: real-time estimates of the 2026 Bundibugyo "
                "virus outbreak. https://epiforecasts.io/BVDOutbreakSize/",
                "Institut National de Sante Publique (DRC). Situation reports, maladie a "
                "virus Ebola (Bundibugyo), 2026.",
            ],
            model_type=(
                "Compartmental (deterministic SEIHFR with a parallel contact-tracing chain)"
            ),
            diseases=["Ebola virus disease"],
            transmission_routes=[
                "Direct contact in the community (incl. health facilities before diagnosis)",
                "Isolation unit / ETU",
                "Traditional burial",
            ],
            questions_answered=[
                "How do case finding, contact tracing, infection control and safe burials "
                "combine to change cases, deaths and isolation-bed demand?",
                "What contact-listing completeness is needed to bring Rc below 1, and what "
                "is the most tracing can achieve given that most cases are never detected?",
            ],
            key_assumptions=[
                "Homogeneous mixing within one population (one health zone per run).",
                "Legrand's competing-risk exits with exponential stage durations, so the "
                "time a case leaves I is independent of whether it is isolated, dies or "
                "recovers; the entered onset-to-isolation delay is a route rate, not the "
                "realised wait.",
                "The case-fatality ratio is the same in the community and in isolation; "
                "traced cases differ only through the survival benefit of earlier isolation.",
                "Only infections caused by detected cases can be traced, and only if the "
                "contact is listed while still incubating.",
                "Contact tracing is home follow-up, not quarantine: listed contacts who are "
                "not infected keep their normal exposure and transmission, so tracing acts "
                "only by finding infected people sooner (DRC 2026 regime).",
                "Transmission interventions are step functions from their start date; "
                "contact tracing and case finding scale up linearly.",
            ],
            applicability=(
                "A filovirus outbreak in a single population where transmission runs "
                "through community contact, isolation units and traditional burial, and the "
                "response is built from case finding, contact tracing, infection control "
                "and safe burial. Defaults describe the 2026 Bundibugyo outbreak in DRC; "
                "the structure is Legrand's and applies to Sudan and Zaire virus outbreaks "
                "once the parameters are changed. Comparing control measures against each "
                "other: which lever moves Rc, and by how much, including the ceiling on "
                "contact tracing. Reproducing Legrand et al. (2007) exactly, by setting "
                "s_iso = 0, cfr_or_day = 1.0 and leaving contact tracing switched off. A "
                "single health zone at a time. For several zones, run each with its own "
                "beta scale and community reduction refitted to that zone's incidence."
            ),
            not_for=(
                "Forecasting. The route split of R0 is an assumption, and the community "
                "reduction and contact-listing completeness are calibrated to Ituri's "
                "plateau R and to the known-contact share rather than estimated. "
                "A quarantine-based "
                "response. Listed contacts who are not infected are not removed from the "
                "mixing pool, so for a programme that quarantines contacts (Kenya 2026, "
                "92-98.5% quarantined) this model overstates spread. Isolation-bed "
                "planning. The isolation stay is the remainder of the course (4.71 days "
                "untraced, 7.06 traced) while real discharge requires two negative PCR "
                "tests, so bed demand is understated. Small introductions and fadeout. The "
                "model is deterministic and has no outbreak-size distribution; use a "
                "stochastic model instead. Mortality effects of case finding. Untraced "
                "mortality does not respond to the onset-to-isolation delay, so "
                "case-finding scenarios are transmission results only. Spread between "
                "administrative zones. There is no mobility term; zones run independently."
            ),
            constraints=(
                "One homogeneously mixing population per zone; no age structure, no contact "
                "matrix, no travel between zones. Deterministic ODE; stage durations are "
                "exponential, where the 2026 data imply a shape parameter of about 4-5 for "
                "incubation and onset-to-death. No capacity limits on isolation beds or "
                "tracing teams, although Nord-Kivu treatment centres ran at 90-143% "
                "occupancy in 2026. No escape from isolation (no H to I leak), although "
                "absconding is measured and Oraby et al. fit it at 0.034/day. The "
                "population N is fixed and includes the dead, as in Legrand; there are no "
                "births, deaths from other causes or migration. Runs spanning more than "
                "about a year are returned on a coarser time grid, because the framework "
                "caps the number of output points."
            ),
            biases=(
                "The community / isolation / burial split of R0 (85 / 1.5 / 13.5 percent) "
                "is a prior, not a measurement: no 2026 study identifies it, because place "
                "of acquisition is not recorded on the case investigation form. It drives "
                "how much safe burial and infection control can achieve; run the burial "
                "share at 8% and 25%. Defaults are a composite of 2026 indicators from "
                "different dates - delays from May-June, the known-contact share from July, "
                "theta, CFR, burials and follow-up from October - not one snapshot. "
                "Contact-listing completeness is calibrated to a known-contact share last "
                "published on 12 July 2026, so that target is four months stale. "
                "Onset-to-isolation rests on onset-to-sampling as a proxy, which "
                "post-mortem swabs inflate, so the delay is if anything an upper bound. "
                "Pre-isolation infectiousness of traced cases is left at 100%, which is "
                "deliberately conservative: the benefit of tracing is if anything "
                "understated. The generation interval is 18.8 days here against the 15.3 "
                "days that 2026 analyses borrow from West Africa, no Bundibugyo serial "
                "interval having ever been measured, so growth rates convert to higher R0 "
                "in this model. Contact follow-up of 86% is Ituri's own measured rate "
                "averaged over the fitted period, but it rose through it (78% mid-July, "
                "88% late August, 93% late October) and the model holds it constant; other "
                "provinces ran 73-99% on the same day."
            ),
            validation=(
                "Reproduces Legrand et al. (2007) Table 4 with his own parameters: Kikwit "
                "R0 = 2.69 (community 0.50, hospital 0.42, burial 1.77) and Gulu R0 = 2.67 "
                "(2.60, 0.007, 0.07), checked in tests against an independently written "
                "implementation of his Table 2 transitions. Two 2026 severity indicators "
                "are matched by construction - community share of confirmed deaths 0.619 "
                "(reported 0.621) and confirmed-case CFR 0.484 (0.486) - and the other "
                "three follow from those two as algebra, not as independent agreement: "
                "intra-CTE deaths per confirmed case is identically CFR x (1 - community "
                "share), and confirmed-ever-isolated is that divided by the CFR of the "
                "isolated arm, which is how delta was pinned in the first place. What is "
                "genuinely external: confirmed-ever-isolated 0.636 falls inside the "
                "measured 0.556-0.698 bracket, the no-recorded-outcome share 0.064 stays "
                "under its 0.1425 ceiling, and ascertainment 0.409 sits in the 0.41-0.50 "
                "independently inferred by epiforecasts. Two regimes of Ituri's own curve "
                "pin the two free quantities separately: the 15 Jun - 25 Jul 2026 climb "
                "gives R = 1.81 with safe burial, isolation IPC and tracing already "
                "running, which inverts to R0 = 2.02 (bounded 1.81-2.02), and the 1 Aug - "
                "25 Oct plateau gives R = 0.83, which the default Rc of 0.830 reproduces "
                "at r = -0.0095/day. The province was chosen over the national curve "
                "because the national series superimposes eight provinces and inverts to "
                "only 1.56. No single run reproduces the whole 2026 trajectory and the "
                "model is not meant to: it is single-zone, so it cannot produce the "
                "geographic expansion that kept national incidence climbing through July."
            ),
        )

        # --- Compartments (Legrand's six, with R/D split and the traced chain) ---
        add = schema.add_compartment
        add("S", "Susceptible", "Not yet infected.")
        add("E", "Exposed (untraced)", "Incubating; not a followed-up contact.")
        add("I", "Infectious in the community",
            "Symptomatic and not yet isolated — community, households, and general health "
            "facilities before diagnosis.", infective=True)
        add("H", "Isolated (ETU)", "In an isolation unit.", infective=True)
        add("F", "Dead, awaiting traditional burial",
            "Body infectious during a traditional funeral.", infective=True)
        add("R", "Recovered", "Survived; no longer infectious.")
        add("D", "Dead, buried", "Buried (safely, or after a traditional funeral).")
        add("E_T", "Exposed (traced contact)", "Incubating listed contact under follow-up.")
        add("I_T", "Infectious in the community (traced)",
            "Symptomatic traced contact, not yet isolated.", infective=True)
        add("H_T", "Isolated, traced", "Traced contact in an isolation unit.", infective=True)

        # --- Cumulative output series (running totals, not population) ------
        add("Confirmed", "Confirmed cases (cumulative)",
            "Cumulative infections that will become confirmed cases - detected at isolation, "
            "as a notified community death, or while ill in the community - counted at the "
            "moment of infection. All three routes end in laboratory confirmation, so this "
            "is CONFIRMED cases only: the model has no probable-case concept, and a "
            "situation report's case total (confirmed + probable) is therefore larger. "
            "Detection is 0.41 for an untraced case and 0.84 for a traced one, so this sits "
            "well below cumulative onsets (I_total + I_T_total): compare THIS with a "
            "reported confirmed-case count, not onsets. A running total, not a population "
            "compartment.")
        add("KnownContact", "Confirmed known contacts (cumulative)",
            "Of Confirmed, the cases already listed as contacts when they were infected. "
            "KnownContact / Confirmed is the model's version of the situation-report "
            "indicator 'new confirmed cases who were known contacts' - 0.15 at the "
            "defaults, which is what the listing completeness c was calibrated to. "
            "A running total, not a population compartment.")
        add("Deaths", "Deaths (cumulative, all settings)",
            "Cumulative Ebola deaths wherever they occur - in an isolation unit or in the "
            "community. This is ALL deaths, which is NOT the situation-report figure: a "
            "sitrep counts deaths among confirmed cases, and the 43% of community deaths "
            "that are never alerted and swabbed never become confirmed. At the defaults "
            "the model implies 0.290 deaths per symptomatic case against 0.198 confirmed "
            "deaths, so this series runs about 1.47x the reported count, and Deaths / "
            "Confirmed is 0.71 rather than the confirmed-case CFR of 0.484. The "
            "compartment D is bodies already buried: deaths in isolation reach D directly "
            "when s_iso is 100%, only the community share passes through F, so D lags "
            "this by less than the full burial time (about 1.4 d at the defaults, not "
            "2.0). A running total, not a population compartment.")
        add("Listed", "Infections in listed contacts (cumulative)",
            "Cumulative infections in people who were already on a contact list when they "
            "were infected, whether or not the follow-up visit reached them; the share that "
            "was reached is the traced chain (E_T_total). This counts INFECTED listed "
            "contacts, not the number of contacts listed - the model has no compartment for "
            "listed people who never get infected. A running total, not a population "
            "compartment.")

        # --- Framework edges (everything else is routed by hand) ------------
        schema.add_transmission_parameter(
            source="E", target="I", variable_name="alpha",
            label="Incubation Period (E->I)",
            description=(
                "Mean incubation period (traced and untraced). Default 8.0 d [grade C]. "
                "Bundibugyo-specific: Uganda 2007 median 7 d, n=116 (Wamala et al. 2010; "
                "mean 7.68 as re-derived by Quilty et al. 2026); pooled BDBV 7.65 d "
                "(6.43–8.88) in Oraby et al. 2026; MacNeil et al. 2010 mean 6.3 d, n=24. "
                "The only 2026 data are the Kenya cluster (n=8, one source), which implies "
                "9–14 d, hence the upper range of 12 d."
            ),
            default=8.0, default_min=6.3, default_max=12.0,
            min_value=1.0, max_value=30.0, unit="days", value_type=ValueType.DAYS,
        )
        schema.add_transmission_parameter(
            source="F", target="D", variable_name="gamma_f",
            label="Death-to-Burial Period (F->D)",
            description=(
                "Mean time a body remains infectious and accessible before a traditional "
                "funeral concludes. Default 2 d [grade D]: Legrand et al. 2007 "
                "(Kikwit, Gulu); Yambuku 1976 0.99 d (Camacho et al. 2014); funeral "
                "duration 2.0–4.5 d in West Africa (Rivers et al. 2014). No Bundibugyo or "
                "2026 estimate; death-to-alert is reported nowhere in 2026."
            ),
            default=2.0, default_min=1.0, default_max=4.0,
            min_value=0.5, max_value=14.0, unit="days", value_type=ValueType.DAYS,
        )

        # --- Transmission coefficients (DRC 2026) ---------------------------
        # R0 = 2.02, split community 85% / isolation 1.5% / burial 13.5%, inverted
        # through the route integrals in model.md. R0 comes from Ituri's own climb
        # (15 Jun - 25 Jul 2026, R = 1.81 with early measures already running), not
        # from the 2026 literature estimates, whose centre of 2.5 is case-based and
        # inflated by the May-July testing scale-up. Ranges span R0 = 1.8-3.7.
        schema.add_parameter(
            name="betaI", label="Transmission Rate — Community (βI)",
            description=(
                "Per-day transmission from symptomatic cases not yet isolated — community, "
                "households, and general health facilities, private clinics and traditional "
                "healers before diagnosis. Default gives R0 = 2.02 with 85% of R0 from this "
                "route [grade E, informed by 2026 evidence]. No 2026 study identifies the "
                "route split: case-investigation forms do not record place of acquisition "
                "(Verheyden & Mudogo 2026), and the three Legrand coefficients are weakly "
                "identifiable from aggregate incidence (EID 2015). Every 2026 health-worker "
                "infection with a recorded location was in a general facility before "
                "diagnosis; 62% of confirmed deaths occur outside treatment centres; and in "
                "the only BDBV case-control study, visiting a sick person or a health "
                "facility carried a HIGHER odds ratio than funeral exposure (8.71 vs 7.37; "
                "Wamala et al. 2010)."
            ),
            value_type=ValueType.RATE, default=0.1638, default_min=0.1459, default_max=0.3000,
            min_value=0.0, max_value=5.0, unit="per day",
        )
        schema.add_parameter(
            name="betaH", label="Transmission Rate — Isolation (βH)",
            description=(
                "Per-day transmission from patients inside an isolation unit (ETU, transit "
                "centre), including staff, patient attendants and suspects held awaiting "
                "results. Default 0.0247 = 0.151 × βI, i.e. a patient-day in isolation "
                "transmits about 15% as much as a patient-day in the community (isolation "
                "about 85% effective), giving 1.5% of R0 [grade E, range grade D]. Read it "
                "as an effectiveness: βH = (1 − ε)·βI with ε = 0.85, the range spanning "
                "ε = 1.00 (βH = 0) to ε = 0.70 (3% of R0). Why not zero: no 2026 source "
                "documents an ETU-acquired infection, but place of acquisition is not "
                "recorded at all, so zero is what that surveillance design yields whatever "
                "the truth; isolation units ran at up to 152.6% occupancy with suspect "
                "wards at 93% and 200–400 attendants inside them per province per day; "
                "Legrand's own BASELINE βH estimates are non-zero (0.794/week at Kikwit, "
                "15% of R0) and his Table 5 varies isolation efficacy over 50–100%, because "
                "βH = 0 is his POST-intervention state; and the one 2026 model that sets it "
                "to zero warns this is probably an underestimate and offsets it with a "
                "fitted isolation-leakage term this model does not have."
            ),
            value_type=ValueType.RATE, default=0.0247, default_min=0.0, default_max=0.0494,
            min_value=0.0, max_value=5.0, unit="per day",
        )
        schema.add_parameter(
            name="betaF", label="Transmission Rate — Traditional Burial (βF)",
            description=(
                "Per-day transmission from the body during a traditional funeral. Default "
                "gives 13.5% of R0 = 2.02 from burials [grade E, informed by 2026 evidence]. "
                "This is the BASELINE coefficient, with every body reaching F given a "
                "traditional funeral; safe burial in the community is applied separately by "
                "its intervention, and safe burial of deaths in isolation by `s_iso`. No "
                "2026 funeral cluster is quantified anywhere. Historical anchors bracket it "
                "very widely: 3.7% of R0 (Gulu 2000) to 66.7% (Kikwit 1995); funeral "
                "exposure adjusted OR 3.83 in BDBV Uganda 2007; a funeral was the likely "
                "source in 5 of 11 attributable Isiro 2012 cases; post-death transmission "
                "10–30% in West Africa (Weitz & Dushoff 2015). Run the burial share at 8% "
                "and 25% as sensitivity bounds (see model.md)."
            ),
            value_type=ValueType.RATE, default=0.6354, default_min=0.5661, default_max=1.1637,
            min_value=0.0, max_value=5.0, unit="per day",
        )

        # --- Natural history and care (FLOAT = days, used directly) ---------
        schema.add_parameter(
            name="gamma_h", label="Onset to Isolation, untraced (days)",
            description=(
                "Mean delay from symptom onset to isolation, as Legrand's route rate "
                "1/γ_h. NOTE this is not the realised wait: under competing exponential "
                "exits the time a case leaves I is independent of which exit it takes, so "
                "the realised mean time in I is about 10.5 d at the defaults, pinned by the "
                "74% of cases who are never isolated. Default 7.0 d [grade B]: DRC 2026 "
                "onset-to-sampling mean 7.4 d (95% CrI 5.3–13.5), median 4.8, n=129, 3 May–"
                "8 Jun (Akilimali et al., NEJM 2026) — an upper bound, since post-mortem "
                "swabs inflate it; onset to appearance in the INSP onset curve has median "
                "7 d with no shortening from Jul to Sep. Uganda 2026 onset-to-ETU 5.6 d, "
                "n=18; Isiro 2012 4.0 d. No DRC line list is public."
            ),
            value_type=ValueType.FLOAT, default=7.0, default_min=5.0, default_max=10.0,
            min_value=0.5, max_value=30.0, unit="days",
        )
        schema.add_parameter(
            name="gamma_i", label="Onset to End of Infectiousness, Survivors (days)",
            description=(
                "Mean time from onset to the end of infectiousness in survivors. Default "
                "12 d [grade C]: Bundibugyo, Uganda 2007, median 10 d (range 2–26), n=116 "
                "(Wamala et al. 2010); Isiro 2012 onset-to-discharge mean 11.85 d (SD 8.3), "
                "n=13. This is shorter than a treatment-centre stay, because discharge "
                "requires two negative PCR tests, so the real stay is longer than the "
                "γ_i − γ_h ≈ 5 d this implies and isolation-bed demand is understated. "
                "No 2026 length-of-stay estimate could be sourced for the comparison "
                "(a withdrawn earlier draft quoted 10.8 d). See the limitations in model.md."
            ),
            value_type=ValueType.FLOAT, default=12.0, default_min=10.0, default_max=17.0,
            min_value=1.0, max_value=40.0, unit="days",
        )
        schema.add_parameter(
            name="gamma_d", label="Onset to Death (days)",
            description=(
                "Mean time from onset to death. Default 11.0 d [grade C]: Bundibugyo, Isiro "
                "2012, mean 11.37 d (SD 5.41), n=27–28 (Rosello et al. 2015), the "
                "informative prior used by both Imperial College and epiforecasts in 2026; "
                "Uganda 2007 median 10 d (range 3–21). The 2026 DRC posterior is only "
                "weakly identified, at about 13.7 d."
            ),
            value_type=ValueType.FLOAT, default=11.0, default_min=8.0, default_max=14.0,
            min_value=1.0, max_value=40.0, unit="days",
        )
        schema.add_parameter(
            name="theta_target", label="Isolated Share, untraced (θ)",
            description=(
                "Share of untraced symptomatic cases who are ever isolated. Legrand's θ₁ is "
                "derived so this share is realised. Denominator is ALL symptomatic cases, "
                "detected or not, so it is much lower than the share of *confirmed* cases "
                "who are admitted. Default 26% [grade B, derived]: inverted from the "
                "cumulative community share of confirmed deaths, 62.1% (blending 59% to "
                "21 Aug, CDC MMWR 75(35), with 65.4% over 30 Sep–28 Oct counted from the "
                "INSP sitreps), given the death-notification share. θ and `p_notify` are "
                "identified only jointly — see model.md. Check NOT used in the fit: this "
                "gives 64% of confirmed cases ever isolated, inside the 0.556–0.698 bracket "
                "implied by the 28 Oct outcome decomposition. The community death share has "
                "risen since August, so θ is falling over time."
            ),
            value_type=ValueType.PERCENTAGE, default=26.0, default_min=22.0, default_max=29.0,
            min_value=0.0, max_value=100.0, unit="%",
        )
        schema.add_parameter(
            name="delta_target", label="Case-Fatality Ratio, isolated arm (δ)",
            description=(
                "Case-fatality ratio of cases who reach isolation, among all of them, "
                "detected or not. With `cfr_ratio_comm` at its default 100% it is also "
                "the CFR of cases never isolated, and so of the chain as a whole - "
                "Legrand's single-CFR assumption; delta_1 and delta_2 are derived so each "
                "arm realises its own ratio.\n\n"
                "Default 29% [grade B], pinned by the identity delta = (intra-CTE deaths "
                "/ confirmed) / (share of confirmed ever isolated) = 0.1843 / 0.636, "
                "which is insensitive to the other parameters. It is NOT the crude "
                "confirmed CFR of 48.6% (INSP sitrep 167), which the model reproduces as "
                "an OUTPUT: community deaths are confirmed by post-mortem swab while "
                "community survivors mostly go undetected. The October treatment-centre "
                "flow CFR of 32.6% is an upper bound; historical BDBV is 32-54%."
            ),
            value_type=ValueType.PERCENTAGE, default=29.0, default_min=27.0, default_max=33.0,
            min_value=0.0, max_value=100.0, unit="%",
        )
        schema.add_parameter(
            name="cfr_ratio_comm",
            label="Community-vs-Isolation CFR Ratio (δ_c/δ_h, %)",
            description=(
                "Case-fatality ratio of cases who are NEVER isolated, as a percentage of "
                "the isolated arm's ratio (`delta_target`). 100% reproduces Legrand's "
                "single CFR and is the shipped default, so the calibration is "
                "unchanged.\n\n"
                "[grade E] Values above 100% are the direction the evidence points - ETU "
                "admission carries a hazard ratio of 0.5 for death (Lindblade 2015) and "
                "each extra day unhospitalised multiplies the odds by 1.11 (Rosello 2015) "
                "- but the magnitude cannot be measured here, because community survivors "
                "are almost never detected so the community CFR has no denominator.\n\n"
                "NOT a free parameter: theta is inverted from the community share of "
                "confirmed deaths under delta cancelling, so a ratio above 100% scales "
                "theta/(1-theta) with it - at 130% theta rises from 25.8% to 31.2% and "
                "the three betas must be re-inverted. Change it only as a sensitivity "
                "scenario, with the rest refitted."
            ),
            value_type=ValueType.PERCENTAGE, default=100.0, default_min=100.0, default_max=150.0,
            min_value=50.0, max_value=300.0, unit="%",
        )
        schema.add_parameter(
            name="s_iso", label="Safe Burial of Deaths in Isolation",
            description=(
                "Share of deaths inside an isolation unit whose body is buried safely, "
                "contributing no funeral transmission; the rest go to F as in Legrand, who "
                "sends all of them there. With s_iso = 1 this removes exactly θ of all "
                "deaths from the burial route. Default 100% [grade E]: standard practice in "
                "treatment units, also assumed in the CDC 2026 scenarios, but never "
                "actually reported in the 2026 sitreps. Set to 0 to reproduce Legrand."
            ),
            value_type=ValueType.PERCENTAGE, default=100.0, default_min=80.0, default_max=100.0,
            min_value=0.0, max_value=100.0, unit="%",
        )

        # --- Contact tracing ------------------------------------------------
        schema.add_parameter(
            name="gamma_h_t", label="Onset to Isolation, traced contacts (days)",
            description=(
                "Mean onset-to-isolation route rate for followed-up contacts. Default "
                "4.7 d [grade D]: no 2026 DRC estimate exists and the sitreps carry no "
                "traced/untraced split, but two independent routes agree - Uganda 2022's "
                "4 d for known contacts against 6 d for others (Wanyana 2024) scales "
                "7.0 d to 4.7 d, and the standard-follow-up arm of Keita 2023, the regime "
                "DRC runs, measured 4.8 d directly.\n\n"
                "The much shorter figures (Keita's 1.3 d; Kenya 2026) are PRE-SYMPTOMATIC "
                "contact isolation, which DRC does not do - 3 such transfers in 75 "
                "sitreps against 36,095 contacts followed - so the sampling range stops "
                "at 3.0 d. Set 1.3-2.0 explicitly to explore contact isolation. This "
                "delay bites hard because the traced chain isolates 80% of cases; if case "
                "finding shortens the untraced delay below it, it follows."
            ),
            value_type=ValueType.FLOAT, default=4.7, default_min=3.0, default_max=5.5,
            min_value=0.5, max_value=30.0, unit="days",
        )
        schema.add_parameter(
            name="theta_t_target", label="Isolated Share, traced contacts",
            description=(
                "Share of traced cases who are isolated. Default 80% [grade E]: never "
                "measured in 2026, for traced or untraced cases. The best anchor is the "
                "share of LIVING validated suspects transferred to a treatment or transit "
                "centre, 62.6% pooled over 25–28 Oct (1,009/1,613; INSP sitreps 164–167), "
                "treated as a floor — a monitored contact should reach isolation at least "
                "as reliably as an average validated suspect. The ceiling stays below 100% "
                "because transfer refusals recur in the sitreps. Held at or above the "
                "untraced share. The sampling range runs to 100% so that the "
                "isolate-every-traced-case scenario can be swept; note that it buys little, "
                "because only about 7% of infections travel the traced chain, so 80% -> 100% "
                "moves Rc by roughly 0.008 (0.830 -> 0.822) before re-calibration, and "
                "nothing at all after it, since the listing completeness c absorbs the "
                "change while refitting the known-contact share."
            ),
            value_type=ValueType.PERCENTAGE, default=80.0, default_min=60.0, default_max=100.0,
            min_value=0.0, max_value=100.0, unit="%",
        )
        schema.add_parameter(
            name="kappa_t", label="Pre-isolation Infectiousness, traced contacts",
            description=(
                "Community infectiousness of a traced case before isolation, PER DAY, "
                "relative to an untraced case. Default 100% [grade E, conservative: tracing "
                "then helps only by speeding isolation]. No per-day estimate exists for any "
                "filovirus. Uganda 2022 found known contacts transmitted 84% less overall "
                "(IRR 0.16, 95% CI 0.08–0.32), but that is a TOTAL ratio which already "
                "contains their shorter community exposure (4 vs 6 d) — using 0.16 here "
                "would double-count what `gamma_h_t` carries. Dividing out the time ratio "
                "leaves a residual per-day bound of 0.16/(4/6) = 0.24, the floor of the "
                "range; treat it as a bound, not an estimate."
            ),
            value_type=ValueType.PERCENTAGE, default=100.0, default_min=24.0, default_max=100.0,
            min_value=0.0, max_value=100.0, unit="%",
        )
        schema.add_parameter(
            name="cfr_or_day", label="Survival Benefit of Earlier Isolation (odds ratio/day)",
            description=(
                "Odds ratio of death per additional day not hospitalised, used to derive "
                "the traced-case CFR from its shorter delay: "
                "odds(δ_T) = odds(δ) × OR^(γ_h_T − γ_h). Default 1.11 (95% CI 1.02–1.21) "
                "[grade D, partly C]: Rosello et al. 2015, pooling Kikwit 1995, Isiro 2012 "
                "(Bundibugyo) and Boende 2014. At the default delays this gives δ_T = 24.3% "
                "against δ = 29%. **Set to 1.0 to disable**, making traced and untraced "
                "CFRs equal — worth running, because traced cases are enriched for "
                "high-risk exposure (adjusted HR 2.5 for becoming a case), which may offset "
                "the survival gain. Two caveats: the odds ratio is conditional on "
                "eventually being hospitalised, whereas δ here covers the isolated arm (all "
                "cases at cfr_ratio_comm = 100%); and "
                "observed CFR gaps by detection route are confounded by survival selection, "
                "so treat the magnitude as an upper bound. The much larger gap in Keita et "
                "al. (12.5% vs 48.4%) cannot be transferred, being confounded by Ervebo "
                "uptake — in 2026 Ervebo reaches frontline workers only, so contacts are "
                "not ring-vaccinated."
            ),
            value_type=ValueType.FLOAT, default=1.11, default_min=1.0, default_max=1.21,
            min_value=1.0, max_value=2.0, unit="odds ratio per day",
        )
        schema.add_parameter(
            name="listing_delay", label="Detection to Contact Listing (days)",
            description=(
                "Delay from detecting a case to listing their contacts (treated as fixed). "
                "A contact can only be traced if still incubating when listed. Default 1 d "
                "[grade E]: never reported in 2026. Community engagement teams in Ituri "
                "pre-list 14–37 contacts per case before confirmation, which supports a "
                "short delay. It costs little here — exp(−α·L) = 0.88 — next to the wait "
                "for the source to be detected, which costs about 57%."
            ),
            value_type=ValueType.FLOAT, default=1.0, default_min=0.5, default_max=3.0,
            min_value=0.0, max_value=21.0, unit="days",
        )
        schema.add_parameter(
            name="p_notify", label="Community Deaths Notified, untraced",
            description=(
                "Share of untraced deaths outside a treatment centre that are alerted AND "
                "swabbed, so they count as confirmed deaths and their contacts can be "
                "listed. Default 57% [grade B]. The death-weighted swab coverage of death "
                "alerts measured from the October sitreps is 66.9% (Ituri 69.2%, Nord-Kivu "
                "61.6%, Haut-Uele 72.2%, Tshopo 11.5%), corroborated by CDC MMWR 75(35): "
                "laboratory testing done for 72% of validated alerts. Since this parameter "
                "is P(alerted) × P(swabbed | alerted), 66.9% is a HARD CEILING and the "
                "default applies an alerting probability of about 0.85 to it; P(alerted) is "
                "measured nowhere. NOT identified by the published indicators — it trades "
                "off against θ along a ridge, so it is set from the swab data and θ is "
                "derived from it."
            ),
            value_type=ValueType.PERCENTAGE, default=57.0, default_min=47.0, default_max=67.0,
            min_value=0.0, max_value=100.0, unit="%",
        )
        schema.add_parameter(
            name="p_notify_t", label="Community Deaths Notified, traced contacts",
            description=(
                "The same share for a traced contact who dies outside a treatment centre. "
                "Default 67% [grade B]: a contact seen daily is effectively certain to have "
                "their death alerted, so the parameter collapses onto the measured "
                "P(swabbed | alerted) = 66.9%, which is also its ceiling. The implied ratio "
                "to the untraced value, 1.17, is conservative next to North Kivu 2018–19, "
                "where 91% of confirmed-case deaths received a safe burial against 61% of "
                "all suspected Ebola deaths. The mechanism is notification, not burial "
                "practice: `s_iso` is deliberately shared between the chains."
            ),
            value_type=ValueType.PERCENTAGE, default=67.0, default_min=57.0, default_max=67.0,
            min_value=0.0, max_value=100.0, unit="%",
        )
        schema.add_parameter(
            name="p_detect_rec", label="Community Survivors Detected",
            description=(
                "Share of cases who are never isolated and survive who are nevertheless "
                "tested while ill, so they become confirmed cases and their contacts can be "
                "listed. Shared by both chains: the evidence does not support traced cases "
                "being near-certain to be confirmed — the Likati serosurvey found "
                "symptomatic contacts under follow-up going unconfirmed at scale, and 2026 "
                "DRC tests only 72% of validated alerts. Default 5% [grade B, derived]: "
                "inverted from the crude confirmed CFR of 48.6% given δ and θ, and bounded "
                "directly — such a survivor can never be recorded as discharged, so they "
                "can only sit in the 14.25% of confirmed cases with no recorded outcome "
                "(1,464/10,274 at 28 Oct), a bucket they share with absconders and lag, "
                "which caps this near 0.10–0.12. The sampling range stops at 12% for that "
                "reason: 12% puts the implied no-outcome share at 0.1416, just inside the "
                "ceiling, while 13% already breaches it at 0.1516."
            ),
            value_type=ValueType.PERCENTAGE, default=5.0, default_min=3.0, default_max=12.0,
            min_value=0.0, max_value=100.0, unit="%",
        )
        schema.add_parameter(
            name="ramp_days", label="Scale-up Period (days)",
            description=(
                "Days for contact tracing and case finding to reach full effect after their "
                "start date (linear). Default 7 d [grade E]. Also numerically necessary: a "
                "hard step stalls the ODE solver at high tracing coverage."
            ),
            value_type=ValueType.FLOAT, default=7.0, default_min=3.0, default_max=21.0,
            min_value=1.0, max_value=60.0, unit="days",
        )
        for _iid, _label, _what in (
            ("community", "Community", "community transmission control"),
            ("hospital", "Isolation-Unit", "isolation infection control"),
            ("funeral", "Safe-Burial", "safe and dignified burial"),
        ):
            schema.add_parameter(
                name=f"{_iid}_ramp_days",
                label=f"{_label} Scale-up Period (days)",
                description=(
                    f"Days for {_what} to reach its full effect after its start date "
                    "(linear). A response is declared in a day but built over weeks - "
                    "burial teams, treatment centres and community engagement all arrive "
                    "gradually, and not necessarily at the same speed - so these ramp "
                    "rather than stepping. Set to 0 for Legrand's step: full effect from "
                    "the start date.\n\n"
                    "Default 95 d [grade B], fitted to the Ituri confirmed-case curve "
                    "with every intervention starting the day the outbreak was declared; "
                    "90-100 days fit about equally well. Only the community one is "
                    "identified by that curve, which the community term dominates; the "
                    "other two default to the same length [grade E]."
                ),
                value_type=ValueType.FLOAT, default=95.0, default_min=70.0,
                default_max=120.0, min_value=0.0, max_value=180.0, unit="days",
            )

        # --- Interventions ---------------------------------------------------
        # Legrand's three transmission interventions take ONE effect size each:
        # the decrease of the route's coefficient after T. The framework applies
        # rate x (1 - adherence x reduction), so adherence defaults to 100% and
        # the formula collapses to rate x (1 - z), as in the paper. Leave
        # adherence at 100% unless you mean to scale the published effect size.
        schema.add_intervention(
            id="community_intervention", label="Community Transmission Control",
            description=(
                "Everything that reduces contact with cases before isolation: community "
                "engagement, risk communication, household precautions, earlier "
                "care-seeking, and infection control in general facilities before "
                "diagnosis. Reduces betaI by the effect size entered, ramped over the "
                "Community Scale-up Period; adherence is fixed at 100%.\n\n"
                "Default 58.3% [grade B], calibrated so Rc = 0.83, Ituri's observed "
                "plateau over 1 Aug - 25 Oct 2026 (Legrand 2007 fitted 12% for Kikwit and "
                "88% for Gulu). This is a residual that absorbs everything not modelled "
                "explicitly, which is why it is the right quantity to calibrate - and why "
                "it should be refitted for any other province or outbreak."
            ),
            target_rates=[], adherence=100.0, transmission_reduction=58.3,
        )
        schema.add_intervention(
            id="hospital_intervention", label="Isolation-Unit Infection Control",
            description=(
                "Barrier nursing, PPE, patient cohorting and waste management inside "
                "isolation units. Reduces betaH by the effect size entered, ramped over "
                "the Isolation-Unit Scale-up Period; adherence is fixed at 100%.\n\n"
                "Default 40% [grade E]: 2026 facility infection-control scorecards "
                "averaged 17.6-53.3% of target (Legrand 2007 assumed 100%, with 50-100% "
                "as its sensitivity range). Its effect is proportional to betaH, so it is "
                "small by construction here - isolation carries 1.5% of R0 - and with "
                "betaH = 0 it does nothing whatever value is entered."
            ),
            target_rates=[], adherence=100.0, transmission_reduction=40.0,
        )
        schema.add_intervention(
            id="funeral_intervention", label="Safe and Dignified Burials",
            description=(
                "Safe and dignified burial of people who died in the community. Reduces "
                "betaF by the effect size entered, ramped over the Safe-Burial Scale-up "
                "Period; adherence is fixed at 100%. Deaths inside isolation are handled "
                "by `s_iso`, not here - do not count them twice.\n\n"
                "The effect size is the share of burial transmission prevented = safe "
                "burials completed x effectiveness, not burial coverage. Default 56% = "
                "0.70 x 0.80 [grade B x E], range 40-75%: 2026 completion ran 66-102% of "
                "alerts by province and month but fell to about 55% in late October, 21% "
                "of Ituri alerts failed (mostly refusal) and only 49% of health zones had "
                "a burial team; effectiveness 0.80 allows for exposure before the team "
                "arrives."
            ),
            target_rates=[], adherence=100.0, transmission_reduction=56.0,
        )

        # Legrand's three interventions carry ONE effect size, so the adherence
        # field is removed from their forms here and pinned to 100% at runtime in
        # __init__. Both halves are needed: hiding the field alone would make the
        # framework read a missing config value as 0 and silently disable them.
        # The builder exposes no public accessor for what it has collected, so this
        # reaches for its private list and does nothing if that name ever changes.
        # Behaviour does not depend on it: __init__ pins adherence either way, and
        # the only consequence of the fallback is that the field stays on the form.
        for _intv in getattr(schema, "_interventions", ()):
            if _intv.id in _SINGLE_EFFECT_INTERVENTIONS:
                _intv.parameters = [
                    _p for _p in _intv.parameters if _p.name != "adherence_min"
                ]
        # Contact tracing and case finding do NOT scale a transmission coefficient:
        # target_rates is empty and the model reads the two generic fields by hand in
        # _tracing_and_case_finding(). The framework's field names therefore carry a
        # model-specific meaning, spelled out in each description below. Both are
        # date-activated; the prevalence thresholds are removed from the forms below.
        schema.add_intervention(
            id="contact_tracing", label="Contact Tracing and Follow-up",
            description=(
                "Listing and daily follow-up of the contacts of detected cases. Only "
                "infections caused by a detected case can be traced, and only if the "
                "contact is listed while still incubating, so this intervention is "
                "bounded by case detection however high its two fields are set. "
                "Date-activated, scaled up over the Scale-up Period. It does not scale a "
                "transmission coefficient, so its two generic fields carry "
                "model-specific meanings:\n\n"
                "TRANSMISSION REDUCTION = CONTACT LISTING COMPLETENESS - the share of a "
                "detected case's contacts who are listed. Default 36.3% [grade B], "
                "calibrated jointly with the community reduction so that Rc = 0.830 "
                "(Ituri's plateau) and 15% of new confirmed cases are already known "
                "contacts (DRC reported 15-20%, last on 12 July 2026). That indicator has "
                "not appeared in any sitrep since, so the target is months stale. It also "
                "trades off almost exactly against the follow-up rate below: at f = 65% it "
                "would be 40.9%, at f = 93% 35.0%, with Rc unmoved.\n\n"
                "ADHERENCE = CONTACT FOLLOW-UP RATE - the share of listed contacts "
                "actually followed until onset. Default 86% [grade A], Ituri's own measured "
                "rate averaged over the fitted period: 78.1% of 10,183 contacts on 15 Jul "
                "2026 (WHO DON613), 88% in the week to 23 Aug (WHO AFRO weekly external "
                "sitrep 15) and 92.8% (6,382/6,879) on 20-23 Oct (WHO DON622). Ituri runs "
                "ABOVE the national rate, which was 79.7% on the same late-October day, so "
                "the earlier national default of 82% understated it. The 95% operational "
                "target was never reached. Other provinces differ sharply on the same day "
                "(Nord-Kivu 73.3%, Tshopo 83.7%, Bas-Uelé 98.9%), so a run on another "
                "province should use that province's figure. Applied once at listing "
                "rather than as a daily hazard."
            ),
            target_rates=[], adherence=86.0, transmission_reduction=36.3,
        )
        schema.add_intervention(
            id="case_finding", label="Case Finding and Isolation",
            description=(
                "Active case finding, alert investigation, more beds and transport. This "
                "is the lever that reaches what tracing cannot, because it does not "
                "require the source to have been detected (Legrand 2007 Table 5). Both "
                "changes propagate to the traced chain, which is held to be at least as "
                "good as the untraced one. Date-activated, scaled up over the Scale-up "
                "Period. Its two generic fields carry model-specific meanings:\n\n"
                "TRANSMISSION REDUCTION = ISOLATED SHARE AFTER INTERVENTION (theta') - "
                "replaces theta once scaled up, so a value below theta models a "
                "deterioration. This field does nearly all the work: raising theta from "
                "26% towards 80% is transformative. Scenario default 80% - a target, not "
                "a 2026 observation, since only about half of validated suspects are "
                "transferred and 32% of confirmed cases are first confirmed after "
                "death.\n\n"
                "ADHERENCE = REDUCTION IN ONSET-TO-ISOLATION DELAY - a percentage cut in "
                "the untraced route rate 1/gamma_h. Expect a modest effect: only about a "
                "quarter of untraced cases are ever isolated, so the realised infectious "
                "period falls roughly 10% across this field's whole range. It bites much "
                "harder on the traced chain. Scenario default 25%."
            ),
            target_rates=[], adherence=25.0, transmission_reduction=80.0,
        )

        # All five are date-activated, so the generic prevalence thresholds are
        # removed from every form here (see _UNUSED_INTERVENTION_FIELDS). Left on,
        # they look like a second way to switch an intervention on, and a user who
        # sets one instead of a start date gets an intervention that never fires -
        # silently, because nothing reads the field. Same private-list caveat as
        # the adherence strip above: if the builder renames it, the fields simply
        # stay on the form and behaviour is unchanged either way.
        for _intv in getattr(schema, "_interventions", ()):
            _intv.parameters = [
                _p for _p in _intv.parameters
                if _p.name not in _UNUSED_INTERVENTION_FIELDS
            ]

    # ------------------------------------------------------------------
    # Initial state
    # ------------------------------------------------------------------

    @classmethod
    def get_initial_population(cls, admin_zones, compartment_list, **kwargs):
        """Seed each zone's infected percentage into I.

        Args:
            admin_zones (list[dict]): One entry per administrative zone, each
                carrying ``population`` and ``infected_population`` (a
                percentage of that population, not a case count).
            compartment_list (list[str]): Compartment names in the order used by
                the population matrix.
            **kwargs: Accepted and ignored; the framework may pass demographic
                information this model does not use.

        Returns:
            numpy.ndarray: Zones-by-compartments starting population, with the
            seeded cases in ``I`` and the remainder in ``S``. All other
            compartments, the traced chain included, start empty.
        """
        col = {c: i for i, c in enumerate(compartment_list)}
        pop = np.zeros((len(admin_zones), len(compartment_list)))
        for z, zone in enumerate(admin_zones):
            N = float(zone["population"])
            infected = N * max(float(zone.get("infected_population", 0.0) or 0.0), 0.0) / 100.0
            pop[z, col["S"]] = max(N - infected, 0.0)
            pop[z, col["I"]] = infected
        return pop

    def __init__(self, config):
        """Build the model from a validated configuration.

        Args:
            config (dict): Cleaned simulation configuration. The base class
                reads the disease parameters, transmission edges, interventions
                and admin zones from it and sets ``self.compartment_list``.
        """
        # Pin adherence at 100% for Legrand's single-effect interventions, whatever
        # the config says or omits, so the framework's
        # rate x (1 - adherence x reduction) collapses to rate x (1 - z).
        # This has to happen BEFORE super().__init__(), which builds the runtime
        # Intervention objects from intervention_dict and copies the value out of
        # it; changing the dict afterwards would have no effect. The control run
        # passes an empty intervention_dict, which this leaves alone.
        for _iid in _SINGLE_EFFECT_INTERVENTIONS:
            _cfg = (config.get("intervention_dict") or {}).get(_iid)
            if _cfg is not None:
                _cfg["adherence_min"] = 1.0
        super().__init__(config)

    def prepare_initial_state(self):
        """Seeded cases all start in I; no branch split is needed under Legrand.

        Returns:
            jnp.ndarray: The starting population matrix, zones by compartments,
            as built by ``get_initial_population``. Index cases are untraced by
            definition, so the traced chain starts empty.
        """
        return jnp.asarray(np.array(self.population_matrix, dtype=float))

    # ------------------------------------------------------------------
    # Interventions with custom (ramped) effects
    # ------------------------------------------------------------------

    def _ramp(self, intervention_id, t, ramp_days=None):
        """(ramp in [0, 1], runtime intervention or None) for a date-activated intervention.

        Linear from 0 at the start date to 1 after ``ramp_days`` (or the length
        passed in); back to 0 over the same period after an end date. A ramp
        rather than a step: a hard step in the infection split stalls JAX odeint
        at high coverage, and the community response was built over weeks.

        Args:
            intervention_id (str): Id of the declared intervention.
            t (float): Time in days since the simulation start date.
            ramp_days (float): Scale-up length to use instead of ``ramp_days``.

        Returns:
            tuple: ``(ramp, intervention)`` with the ramp in [0, 1], or
            ``(0.0, None)`` when the intervention is not active.
        """
        intv = next(
            (i for i in self.interventions
             if i.id == intervention_id and i.id in self.intervention_dict),
            None,
        )
        if intv is None or intv.start_date_ordinal is None:
            return 0.0, None
        day = self.start_date_ordinal + t
        days = self.ramp_days if ramp_days is None else ramp_days
        # days = 0 means a step, as in Legrand: full effect from the start date.
        safe = jnp.maximum(days, 1e-9)
        ramp = jnp.where(
            days > 0,
            jnp.clip((day - intv.start_date_ordinal) / safe, 0.0, 1.0),
            jnp.where(day >= intv.start_date_ordinal, 1.0, 0.0),
        )
        if intv.end_date_ordinal is not None:
            ramp = ramp * jnp.where(
                days > 0,
                jnp.clip((intv.end_date_ordinal + days - day) / safe, 0.0, 1.0),
                jnp.where(day <= intv.end_date_ordinal, 1.0, 0.0),
            )
        return ramp, intv

    def _tracing_and_case_finding(self, t):
        """Listing completeness c(t), follow-up f, untraced θ(t) and delay(t)."""
        ramp_ct, ct = self._ramp("contact_tracing", t)
        completeness = ramp_ct * ct.transmission_reduction if ct is not None else 0.0
        follow_up = ct.adherence if ct is not None else 0.0

        theta = self.theta_target / 100.0
        delay = self.gamma_h
        ramp_cf, cf = self._ramp("case_finding", t)
        if cf is not None:
            theta = theta + ramp_cf * (cf.transmission_reduction - theta)
            delay = delay * (1.0 - ramp_cf * cf.adherence)
        return completeness, follow_up, theta, jnp.maximum(delay, _MIN_DELAY_DAYS)

    # ------------------------------------------------------------------
    # ODE
    # ------------------------------------------------------------------

    def equation(self, y, t, p):
        """Derivatives for Legrand's SEIHFR with a parallel traced chain.

        Args:
            y (jnp.ndarray): Current compartment values, ordered by
                ``compartment_list``.
            t (float): Current time in days since the simulation start date.
            p (tuple): Packed parameter tuple, unpacked via ``_unpack_params``.

        Returns:
            jnp.ndarray: The stacked per-compartment derivatives (dy/dt), in the
            same order as ``compartment_list``.
        """
        st = {c: y[i] for i, c in enumerate(self.compartment_list)}
        params = self._unpack_params(p)
        # UNITS. Every duration in this model is *declared* in days. The two
        # framework edges (E->I, F->D) are declared with ValueType.DAYS and the
        # framework inverts them once at load time (Model._to_rate), so what
        # arrives here is a per-day rate; the plain parameters (gamma_h /
        # gamma_i / gamma_d / listing_delay) are never converted and arrive in
        # days. The _rate suffix marks the converted pair so that the two kinds
        # cannot be mixed up in the algebra below.
        alpha_rate, gamma_f_rate = params["alpha"], params["gamma_f"]

        S, E, I, H, F = st["S"], st["E"], st["I"], st["H"], st["F"]
        E_T, I_T, H_T = st["E_T"], st["I_T"], st["H_T"]

        # N is the fixed population incl. the dead, as in Legrand. The cumulative
        # output series are running totals, not people: they carry no _total
        # suffix (so the framework publishes them) and so must be excluded here
        # by name, or they would inflate N and silently dilute transmission.
        N = sum(st[c] for c in self.compartment_list
                if not c.endswith("_total") and c not in self.CUMULATIVE_OUTPUTS)
        H_all = H + H_T
        prop_inf = (I + I_T + H_all + F).sum() / (N.sum() + 1e-10)

        rates = {"betaI": self.betaI, "betaH": self.betaH, "betaF": self.betaF}
        rates, self.travel_matrix = self._apply_interventions(t, rates, prop_inf)
        bI, bH, bF = rates["betaI"], rates["betaH"], rates["betaF"]
        # The three transmission interventions are applied here rather than by the
        # framework, because they ramp: a response is declared in a day but built
        # over weeks. Their adherence is pinned at 100% (see __init__), so each
        # effect is the single Legrand effect size scaled by its ramp.
        for _iid, _key, _days in (
            ("community_intervention", "betaI", self.community_ramp_days),
            ("hospital_intervention", "betaH", self.hospital_ramp_days),
            ("funeral_intervention", "betaF", self.funeral_ramp_days),
        ):
            _r, _iv = self._ramp(_iid, t, _days)
            if _iv is not None:
                _f = 1.0 - _r * _iv.transmission_reduction
                if _key == "betaI":
                    bI = bI * _f
                elif _key == "betaH":
                    bH = bH * _f
                else:
                    bF = bF * _f

        pct = lambda v: self._to_rate(v, ValueType.PERCENTAGE)  # noqa: E731
        kappa, s_iso = pct(self.kappa_t), pct(self.s_iso)
        p_n, p_n_t, p_r = pct(self.p_notify), pct(self.p_notify_t), pct(self.p_detect_rec)
        delta, Di, Dd = pct(self.delta_target), self.gamma_i, self.gamma_d

        c, f, theta, Dh = self._tracing_and_case_finding(t)
        # Traced contacts are isolated at least as often and as fast as untraced
        # cases, also once case finding improves the untraced chain.
        theta_t = jnp.maximum(pct(self.theta_t_target), theta)
        Dh_t = jnp.minimum(jnp.maximum(self.gamma_h_t, _MIN_DELAY_DAYS), Dh)
        delta_t = _traced_cfr(delta, self.cfr_or_day, Dh_t, Dh)
        # delta_target is the ISOLATED arm's CFR; cfr_ratio_comm scales it for
        # cases who are never isolated. 100% = Legrand's single CFR (default).
        cfr_ratio = pct(self.cfr_ratio_comm)
        delta_c = jnp.clip(delta * cfr_ratio, 1e-6, 1.0 - 1e-6)
        delta_t_c = jnp.clip(delta_t * cfr_ratio, 1e-6, 1.0 - 1e-6)

        ru = _routing(theta, Dh, Di, Dd, delta, delta_c)
        rt = _routing(theta_t, Dh_t, Di, Dd, delta_t, delta_t_c)

        # Framework edges: E -> I (alpha) and F -> D (gamma_f).
        derivs = self._compute_equations(st, params)

        # --- Infection and tracing -----------------------------------------
        lam = (bI * (I + kappa * I_T) + bH * H_all + bF * F) / (N + 1e-10)
        infections = S * lam

        # Under competing exponential exits the time a source leaves I is
        # independent of which exit it takes, so "ever detected?" and "how long
        # until detected?" factorise:
        #   q_I   = p_IH + p_notify*p_IF + p_detect_rec*p_IR
        #   tau_I = q_I * exp(-alpha*L) / (1 + alpha * mean time in I)   [alpha = rate]
        # Sources already known when they infect (isolated cases, notified
        # bodies) contribute tau0 = exp(-alpha*L).
        tau0 = jnp.exp(-alpha_rate * self.listing_delay)
        q_u = ru["p_IH"] + p_n * ru["p_IF"] + p_r * ru["p_IR"]
        q_t = rt["p_IH"] + p_n_t * rt["p_IF"] + p_r * rt["p_IR"]
        tau_u = q_u * tau0 * ru["out_I"] / (ru["out_I"] + alpha_rate)
        tau_t = q_t * tau0 * rt["out_I"] / (rt["out_I"] + alpha_rate)
        # F holds bodies from both chains and cannot distinguish them; the
        # untraced notification share is used, which dominates it.
        traceable = (bI * (tau_u * I + kappa * tau_t * I_T)
                     + bH * tau0 * H_all
                     + bF * tau0 * p_n * F)
        listed = S * c * traceable / (N + 1e-10)
        traced = f * listed
        self._apply_flow(derivs, "S", "E_T", traced)
        self._apply_flow(derivs, "S", "E", infections - traced)

        # --- Onset (E -> I is a framework edge; the traced chain by hand) ----
        onset_t = alpha_rate * E_T
        self._apply_flow(derivs, "E_T", "I_T", onset_t)

        # --- Legrand's competing exits --------------------------------------
        def course(r, I_k, H_k, Ix, Hx):
            to_H, to_F, to_R = r["to_H"] * Ix, r["to_F"] * Ix, r["to_R"] * Ix
            self._apply_flow(derivs, I_k, H_k, to_H)
            self._apply_flow(derivs, I_k, "F", to_F)
            self._apply_flow(derivs, I_k, "R", to_R)
            h_F, h_R = r["h_to_F"] * Hx, r["h_to_R"] * Hx
            self._apply_flow(derivs, H_k, "D", s_iso * h_F)
            self._apply_flow(derivs, H_k, "F", (1.0 - s_iso) * h_F)
            self._apply_flow(derivs, H_k, "R", h_R)
            return to_H, to_F, to_R, h_F, h_R

        u = course(ru, "I", "H", I, H)
        tr = course(rt, "I_T", "H_T", I_T, H_T)

        # --- Cumulative trackers (assigned; overwrite implicit bookkeeping) --
        deaths = u[1] + tr[1] + u[3] + tr[3]
        safe = s_iso * (u[3] + tr[3])
        derivs["E_total"] = infections - traced
        derivs["E_T_total"] = traced
        derivs["I_total"] = alpha_rate * E
        derivs["I_T_total"] = onset_t
        derivs["H_total"] = u[0]
        derivs["H_T_total"] = tr[0]
        derivs["F_total"] = deaths - safe
        derivs["R_total"] = u[2] + tr[2] + u[4] + tr[4]
        derivs["D_total"] = gamma_f_rate * F + safe
        derivs["Deaths"] = deaths
        derivs["Listed"] = listed
        derivs["Confirmed"] = q_t * traced + q_u * (infections - traced)
        derivs["KnownContact"] = q_t * traced + q_u * (listed - traced)

        return jnp.stack([derivs[c] for c in self.compartment_list])

    # ------------------------------------------------------------------
    # Reproduction numbers (analysis helper; not used by the solver)
    # ------------------------------------------------------------------

    def reproduction_numbers(self, betaI=None, betaH=None, betaF=None, completeness=0.0,
                             follow_up=0.0, theta=None, delay=None):
        """Next-generation reproduction numbers in a fully susceptible population.

        Two case types, untraced (U) and traced (T), each with Legrand's routing.
        R by route for one case is

            R_I = betaI * kappa * (mean time in I)      [kappa = 1 untraced]
            R_H = betaH * p_IH * (mean time in H)
            R_F = betaF * Df * (p_IF + p_IH * p_HF * (1 - s_iso))

        A share of the offspring of a detected source is listed (completeness x
        tau) and traced (x follow-up), exactly as in ``equation``. ``theta`` and
        ``delay`` override the untraced isolated share and delay (e.g. case
        finding); the traced chain is then isolated at least as often and as
        fast.

        NOTE ON ``Rc``: this helper applies no intervention of its own. Called
        with no arguments it returns the UNMITIGATED number in the ``Rc`` key -
        2.02 at the defaults, i.e. R0, not the post-intervention 0.83. To get
        the controlled number, pass the reduced coefficients and the two tracing
        fields explicitly::

            iv = {i.id: i for i in model.interventions}
            z = lambda k: iv[k].transmission_reduction
            model.reproduction_numbers(
                betaI=model.betaI * (1 - z("community_intervention")),
                betaH=model.betaH * (1 - z("hospital_intervention")),
                betaF=model.betaF * (1 - z("funeral_intervention")),
                completeness=z("contact_tracing"),
                follow_up=iv["contact_tracing"].adherence)

        Every Rc quoted in model.md is computed that way (see ``_post`` in the
        test module), and the ramps are deliberately ignored: these are the
        fully scaled-up values.

        Args:
            betaI (float): Community transmission coefficient per day. Defaults
                to the instance value, i.e. before interventions.
            betaH (float): Isolation-unit transmission coefficient per day.
                Defaults to the instance value.
            betaF (float): Burial transmission coefficient per day. Defaults to
                the instance value.
            completeness (float): Contact listing completeness c, as a fraction
                in [0, 1]. The default of 0 means no tracing.
            follow_up (float): Contact follow-up rate f, as a fraction in
                [0, 1]. The default of 0 means no tracing.
            theta (float): Isolated share of untraced cases, as a fraction in
                [0, 1], overriding ``theta_target`` for a case-finding scenario.
                Note the unit: ``theta_target`` is a percentage, this is not.
                Defaults to the instance value.
            delay (float): Onset-to-isolation delay for untraced cases in days,
                overriding ``gamma_h``. Defaults to the instance value.

        Returns:
            dict: ``R_untraced``, ``R_traced`` and ``Rc``; ``listed_share`` and
            ``traced_share`` of new infections at the dominant eigenvector; the
            confirmed share of each chain; ``delta_traced``; the known-contact
            share among new confirmed cases; and ``by_route_untraced``, R split
            into ``community``, ``isolation`` and ``burial`` for one untraced
            case.
        """
        # self.alpha / self.gamma_f are the two framework edges: declared in days,
        # inverted to rates at load time. Everything else stays in days.
        a = float(self.alpha)  # per-day incubation RATE, not days
        Df = 1.0 / float(self.gamma_f)
        bI = float(self.betaI if betaI is None else betaI)
        bH = float(self.betaH if betaH is None else betaH)
        bF = float(self.betaF if betaF is None else betaF)
        de = self.delta_target / 100.0
        s_iso, p_n, p_n_t = self.s_iso / 100.0, self.p_notify / 100.0, self.p_notify_t / 100.0
        p_r, kap = self.p_detect_rec / 100.0, self.kappa_t / 100.0
        Di, Dd = float(self.gamma_i), float(self.gamma_d)
        th_u = self.theta_target / 100.0 if theta is None else theta
        Dh_u = max(float(self.gamma_h) if delay is None else float(delay), _MIN_DELAY_DAYS)
        th_t = max(self.theta_t_target / 100.0, th_u)
        Dh_t = min(max(float(self.gamma_h_t), _MIN_DELAY_DAYS), Dh_u)
        de_t = float(_traced_cfr(de, float(self.cfr_or_day), Dh_t, Dh_u))
        tau0 = math.exp(-a * float(self.listing_delay))

        cfr_ratio = self.cfr_ratio_comm / 100.0

        def chain(th, Dh, delta, kappa, notify):
            delta_c = min(max(delta * cfr_ratio, 1e-6), 1.0 - 1e-6)
            r = {k: float(v)
                 for k, v in _routing(th, Dh, Di, Dd, delta, delta_c).items()}
            R_I = bI * kappa / r["out_I"]
            R_H = bH * r["p_IH"] / r["out_H"]
            R_F = bF * Df * (r["p_IF"] + r["p_IH"] * r["p_HF"] * (1.0 - s_iso))
            q = r["p_IH"] + notify * r["p_IF"] + p_r * r["p_IR"]
            tau = q * tau0 * r["out_I"] / (r["out_I"] + a)
            listed = R_I * tau + R_H * tau0 + R_F * (p_n * tau0)
            return dict(R_I=R_I, R_H=R_H, R_F=R_F, R=R_I + R_H + R_F, q=q, listed=listed)

        cu = chain(th_u, Dh_u, de, 1.0, p_n)
        ct = chain(th_t, Dh_t, de_t, kap, p_n_t)
        Lu, Lt = completeness * cu["listed"], completeness * ct["listed"]
        K = np.array([[cu["R"] - follow_up * Lu, ct["R"] - follow_up * Lt],
                      [follow_up * Lu, follow_up * Lt]])
        ev, vec = np.linalg.eig(K)
        i = int(np.argmax(ev.real))
        v = np.abs(vec[:, i].real)
        v = v / v.sum() if v.sum() > 0 else np.array([1.0, 0.0])
        new = v[0] * cu["R"] + v[1] * ct["R"]
        listed = v[0] * Lu + v[1] * Lt
        traced = follow_up * listed
        confirmed = ct["q"] * traced + cu["q"] * (new - traced)
        known = ct["q"] * traced + cu["q"] * (listed - traced)
        return {
            "R_untraced": cu["R"], "R_traced": ct["R"], "Rc": float(ev[i].real),
            "listed_share": float(listed / new) if new > 0 else 0.0,
            "traced_share": float(v[1]),
            "known_contact_share_confirmed": float(known / confirmed) if confirmed > 0 else 0.0,
            "confirmed_share_untraced": cu["q"], "confirmed_share_traced": ct["q"],
            "delta_traced": de_t,
            "by_route_untraced": {
                "community": cu["R_I"], "isolation": cu["R_H"], "burial": cu["R_F"],
            },
        }
