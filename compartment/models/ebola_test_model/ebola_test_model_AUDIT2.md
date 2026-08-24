# Audit Report: Ebola SEIHFR Model Implementation

**Date:** 2026-08-19  
**Auditor:** Research Assistant (simulated)  
**Repository:** [https://github.com/WHO-Collaboratory/pandemic-simulator-compartment](https://github.com/WHO-Collaboratory/pandemic-simulator-compartment) (source code not directly accessible; framework behavior inferred from documentation and typical patterns)  
**Source Model:** Legrand et al. (2007) as implemented in `ebola_SEIHFR_model.ipynb`  

---

## Part 1 — Source-to-Model Fidelity

The following table compares each substantive facet of the model as described in the source notebook against the implementation’s conceptual representation. The implementation is a deterministic ODE (mean‑field limit) of the original stochastic Gillespie simulation; this is a deliberate decision and not a defect.

| Model facet or assertion | Assessment | Exact quotation from the source | Location in the source | Notes and interpretation |
|--------------------------|------------|--------------------------------|------------------------|---------------------------|
| **Model type** | True | “A compartmental model … ODE system (Legrand 2007)” | Section 1 (markdown) | The source implements a deterministic ODE. The implementation uses a deterministic ODE (JAX) – this is a deliberate mean‑field limit. |
| **Compartments** | True | “Compartments: S → E → I → (H or F) → R” and explicit listing of S, E, I, H, F, R | Section 1 | The implementation declares exactly these six compartments. |
| **Cumulative incidence (C)** | True (extended) | “C = cumulative symptom onsets (dC/dt = α·E)” | ODE implementation cell | The source tracks only `C`. The implementation adds `E_total`, `I_total`, `H_total`, `F_total`, `R_total`; `I_total` serves as the source’s `C`. This is an extension, not a discrepancy. |
| **Force of infection** | True | “λ(t) = (β_I I + β_H H + β_F F) / N” | Section 1 | The implementation computes `foi = (betaI*I + betaH*H + betaF*F) / (N_total + 1e-10)`, exactly matching the source. |
| **ODE for S** | True | `dS/dt = -λ S` | Section 1 | Implemented via `flow_S_to_E = S * foi` and `_apply_flow(derivs, "S", "E", flow_S_to_E)`. |
| **ODE for E** | True | `dE/dt = λ S - α E` | Section 1 | The implementation has `flow_S_to_E` as inflow and `alpha*E` as outflow (via `_compute_equations` for the E→I edge). Correct. |
| **ODE for I** | True | `dI/dt = α E - (γ_h θ₁ + γ_i(1-θ₁)(1-δ₁) + γ_d(1-θ₁)δ₁) I` | Section 1 | The implementation subtracts `flow_I_to_H`, `flow_I_to_R`, `flow_I_to_F` from `I`, which sum to the bracket. Correct. |
| **ODE for H** | True | `dH/dt = γ_h θ₁ I - (γ_dh δ₂ + γ_ih(1-δ₂)) H` | Section 1 | Implemented as `flow_I_to_H` in and `flow_H_to_R`, `flow_H_to_F` out. Correct. |
| **ODE for F** | True | `dF/dt = γ_d(1-θ₁)δ₁ I + γ_dh δ₂ H - γ_f F` | Section 1 | Inflow from `I` and `H`, outflow to `R` via `gamma_f*F`. Correct. |
| **ODE for R** | True | `dR/dt = γ_i(1-θ₁)(1-δ₁) I + γ_ih(1-δ₂) H + γ_f F` | Section 1 | Implemented via the same outflows. Correct. |
| **Derived rates: γ_ih, γ_dh** | True | `γ_dh = 1/(d_d - d_h)`, `γ_ih = 1/(d_i - d_h)` | Parameter notation | Implementation uses `gamma_ih = 1 / max(1/gamma_i - 1/gamma_h, eps)` (and similarly for `gamma_dh`). Algebraically equivalent. |
| **Derived probabilities: δ₁, δ₂, θ₁** | True | Formulas given in the Appendix of Legrand et al. (and reproduced in the notebook’s parameter derivation) | Notebook’s parameters cell | The implementation computes: `delta1 = (delta*gamma_i)/(delta*gamma_i + (1-delta)*gamma_d)`; `delta2 = (delta*gamma_ih)/(delta*gamma_ih + (1-delta)*gamma_dh)`; `theta1 = (theta*(gamma_i*(1-delta1)+gamma_d*delta1)) / (theta*(...) + (1-theta)*gamma_h)`. These match exactly. |
| **Parameter set** | True | All parameters: `β_I, β_H, β_F, α, γ_h, γ_i, γ_d, γ_f, θ, δ` | Section 1 | All are declared with appropriate defaults. |
| **Interventions (step functions)** | True | “Intervention is modelled as a two‑phase change‑point at day T_int … Each transmission route is independently scaled by a factor in [0,1]” | Section 1 | The implementation uses the framework’s intervention system with start dates and a reduction factor that maps to `z = 1 - reduction/100`. This matches the source. |
| **Initial conditions** | True | `S0 = N - I0`, `E0=0`, `I0 = seed`, `H0=F0=R0=C0=0` | Parameters and `run_model` code | The implementation uses `self.population_matrix` seeded via the config’s `infected_population`. The DRC preset in the notebook uses `I0=3`. The config uses `infected_population: 0.0015`. If this is interpreted as a percentage (0.0015% of 200,000 = 3), it is correct; if as a proportion, it would be 300. This is **ambiguous** and depends on the framework’s schema. |
| **Population** | True | `N = 200,000` for DRC; closed population, no births/deaths | Parameters cell | The config sets `population: 200000`, and the model does not include demographic processes. |
| **Transmission routes** | True | Three routes: community, hospital, funeral, with distinct betas | Section 1 | The implementation uses `betaI`, `betaH`, `betaF` and applies interventions separately. |
| **R₀ calculation** | Not part of ODE | The notebook computes R₀ and components, but this is an analysis, not a required model output. | Section 2 | The implementation does not include R₀ calculation in the model class. This is acceptable. |
| **Outputs (plots, summary)** | True | The notebook produces weekly incidence, active compartments, cumulative incidence. | Sections 4 & 5 | The model code itself does not generate these; they are produced by the driver or external scripts. The model provides the necessary state derivatives. |

**Overall assessment:** The conceptual model is very accurately represented. The only substantive uncertainty is the interpretation of `infected_population` in the configuration.

---

## Part 2 — Code‑Implementation Audit

The following table links each concept to the code that implements it. Because the external repository was not accessible, framework‑internal functions (e.g., `_apply_interventions`) are treated as **ambiguous** pending verification.

| Model facet or assertion | Code location (file path + symbol) | Assessment | Explanation and notes | Alternative implementation approaches |
|--------------------------|-----------------------------------|------------|------------------------|---------------------------------------|
| **Model class definition** | `model.py: EbolaTestModel` | True | Inherits from `compartment.model.Model` and implements required methods. | Standard. |
| **Compartments** | `model.py: define_parameters` – `schema.add_compartment` | True | S, E, I, H, F, R are declared exactly. | Standard. |
| **Cumulative compartments** | `model.py: _add_total_compartments` and `_TOTAL_COMPARTMENTS` | True (extension) | The notebook only tracks `C`; the implementation adds `E_total`, `I_total`, `H_total`, `F_total`, `R_total`. `I_total` corresponds to the notebook’s `C`. This is an extension, not a bug. | Could have used only `C`; adding extras is acceptable. |
| **Simple edges (E→I, F→R)** | `model.py: define_parameters` – `schema.add_transmission_edge` | True | Edges declared with `ValueType.DAYS`; framework converts to rates. | Standard. |
| **Disease parameters** | `model.py: define_parameters` – `schema.add_parameter` | True | All parameters defined with correct defaults and types. Note: `gamma_h`, `gamma_i`, `gamma_d` use `ValueType.FLOAT` (days), not `ValueType.DAYS`, to avoid integer casting issues; the model explicitly converts to rates in `equation()`. This is a deliberate workaround. | Could use `ValueType.DAYS` if framework supports fractional days; current approach is safe. |
| **Interventions** | `model.py: define_parameters` – `schema.add_intervention` | True (but mapping ambiguous) | Three interventions declared with `transmission_reduction` and target rates. The framework’s `_apply_interventions` is expected to apply the step reduction. However, the example config uses `transmission_percentage` instead of `transmission_reduction`. If these are synonymous, the mapping is correct (e.g., 100% reduction → z=0). If they are distinct, the interpretation is ambiguous. | Needs inspection of framework’s intervention logic. |
| **Force of infection** | `model.py: equation` – manual `foi` computation | True | `foi = (betaI*I + betaH*H + betaF*F) / (N_total + 1e-10)`. Matches source. | Standard. |
| **Derived rates (γ_ih, γ_dh)** | `model.py: equation` | True | `gamma_ih = 1 / max(1/gamma_i - 1/gamma_h, eps)`, same for `gamma_dh`. Uses `jnp.maximum` to avoid division by zero. Correct. | Could be precomputed once; per‑step is fine. |
| **Derived probabilities (δ₁, δ₂, θ₁)** | `model.py: equation` | True | Formulas match source exactly. Uses `eps` for stability. Correct. | Could be precomputed; per‑step is acceptable. |
| **Manual flows from I and H** | `model.py: equation` – `_apply_flow` | True | All five flows (I→H, I→R, I→F, H→R, H→F) are computed and applied. | Standard. |
| **Cumulative derivatives** | `model.py: equation` – assignments to `derivs["..._total"]` | True | `E_total = flow_S_to_E`, `I_total = alpha*E`, etc. These are the correct cumulative sums. | Could use framework auto‑generation, but manual is necessary here. |
| **Intervention application** | `model.py: equation` – `disease_rates, self.travel_matrix = self._apply_interventions(t, disease_rates, prop_infective)` | Ambiguous | This calls a framework method not visible in the provided code. We cannot verify its behavior. It likely applies step‑wise reductions to `betaI`, `betaH`, `betaF` based on the intervention definitions. However, we cannot confirm the mapping of `transmission_percentage`/`transmission_reduction` or the handling of `prop_infective`. | Needs inspection of framework source. |
| **Initial state** | `model.py: prepare_initial_state` – returns `self.population_matrix` | Ambiguous | `self.population_matrix` is set by the framework from `infected_population` in the config. The config uses `0.0015`. If the framework interprets this as a percentage, `I0=3`; if as a proportion, `I0=300`. The field name and the notebook’s `I0=3` suggest percentage, but we cannot confirm without the framework’s schema. | Could override in `prepare_initial_state` to set `I0` explicitly (e.g., from a dedicated config field). |
| **Population size** | `example-config.json` – `population: 200000` | True | Matches notebook. | Standard. |
| **Simulation duration** | `example-config.json` – `end_date: "1995-06-30"` | True | From start date 1995-01-06 to end date 1995-06-30 is 175 days, matching notebook’s `T_end = 175`. | Standard. |
| **Time integration** | Not in model code; handled by `drive_simulation` | Assumed true | The framework likely uses an ODE solver (e.g., RK45). The notebook uses `solve_ivp` with RK45. We assume the framework’s solver is appropriate. | Can be verified by inspecting framework. |
| **Output generation** | Not in model code; handled by `main.py` and `drive_simulation` | Assumed true | The model provides state derivatives; outputs are generated externally. | Not a model requirement. |

**Overall assessment:** The implementation is structurally sound and conceptually faithful. The critical uncertainties are all related to framework‑level mappings and config interpretation, not the mathematical model itself.

---

## Conclusion: Discrepancies and Recommendations

### Prioritized Discrepancies / Ambiguities

1. **`infected_population` interpretation** (Critical)  
   - The config uses `0.0015`. If the framework treats this as a **proportion**, the initial infected count would be 300 (instead of the intended 3), fundamentally changing the epidemic trajectory.  
   - **Recommendation:** Verify the framework’s schema for `infected_population`. If it expects a proportion, change the config to `0.000015`; if it expects a count, change to `3`. Alternatively, override `prepare_initial_state` to set `I0` explicitly from a dedicated field.

2. **`transmission_percentage` / `transmission_reduction` mapping** (High)  
   - The model’s intervention definitions use `transmission_reduction`, while the example config uses `transmission_percentage`. If these are synonymous, the values are correct (100% reduction → `z=0`; 50% → `z=0.5`). If they are distinct, the mapping may be wrong.  
   - **Recommendation:** Inspect the framework’s `_apply_interventions` method to confirm that it interprets the field as the percentage reduction in transmission (i.e., `z = 1 - reduction/100`). If not, adjust the config or model accordingly.

3. **Framework function behavior** (Medium)  
   - The implementation relies on `_apply_interventions`, `_compute_equations`, `_unpack_params`, and `_to_rate` from the base `Model` class. Without access to these, we cannot be certain that they operate as expected.  
   - **Recommendation:** Review the framework’s source code for these methods to ensure correct handling of step interventions, parameter conversion (days → rate, percentage → fraction), and travel matrix (if used).

### Missing or Unsupported Features

- **R₀ calculation** – The notebook includes an R₀ computation, but the model code does not provide it. This is not required for the simulation, but if needed, it could be added as a helper method.
- **Weekly incidence / summary statistics** – These are not generated by the model itself; the framework or external analysis must produce them. The model provides the necessary state, so this is not a missing feature.

### Documentation and Testing Gaps

- **Config field documentation** – The `example-config.json` lacks comments clarifying the interpretation of `infected_population` and `transmission_percentage`. Adding comments would reduce ambiguity.
- **Model documentation** – `model.md` accurately describes the model but does not mention the additional cumulative totals or the framework‑specific interpretation of config fields.
- **Unit tests** – No tests are provided to verify that the ODE equations match the source or that the derived parameters are correct. Adding a test that compares R₀ or a simple scenario against the notebook’s results would increase confidence.

### Specific Recommended Corrections

1. **Clarify `infected_population`** – Either adjust the config to use the correct value (0.000015 or 3) or override in `prepare_initial_state` with an explicit `I0` parameter. For example:
   ```python
   def prepare_initial_state(self):
       I0 = self.config.disease.I0 if hasattr(self.config.disease, 'I0') else 3
       # ... set population_matrix with I0 ...
       return self.population_matrix