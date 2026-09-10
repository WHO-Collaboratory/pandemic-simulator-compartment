# Measles SEIR calibrated to the 2025 Texas outbreak (González-Parra et al.)

A worked example of the **input-dataset workflow**: an SEIR measles model that reads a recent, real case series through `self.dataset(...)`, fits its transmission rate to that series *inside the model* at load time, and then simulates forward.

## The dataset

Four weekly incidence data points digitised from **Figure 1** of González-Parra, Vestrand & Mujynya (2025), covering the early rising phase of the 2025 Southwest US outbreak in the affected 5-county community (Lea NM; Dawson, Gaines, Terry, Yoakum TX; combined population N = 128,924):

| week_start | day | cases |
|------------|-----|-------|
| 2025-01-29 | 0   | 2     |
| 2025-02-05 | 7   | 15    |
| 2025-02-12 | 14  | 25    |
| 2025-02-26 | 28  | 85    |

Day 21 (2025-02-19) is absent from the figure. Day 0 (2 cases) records the imported index cases that seeded the outbreak; it is excluded from the growth-rate OLS (see §The fit). Days 7, 14, and 28 form the established community-transmission rising limb used for calibration.

Shipped file: `data/measles_tx_2025_weekly.csv` — columns `week_start, day, cases, population`.

### References

- González-Parra G., Vestrand A., Mujynya R. (2025). Modeling and Characterizing the Growth of the Texas–New Mexico Measles Outbreak of 2025. *Epidemiologia* 6(4):60. doi:10.3390/epidemiologia6040060 — model basis, parameters, population setup, and R₀ estimation.
- Anderson R.M., May R.M. (1991). *Infectious Diseases of Humans: Dynamics and Control.* Oxford Univ. Press, ch. 4 — all-or-nothing vaccine model: effective susceptible fraction = 1 − p · VE.
- Keeling M.J., Rohani P. (2008). *Modeling Infectious Diseases in Humans and Animals.* Princeton Univ. Press — SEIR structure and vaccination modelling.
- Wallinga J., Lipsitch M. (2007). How generation intervals shape the relationship between growth rates and reproductive numbers. *Proc R Soc B* 274:599–604 — SEIR formula mapping exponential growth rate r to R_t.
- Guerra F.M. et al. (2017). The basic reproduction number (R₀) of measles: a systematic review. *Lancet Infect Dis* 17(12):e420–e428 — R₀ varies widely across settings and methods.

## Population and initial conditions

Following González-Parra et al. (2025), the model uses the **combined 5-county population** of the affected community:

- **N = 128,924** (Lea NM; Dawson, Gaines, Terry, Yoakum TX)

Initial conditions are controlled by the **user-facing `initial_immune_fraction` parameter** (percentage, 0–100):

- **R(0) = f_imm × N** — already immune at t = 0, where f_imm = `initial_immune_fraction` / 100
- **S(0) = (1 − f_imm) × N** — susceptible at t = 0
- **I(0)** = seed infections (set via `infected_population` in the case file)

The **default `initial_immune_fraction` = 92.15%** corresponds to the TX 2025 5-county community under the all-or-nothing vaccine model (Anderson & May 1991): p × VE = 0.95 × 0.97 = 0.9215, giving S(0) ≈ 10,120 and R(0) ≈ 118,804. To simulate a different population, set `initial_immune_fraction` in the Simulator UI or the config's `Disease` block — the calibrated β is unaffected.

> **`initial_immune_fraction` and `CALIBRATION_SUSCEPTIBLE_FRACTION` are independent.** The calibration constant (0.0785) is locked to the TX 2025 outbreak dataset and used only in `_calibrated_beta()`. `initial_immune_fraction` controls the simulation's S(0)/R(0) split for whatever population you are modelling.

## Model structure

```
S --beta*S*I/N--> E --sigma*E--> I --gamma*I--> R
```

Frequency-dependent SEIR, following González-Parra et al. (2025).

| Parameter | Symbol | Value | Source |
|-----------|--------|-------|--------|
| Latent period | T_E = 1/σ | **11 days** | González-Parra 2025 |
| Infectious period | T_I = 1/γ | **7 days** | González-Parra 2025 |
| Initial immune fraction | f_imm | **92.15% (user)** | p × VE = 0.95 × 0.97 (Anderson & May 1991) |
| Transmission rate | β | *calibrated from data* | see §The fit |

**beta is not a user input.** On this platform every declared transmission edge becomes a field in the Simulator interface. Because `beta` here is fully determined by the data, it is deliberately *not* declared as an edge — so it never appears as a control, and there is no box for the user to fill. It is fitted at load time and applied through a manual force-of-infection term in `equation()`. User-facing controls are `sigma`, `gamma`, and `initial_immune_fraction`. (The MMR vaccination intervention still scales the fitted `beta`.)

## The fit

At construction, `__init__` reads the CSV and estimates `beta` in three steps, matching the approach of González-Parra et al. (2025) and Wallinga & Lipsitch (2007).

**Step 1 — fit the growth rate r.**
Log-linear OLS regression over the **established rising limb** — days 7, 14, and 28. Day 0 (2 imported seed cases) is excluded: it records index-case importation, not community-generated transmission, so including it inflates r and over-estimates R₀.

```
log(cases_t) = log(C₀) + r · t      (t = 7, 14, 28 days)
```

The slope `r` is the exponential growth rate in cases per day.

**Step 2 — Wallinga-Lipsitch SEIR formula → R_t.**
The SEIR generation-interval formula maps r to the effective reproduction number in the observed (partially immune) population:

```
R_t = (1 + r · T_E)(1 + r · T_I)
```

where T_E = 1/σ = 11 d and T_I = 1/γ = 7 d.

**Step 3 — recover β consistent with the initial conditions.**
In the SEIR ODE, the effective reproduction number at t = 0 is:

```
R_eff(0) = β · S(0) / (N · γ) = β · f_s / γ
```

Setting R_eff(0) = R_t and solving:

```
R₀_intrinsic = R_t / f_s          (intrinsic R₀ in a fully susceptible population)
β             = R₀_intrinsic · γ  = R_t · γ / f_s
```

This step is the same relationship as González-Parra's R₀ = R_t / (1 − p · VE), since f_s = 1 − p · VE = 0.0785.

**Numerical result** with days 7, 14, 28 (day 0 seed excluded), σ = 1/11, γ = 1/7, f_s = 0.0785:

- r ≈ 0.083 /day (OLS slope over days 7, 14, 28)
- R_t ≈ (1 + 0.083 × 11)(1 + 0.083 × 7) ≈ 1.91 × 1.58 ≈ **3.0**
- R₀_intrinsic ≈ 3.0 / 0.0785 ≈ **38.6** — within the paper's reported range of 32–40 ✓
- β ≈ 38.6 × (1/7) ≈ **5.5 /day**
- Check: R_eff(0) = β · f_s / γ = 5.5 × 0.0785 × 7 ≈ 3.0 ✓

**Fit-once-and-cache** on the class (one fit per run, reused across the with/without-intervention baseline and all uncertainty draws), and a **safe fallback** to a literature value (0.4 /day) if the dataset is missing or the fit fails.

## The fitted R₀ in context

The fit on days 7, 14, 28 recovers **R₀_intrinsic ≈ 38.6**, consistent with González-Parra et al.'s (2025) reported range of 32–40. This reflects the actual measles intrinsic transmissibility in a highly vaccinated community where even a small susceptible pool sustains rapid growth. Note that Guerra et al. (2017) document wide variation in measles R₀ estimates across settings and methods; the textbook 12–18 range applies to naive (unvaccinated) populations and a different estimation context.

Treat this as a demonstration of the calibration workflow on real data applied to this specific community setting. A defensible estimate for a different context needs its own population susceptibility, data, and generation-interval model.

## Intervention

`mmr_vaccination` represents an emergency vaccination campaign during the outbreak. It scales the calibrated β by a transmission-reduction factor (adherence × efficacy), following standard population-level campaign modelling (Keeling & Rohani 2008, ch. 7; Vynnycky & White 2010). This is separate from the baseline population immunity encoded in `prepare_initial_state()`. Every run executes twice — with and without the intervention — for comparison.

## Not for

Real-world forecasting or a universally-applicable measles R₀ — this exists to show how a model calibrates itself to a recent, real case series via the dataset input, faithfully reproducing the González-Parra et al. approach.
