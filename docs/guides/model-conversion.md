Model conversion with AI
Differences between models in different languages
When converting a model from another language to Python JAX, you may want to compare the outputs to verify that the conversion was successful. Small differences in the results are expected, as the Python JAX implementation may use a different numerical solver than the original model.
Printing code
Due to the way JAX transforms, compiles and optimizes code, Python variables may not hold normal values and instead be abstract representations that will exist later when the compiled function actually runs. 
Because of this, a regular Python print() statement may not show the values you expect. To inspect values in transformed JAX code, use JAX-specific debugging tools such as jax.debug.print().
Model Conversion Process
While a human in the loop is still necessary to ensure that the model is correct and implemented properly, we recommend using a 3 step process that will ensure that the code generated has been reviewed thoroughly before the modeler reviews/verifies it. This method is detailed in a The Wall Street Journal article Yes, AI Can Make Mistakes. AI Can Find Them, Too. 

Step 1: Convert the model using AI
Step 2: Use another AI (or 2) to check over the work that was done in Step 1
Step 3 (optional): Compare AI fact checking outputs
Step 1: Model conversion
You can either use an AI that is built into your code editor or share the public Github repository with the AI that you are using so it can review the relevant documents and see the formatting needed for models to work in the Pandemic Simulator. The public repository is here: https://github.com/WHO-Collaboratory/pandemic-simulator-compartment 

Below is an example of model conversion that was done using the model detailed in Understanding the dynamics of Ebola epidemics. This model did not have a link to any code so I asked the AI to devise code that uses the formatting needed in the repository and represents the model accurately.

Copy and paste the script below into your AI of choice:

Script

You are an epidemiologist/modeler who wants to upload code to this Pandemic Simulator platform for users to be able to run your model and see different scenarios from it. The model in this paper needs to be converted to the repository’s Python/JAX framework and formatted properly to be placed in the simulator in an easy and understandable way: https://pmc.ncbi.nlm.nih.gov/articles/PMC2870608/. 

Read through the guides in the docs directory and look at the 3 example models in the compartment/models directory for formatting (example_parameter_uncertainty_custom_model, example_parameter_uncertainty_declarative_model, example_stochastic_model). You can also read through the other models if needed. Do NOT invent patterns the framework doesn't use. Also look at the .claude file for info. If there is a conflict between the information in it and the information in docs/guides/model-integration-documentation.md the latter is the authority.

Step 1 — Analyze my source model and produce a mapping table
Before writing any Python, restate my model back to me as a structured mapping so we
can catch errors early. Produce:
- **Compartments:** source name → framework compartment id/label, and which ones are
  `infective=True` (contribute to force of infection).
- **Transmission edges / flows:** every term that moves people between compartments,
  written as `source -> target`, the rate variable, its source value + unit, and the
  matching `ValueType` (`RATE`, `DAYS`, `PERCENTAGE`, etc.). Note frequency- vs
  density-dependent transmission for each FOI term.
- **Parameters that aren't edges** (births, deaths, seasonality, etc.) → how you'll
  represent them (`add_parameter` + manual `_apply_flow`).
- **Structure:** age/demographic groups, spatial/travel coupling, stochasticity
  (`STOCHASTIC`/`SOLVER`), and any custom `_total` accumulation.
- **Interventions:** map to `schema.add_intervention(target_rates=[...])`.
- **Initial conditions** → how they map to `prepare_initial_state` and the config.

Call out, explicitly, every place where my source was ambiguous or where you had to
make an assumption. List these as "Assumptions / open questions" and prefer asking me
over guessing on anything that changes the dynamics. Wait for nothing trivial, but
surface the risky calls.

### Step 2 — Scaffold, then write the files
Use the repo's scaffold command rather than copying and pasting files, this code is found in the model-integration-documentation.md file.

Then fill in / create the necessary files:
- `compartment/models/<name>_jax_model/model.py` — `define_parameters()` +
  `__init__()` (usually just `super().__init__(config)` plus model-specific fields) +
  `prepare_initial_state()` + `derivative()`.
- `main.py` — thin `drive_simulation()` wrapper (copy the standard one).
- `__init__.py` — empty package marker.
- `example-config.json` — generate it from the schema with
  `python -m compartment.generate_artifact <DISEASE_TYPE> --example-config
  --config-output compartment/models/<name>_jax_model/example-config.json`,
  then fill in realistic values from my source model.

Follow the authoring recipe order from the `.claude` file for `define_parameters()`.
Lean on `_compute_derivatives()` for standard edges; only drop to manual `_apply_flow()`
for spatial/age-stratified FOI, births/deaths, or multi-rate FOI (see the Pitfalls
section). Translate the math faithfully — match the source equations term for term,
preserve units (don't pre-divide `DAYS`/`PERCENTAGE` values), and never hardcode
compartment order (always `jnp.stack([derivs[c] for c in self.compartment_list])`).

### Step 3 — Check your own work (do not skip)
After writing, verify and report results — do not just claim success:
1. **Imports / discovery:** `python -c "import compartment.models.<name>_jax_model.model"`
   then `python -m compartment.generate_artifact --list` and confirm my `DISEASE_TYPE`
   appears.
2. **Runs end-to-end locally:**
   `python -m compartment.models.<name>_jax_model.main --mode local
   --config_file compartment/models/<name>_jax_model/example-config.json
   --output_file results/<name>.json`
3. **Smoke tests:**
   `python -m pytest tests/test_smoke.py -v -m integration -k <name>`
4. **Sanity-check the dynamics against my source:** population is conserved (modulo
   births/deaths), no negative or NaN compartments, and the qualitative behavior
   (peak timing, final sizes, R0-driven growth) is consistent with the original. If
   you can, compare against any numbers/plots I provided. Report concrete output
   values, not just "it works."

If anything fails, debug using the "Common fix-it flows" in the `.claude` file and
iterate until it passes — then report what was wrong and how you fixed it.

### Step 4 — Explain what you did
Give me a concise summary:
- The compartment/edge/parameter mapping you settled on (final version).
- Every assumption you made and anywhere the translation is approximate or where my
  source was ambiguous.
- Any source feature you could NOT faithfully represent in this framework and why.
- The exact commands I can run to reproduce your verification.

## Ground rules
- Accuracy over completeness: if you're unsure how a term maps, flag it — do not
  silently guess.
- Read the real framework files when in doubt; the `.claude` reference notes the code
  changes faster than the docs.
- Don't edit `MODEL_REGISTRY`, `validation/__init__.py`, or hand-write a Pydantic
  config — those are auto-derived.
- Keep me in the loop on decisions that change the model's behavior.


Step 2: Reviewing code with AI
Next, have an AI review the code that the other AI created. Make sure when you do this that you open a new chat window that will not have the chat history that was used to create the model files. You can use the same AI agent you used to generate the code or a different one
but be sure you have a new window without the previous context.

Script
You are a research assistant hired to audit an implementation of the model described in this paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC2870608/.
The implementation is located in the ebola_seihfr_burial_legrand_model directory and has been adapted to the conventions, documentation, and example-model framework of the current repository. Note that the modeler made the following decisions: this model is a deterministic ODE, not the paper's literal stochastic Gillespie simulation — this is the standard mean-field limit used by most secondary literature reproducing this paper. Additionally, DRC 1995 defaults; Uganda 2000 values are documented but not wired into the example config.
Conduct the audit in two parts:
Paper-to-model fidelity
Identify every substantive facet of the model represented by the implementation, including its assumptions, compartments, parameters, equations, transitions, initial conditions, interventions, outputs, and other modeled behavior.
Create a table with one row per facet and the following columns:
Model facet or assertion
Paper assessment: true, false, ambiguous, or unsupported
Exact quotation from the paper supporting the assessment
Paper location, such as the section, equation, table, figure, or page
Notes and interpretation
Every assertion must be evaluated against the paper. Use ambiguous when the paper does not provide enough detail to reach a definitive interpretation, and unsupported when no support can be found. Do not infer support without clearly labeling the inference.
Code-implementation audit
Review the entire relevant codebase and determine whether the implementation accurately executes every concept identified in the first table. Include repository-level framework code whenever it affects the model’s behavior.
Verify that:
Every called function is defined either within the model implementation or elsewhere in the repository.
Imports, references, parameters, and dependencies resolve correctly.
Each function is appropriate for its intended purpose.
Equations, state transitions, parameter mappings, initial conditions, units, numerical methods, interventions, and outputs match the paper.
The implementation follows the repository’s established framework and conventions without changing the model’s intended behavior.
Documentation and examples accurately describe the implemented model.
Create a second table connecting every concept in the first table to the code that implements it. Include the following columns:
Model facet or assertion
Code location, including file path and line number or symbol
Implementation assessment: true, false, ambiguous, or unsupported
Explanation and notes
Alternative implementation approaches; if the current approach is effectively the only reasonable method, state that explicitly
Use true only when the implementation can be verified as correct. Use false when it conflicts with the paper or does not execute as intended. Use ambiguous when correctness cannot be determined from the available evidence. Use unsupported when the concept has no corresponding implementation or the relevant dependency cannot be found.
Conclude with:
A prioritized list of discrepancies and defects
Any missing or unsupported model features
Undefined, unresolved, misused, or inappropriate functions
Documentation or testing gaps
Specific recommended corrections, with file and symbol references
Be exhaustive, evidence-based, and precise. Clearly distinguish direct evidence from interpretation, and do not treat repository conventions as evidence that the scientific model was implemented correctly.
Step 3 (optional): Comparing AI fact checking outputs
If you do Step 2 with 2 AIs (starting fresh each time), give their results to a 3rd fact checker AI and ask it to identify anywhere the 2 previous checkers disagreed and ask it to do its own tie-breaking research. 
