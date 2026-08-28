# Converting a Model with AI

You can use an AI coding assistant to convert a model written in another language — or described only in a paper — into this repository's Python/JAX framework.

A human still has to check the result. This guide gives you a three-step process that puts two or three independent reviews between the AI's first draft and your own final review.

---

## Before you start

### Expect small numerical differences

If you are converting an existing model and comparing outputs, the numbers will not match exactly. This repository uses a different numerical solver than most other modeling tools, so small differences are normal. Look for the same qualitative behavior: peak timing, peak size, final epidemic size, and growth rate.

### `print()` does not work the way you expect

JAX runs your equations in two stages. First it reads through your code to learn its shape, and only later does it plug in real numbers. A normal `print()` runs during that first stage, so it prints a placeholder instead of the value you wanted.

Use JAX's own print instead. It waits until the real numbers are there:

```python
jax.debug.print("infected = {x}", x=infected)
```

---

## The three-step process

| Step | What you do | Why |
| --- | --- | --- |
| **1** | Convert the model with one AI | Produces the first draft |
| **2** | Audit that draft with a **second** AI in a **fresh chat**. You can run this audit once, or run it twice with two different AIs | Catches errors the first AI cannot see in its own work |
| **3** | *(Optional)* If you ran the audit twice, have a third AI compare the two results | Resolves disagreements between reviewers |

This approach is described in the Wall Street Journal article [Yes, AI Can Make Mistakes. AI Can Find Them, Too.](https://www.wsj.com/tech/ai/yes-ai-can-make-mistakes-ai-can-find-them-too-6d1ad2c1)

---

## Step 1 — Convert the model

### Give the AI access to the framework

Choose one:

- Use an AI built into your code editor, with this repository open, **or**
- Point the AI at the public repository so it can read the docs and example models: <https://github.com/WHO-Collaboratory/pandemic-simulator-compartment>

### Fill in your own details

The prompt below is a template. Replace every `[BRACKETED]` placeholder before you send it:

| Placeholder | What to put there |
| --- | --- |
| `[SOURCE]` | A link to your paper, or a description of where your model lives (a file path, a GitHub repo, an attached PDF, a set of equations) |
| `[MODEL_NAME]` | A short lowercase name for your model directory, words separated by underscores — for example `ebola_seihfr_burial` |

If your model has no published code, say so in the prompt. Ask the AI to write code that matches the framework and represents the paper's equations faithfully.

> **Example.** The walkthrough in this guide converts the model in [Understanding the dynamics of Ebola epidemics](https://pmc.ncbi.nlm.nih.gov/articles/PMC2870608/). That paper publishes no code, so the AI was asked to derive the implementation from the equations. The result is in [compartment/models/ebola_seihfr_burial_legrand_model/](https://github.com/WHO-Collaboratory/pandemic-simulator-compartment/tree/main/compartment/models/ebola_seihfr_burial_legrand_model).

### The prompt

````text
You are an epidemiologist/modeler who wants to upload a model to the Pandemic
Simulator platform so users can run it and explore different scenarios. Convert
the model in this source into this repository's Python/JAX framework, formatted
so it works in the simulator: [SOURCE]

Before writing code, read:
- The guides in the docs/ directory.
- The three example models in compartment/models/ —
  example_parameter_uncertainty_custom_model,
  example_parameter_uncertainty_declarative_model, and
  example_stochastic_model. Read the other models too if you need more examples.
- .claude/MODEL_AUTHORING_REFERENCE.md.

Do NOT invent patterns the framework does not use. If
.claude/MODEL_AUTHORING_REFERENCE.md and
docs/guides/model-integration-documentation.md disagree, the latter is the
authority.

## Task 1 — Map my model before you write any code

Restate my model back to me as a structured mapping so we can catch errors
early. Produce:

- Compartments: source name -> framework compartment id/label, and which ones
  are infective=True (contribute to force of infection).
- Transmission edges / flows: every term that moves people between
  compartments, written as `source -> target`, with the rate variable, its
  value and unit from the source, and the matching ValueType (RATE, DAYS,
  PERCENTAGE, etc.). Say whether each force-of-infection term is frequency- or
  density-dependent.
- Parameters that are not edges (births, deaths, seasonality, and so on) and
  how you will represent them (add_parameter plus a manual _apply_flow).
- Structure: age/demographic groups, spatial or travel coupling, stochasticity
  (STOCHASTIC / SOLVER), and any custom _total accumulation.
- Interventions, mapped to schema.add_intervention(target_rates=[...]).
- Initial conditions, and how they map to prepare_initial_state(),
  get_initial_population() (required when there is no compartment literally
  named `I`), and the config.

List every place my source was ambiguous or where you had to assume something,
under a heading "Assumptions / open questions". Do not stop for trivial
details, but ask me before making any call that changes the model's dynamics.

## Task 2 — Scaffold, then write the files

Use the repo's scaffold command rather than copying files by hand. The command
is documented in docs/guides/model-integration-documentation.md.

Then fill in:
- compartment/models/[MODEL_NAME]_model/model.py — define_parameters(),
  __init__() (usually just super().__init__(config) plus model-specific
  fields), prepare_initial_state(), equation(), and get_initial_population()
  when the model has no compartment literally named `I` (otherwise config
  validation raises KeyError: 'I' — see model-integration-documentation.md and
  example_stochastic_model).
- main.py — a thin drive_simulation() wrapper (copy the standard one).
- __init__.py — empty package marker.
- example-config.json — generate it from the schema, then fill in realistic
  values from my source:
      python -m compartment.generate_artifact <DISEASE_TYPE> --example-config \
        --config-output compartment/models/[MODEL_NAME]_model/example-config.json

Follow the authoring recipe order in .claude/MODEL_AUTHORING_REFERENCE.md for
define_parameters(). Use schema edges and _compute_equations() for standard
flows; only drop to a manual _apply_flow() for spatial or age-stratified force
of infection, births and deaths, or multi-rate force of infection (see that
file's Pitfalls section).

Translate the math faithfully: match the source equations term for term,
preserve units (do not pre-divide DAYS or PERCENTAGE values), and never
hardcode compartment order — always use
jnp.stack([derivs[c] for c in self.compartment_list]).

## Task 3 — Check your own work (do not skip)

Verify and report actual results. Do not simply claim success.

1. Imports and discovery:
       python -c "import compartment.models.[MODEL_NAME]_model.model"
       python -m compartment.generate_artifact --list
   Confirm my DISEASE_TYPE appears in the list.

2. Runs end to end locally:
       python -m compartment.models.[MODEL_NAME]_model.main --mode local \
         --config_file compartment/models/[MODEL_NAME]_model/example-config.json \
         --output_file results/[MODEL_NAME].json

3. Smoke tests:
       python -m pytest tests/test_smoke.py -v -m integration -k [MODEL_NAME]

4. Sanity-check the dynamics against my source: population is conserved (apart
   from births and deaths), no negative or NaN compartments, and the
   qualitative behavior (peak timing, final sizes, R0-driven growth) matches
   the original. Compare against any numbers or plots I gave you. Report
   concrete output values, not "it works".

If anything fails, debug using the "Common fix-it flows" section of
.claude/MODEL_AUTHORING_REFERENCE.md and iterate until it passes. Then tell me
what was wrong and how you fixed it.

## Task 4 — Explain what you did

Give me a short summary covering:
- The final compartment / edge / parameter mapping.
- Every assumption you made, and anywhere the translation is approximate or my
  source was ambiguous.
- Any feature of my source you could NOT represent faithfully, and why.
- The exact commands I can run to reproduce your verification.

## Ground rules

- Accuracy over completeness. If you are unsure how a term maps, flag it. Do
  not guess silently.
- Read the real framework files when in doubt. The code changes faster than the
  docs.
- Do not edit MODEL_REGISTRY or validation/__init__.py, and do not hand-write a
  Pydantic config. Those are generated automatically.
- Keep me in the loop on any decision that changes the model's behavior.
````

---

## Step 2 — Audit the code with a second AI

Open a **new chat window** with no history from Step 1. An AI that just wrote the code cannot review it with fresh eyes. You can use the same tool or a different one, as long as the context is empty.

### Fill in your own details

| Placeholder | What to put there |
| --- | --- |
| `[SOURCE]` | The same paper link or model description you used in Step 1 |
| `[MODEL_DIRECTORY]` | The directory the code was written into, for example `ebola_seihfr_burial_legrand_model` |
| `[YOUR_DECISIONS]` | Every deliberate choice you or the AI made that departs from the source. List them plainly so the auditor does not report them as bugs |

Examples of decisions worth listing under `[YOUR_DECISIONS]`:

- *"This is a deterministic ODE, not the paper's stochastic Gillespie simulation. This is the standard mean-field limit used by most secondary literature reproducing this paper."*
- *"The example config uses the DRC 1995 parameter values. The Uganda 2000 values are documented in the code but not wired into the config."*

### The prompt

````text
You are a research assistant hired to audit an implementation of the model
described in this source: [SOURCE]

The implementation is in the [MODEL_DIRECTORY] directory and has been adapted
to the conventions, documentation, and example-model framework of the current
repository.

The modeler made the following deliberate decisions. Treat these as intended,
not as defects:
[YOUR_DECISIONS]

Conduct the audit in two parts.

## Part 1 — Source-to-model fidelity

Identify every substantive facet of the model that the implementation
represents: assumptions, compartments, parameters, equations, transitions,
initial conditions, interventions, outputs, and any other modeled behavior.

Build a table with one row per facet and these columns:

| Model facet or assertion | Assessment (true / false / ambiguous / unsupported) | Exact quotation from the source | Location in the source (section, equation, table, figure, page) | Notes and interpretation |

Evaluate every assertion against the source itself.
- Use "ambiguous" when the source gives too little detail to reach a
  definitive interpretation.
- Use "unsupported" when you can find no support at all.
- Never infer support without labeling the inference clearly.

## Part 2 — Code-implementation audit

Review the whole relevant codebase and decide whether the implementation
accurately executes every concept in the first table. Include repository-level
framework code wherever it affects the model's behavior.

Verify that:
- Every function called is defined, either in the model implementation or
  elsewhere in the repository.
- Imports, references, parameters, and dependencies all resolve.
- Each function is appropriate for its purpose.
- Equations, state transitions, parameter mappings, initial conditions, units,
  numerical methods, interventions, and outputs match the source.
- The implementation follows the repository's conventions without changing the
  model's intended behavior.
- Documentation and examples describe the model that was actually implemented.

Build a second table linking every concept in the first table to the code that
implements it:

| Model facet or assertion | Code location (file path plus line number or symbol) | Assessment (true / false / ambiguous / unsupported) | Explanation and notes | Alternative implementation approaches |

In the last column, if the current approach is effectively the only reasonable
one, say so explicitly.

- Use "true" only when you can verify the implementation is correct.
- Use "false" when it conflicts with the source or does not execute as
  intended.
- Use "ambiguous" when the available evidence cannot settle correctness.
- Use "unsupported" when the concept has no implementation, or the dependency
  cannot be found.

## Conclusion

Finish with:
- A prioritized list of discrepancies and defects.
- Any missing or unsupported model features.
- Undefined, unresolved, misused, or inappropriate functions.
- Documentation or testing gaps.
- Specific recommended corrections, with file and symbol references.

Be exhaustive, evidence-based, and precise. Clearly separate direct evidence
from interpretation. Do not treat repository conventions as evidence that the
scientific model was implemented correctly.
````

---

## Step 3 (optional) — Compare two audits

Run Step 2 twice, each time in a fresh chat, ideally with two different AI tools. Then give both audit reports to a third AI:

````text
Below are two independent audits of the same model implementation. The source
model is here: [SOURCE]

Identify every point where the two audits disagree. For each disagreement, do
your own research against the source and the code, then say which audit is
correct and why. If neither is correct, say what the right answer is.

Audit A:
[PASTE AUDIT A]

Audit B:
[PASTE AUDIT B]
````

---

## After the audits — update your model

The audits give you a list of findings. They do not fix anything. Read through the findings and make the changes to your model by hand.

1. **Fix the real errors.** Anything marked `false` — an equation that does not match your source, a wrong unit, a parameter wired to the wrong compartment.
2. **Resolve the `ambiguous` items.** These are usually places where your source was unclear. You are the modeler, so this is your call to make.
3. **Decide on the `unsupported` items.** Either implement the missing feature, or write down why you left it out.
4. **Ignore findings that describe a choice you made on purpose** — but check that the choice is documented in the model directory so the next reader does not have to rediscover it.

You are the last reviewer. Once your model is complete, continue with the next steps to test it, documented in [model-integration-documentation.md](./model-integration-documentation.md).

---

**Last Updated:** August 24, 2026
