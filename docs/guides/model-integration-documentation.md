# Model Integration Documentation

## Overview

This documentation describes the steps required to integrate a compartmental model into the Pandemic Simulator. Once integrated, users can run the model in the Pandemic Simulator and modify the parameters you define to explore different scenarios through the user interface.

Two roles recur throughout this document: a **user** runs a model from the Pandemic Simulator's front end, choosing location, parameter values, and interventions through the interface, while a **modeler** writes the model code that runs in the backend so that users can run it. One person can be both; this guide is written for modelers.

The Pandemic Simulator supports **deterministic** compartmental models, models with **parameter uncertainty** (Latin Hypercube Sampling), and **stochastic** models. After running a simulation, users can compare results across all models, evaluate intervention scenarios alongside control (no-intervention) simulations, and view AI-generated interpretations of the results.

### How to read the commands in this guide

Terminal commands are shown once when they are identical on every platform. When they differ, both forms are given:

**macOS / Linux**
```shell
example --with \
    --line-continuations
```

**Windows Command Prompt**
```shell
example --with --everything-on-one-line
```

The only routine differences are line continuations (`\` on macOS/Linux, not supported in Command Prompt) and path separators (`/` vs `\`).

### Example models

This guide follows three models that ship with the repository.

| Directory | `disease_type` | What it demonstrates |
| :---- | :---- | :---- |
| `compartment/models/example_parameter_uncertainty_declarative_model` | `example_parameter_uncertainty_declarative` | The **default path**. A SIR model where the framework generates the equations from declared transmission edges. Includes age demographics, an intervention, and parameter uncertainty. |
| `compartment/models/example_parameter_uncertainty_custom_model` | `example_parameter_uncertainty_custom` | The same SIR model with the equations **written by hand**, plus a custom intervention that ramps up and down instead of switching on and off. |
| `compartment/models/example_stochastic_model` | `example_stochastic` | A **stochastic** SIR model (tau-leaping, Euler integration, multiple trajectories) with split asymptomatic / symptomatic infectious compartments. |

### What a local run represents

Your model code is identical whether it runs locally on your own machine or remotely in the Pandemic Simulator. The only thing that changes is where its inputs come from.

Running locally, you hand the model a JSON file. In the Simulator, the frontend collects the user's choices and the backend supplies the population and geography — then passes them to your model in that same shape. So `example-config.json` is a stand-in for a filled-in UI form plus the backend data behind it, and a model that runs locally will run in the Simulator.

Think of your model directory as having two halves: what the user can touch, and what only you control. The rule of thumb: **anything you declare on the `schema` becomes something a user can see or change. Anything you write in `equation()` is yours alone.** For example, if the modeler changes a default value when adding a transmission edge via `schema.add_transmission_parameter()` this shows that new default value in the UI for every future user; rewriting `equation()` changes the model's behavior without changing the UI at all.

| Model component | Location | Representation in the UI |
| :---- | :---- | :---- |
| `schema.set_model_info()` | `model.py` | Your model's name and description in the model picker. |
| `schema.set_model_metadata()` | `model.py` | Authorship, license, assumptions, and similar context on the Simulation Configuration page. |
| `schema.add_compartment()` | `model.py` | The compartments offered for the model, which users can select or deselect. |
| `schema.add_transmission_parameter()` | `model.py` | The arrows between compartments, each with a control whose default, minimum, and maximum are the ones you declared. |
| `schema.add_parameter()` | `model.py` | An extra input field, for anything that isn't a flow between compartments. |
| `schema.add_intervention()` | `model.py` | An intervention the user can switch on and give dates, adherence, and a transmission reduction. |
| `schema.add_demographic_group()` | `model.py` | The age breakdown available for the run. |
| `equation()` | `model.py` | **Nothing.** It runs only in the backend. No UI user can see or change it. |
| `example-config.json` | standalone file | The user's form selections, plus backend data — populations from WorldPop, admin zone coordinates from OpenStreetMap. Editing it locally is you standing in for the user. |
| `model.md` | standalone file | The "You should know" panel on the results page. |
| `main.py` | standalone file | The file that starts a run. Locally you launch it yourself from the terminal; in the Simulator the platform launches it for you when a user starts a simulation. |

A local run also produces the same results JSON the Simulator charts, which is why the [results viewer](#visualize-model-results) looks like the results page.

---

## Table of Contents

- [Technical API](#technical-api)
- [Initial environment setup](#initial-environment-setup)
  - [Cloning the repository](#cloning-the-repository)
  - [Installing Python](#installing-python)
  - [Installing uv](#installing-uv)
  - [Create a virtual environment](#create-a-virtual-environment)
  - [Git workflow](#git-workflow)
- [Add a new model](#add-a-new-model)
  - [Choose a unique name](#choose-a-unique-name)
  - [Run the scaffold](#run-the-scaffold)
  - [Your model is found automatically](#your-model-is-found-automatically)
  - [Copy an existing model instead](#copy-an-existing-model-instead)
- [Writing the model](#writing-the-model)
  - [How the framework uses your class](#how-the-framework-uses-your-class)
  - [model.py](#modelpy)
    - [Automated vs. manual equations](#automated-vs-manual-equations)
    - [Optional functions to override](#optional-functions-to-override)
    - [Built-in base class functions](#built-in-base-class-functions)
    - [Stochastic models](#stochastic-models)
    - [Parameter uncertainty](#parameter-uncertainty)
  - [main.py](#mainpy)
  - [Additional files](#additional-files)
- [Documenting your model](#documenting-your-model)
  - [model.md](#modelmd)
  - [Model metadata](#model-metadata)
  - [Docstrings for the technical API](#docstrings-for-the-technical-api)
- [example-config.json](#example-configjson)
  - [Generating artifacts](#generating-artifacts)
- [Install additional packages](#install-additional-packages)
- [Run the model locally](#run-the-model-locally)
  - [Visualize model results](#visualize-model-results)
- [Run tests](#run-tests)
- [Submit a pull request](#submit-a-pull-request)
- [Approve a model](#approve-a-model)
  - [Review the pull request](#1-review-the-pull-request)
  - [Tag a release](#2-tag-a-release)
  - [Watch the pipeline](#3-watch-the-pipeline)
  - [Review the model in Model Approvals](#4-review-the-model-in-model-approvals)
  - [Simulate before publishing](#5-simulate-before-publishing)
  - [Publish](#6-publish)
  - [Unpublishing and archiving](#unpublishing-and-archiving)

---

## Technical API

The API reference site is generated from the source code itself, so it lists every argument, default, and return value of the framework's classes and methods, along with a page for each model already in the repository. This guide walks you through integrating a model from start to finish; the technical API is the lookup you reach for while writing the code, when you need the exact signature of a call.

<https://who-collaboratory.github.io/pandemic-simulator-compartment/>

---

## Initial environment setup

Complete this section once. Later sessions only need the virtual environment activation step.

### Cloning the repository

Repository: <https://github.com/WHO-Collaboratory/pandemic-simulator-compartment>

Confirm Git is installed (download from <https://git-scm.com/downloads> if not):

```shell
git --version
```

Move to where you want the repository stored, then clone it. HTTPS is easiest for new users; SSH avoids repeated authentication prompts.

The examples below clone into the Desktop, which this guide uses as its example location throughout. The repository can live in any directory you like — substitute your own path wherever you see `~/Desktop` (macOS/Linux) or `%USERPROFILE%\Desktop` (Windows).

**macOS / Linux**
```shell
cd ~/Desktop
git clone https://github.com/WHO-Collaboratory/pandemic-simulator-compartment.git
cd ~/Desktop/pandemic-simulator-compartment
```

**Windows Command Prompt**
```shell
cd %USERPROFILE%\Desktop
git clone https://github.com/WHO-Collaboratory/pandemic-simulator-compartment.git
cd %USERPROFILE%\Desktop\pandemic-simulator-compartment
```

> If a path contains a space, wrap it in quotation marks: `cd "%USERPROFILE%\Desktop\my folder"`.

Verify:

```shell
git status
git remote -v
```

Useful commands after cloning (`git branch -a` opens a pager — type `q` to exit):

```shell
git branch -a                              # list branches
git pull                                   # download latest changes
git checkout -b add-example-disease-model  # create a branch
git log --oneline                          # view commit history
```

> This guide uses `add-example-disease-model` as the working branch, matching the example model it builds. Name your branch after the model you are adding, and use that name wherever this guide shows `add-example-disease-model`.

**Troubleshooting**

- `Permission denied` — confirm you have repository access, are authenticated, and that your SSH key is configured if using SSH.
- `Repository not found` — confirm the URL is correct and you have permission to access it.

References: [Git docs](https://git-scm.com/doc) · [git clone](https://git-scm.com/docs/git-clone) · [Cloning a repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) · [SSH key setup](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)

### Installing Python

This project requires **Python 3.13+**. Download it from the [Python downloads page](https://www.python.org/downloads/).

- **Windows** — run the installer and **check "Add Python to PATH"** before clicking Install Now.
- **macOS** — run the `.pkg` installer and accept the defaults.
- **Linux** — use your package manager, e.g. `sudo apt update && sudo apt install python3` (Ubuntu/Debian) or `sudo dnf install python3` (Fedora).

Verify the installation:

**macOS / Linux**
```shell
python3 --version
```

**Windows Command Prompt**
```shell
python --version
```
(If `python` is not recognized, try `py --version`.)

**Troubleshooting**

- `python is not recognized as a command` — Python was installed without being added to PATH. Re-run the Windows installer and select **Add Python to PATH**.
- Multiple versions installed — list them with `py --list` (Windows) or `which python3` (macOS/Linux).

**Optional but recommended:** install [Visual Studio Code](https://code.visualstudio.com/) and its **Python** extension.

### Installing uv

`uv` is the environment and dependency manager for this project: it installs dependencies faster and matches the project's development workflow. It is a separate tool, so install it once before setting up the environment.

**macOS / Linux**
```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows Command Prompt**
```shell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

If you would rather use a package manager you already have, `pip install uv`, `brew install uv` (macOS), and `winget install --id=astral-sh.uv -e` (Windows) all work too.

Open a new terminal so the updated PATH takes effect, then verify:

```shell
uv --version
```

**Troubleshooting**

- `uv is not recognized as a command` — the installer updated your PATH, but the terminal you ran it in still has the old one. Close it and open a new one.
- Installed with `pip` while a virtual environment was active — `uv` then exists only inside that environment. Install it with one of the commands above instead, outside any environment.

Reference: [uv installation docs](https://docs.astral.sh/uv/getting-started/installation/)

### Create a virtual environment

Run these three steps from the repository root the first time you set up a clone: create the environment, activate it, then install the project's dependencies into it.

1. **Create the environment.** This makes a `.venv/` folder in the repository, isolated from your system Python.

   ```shell
   uv venv
   ```

2. **Activate it. Do this every time you open a new terminal**, not just the first time.

   **macOS / Linux**
   ```shell
   source .venv/bin/activate
   ```

   **Windows Command Prompt**
   ```shell
   .venv\Scripts\activate.bat
   ```

3. **Install the dependencies** into the activated environment. Re-run this whenever the project's dependencies change.

   ```shell
   uv sync
   ```

Confirm the right environment is active:

**macOS / Linux**
```shell
which python
```

**Windows Command Prompt**
```shell
where python
```

### Git workflow

Commit regularly as you work. A commit is a checkpoint you can return to.

```shell
git status                # see what changed
git diff                  # review the changes
git add .                 # stage everything (or: git add <filename>)
git commit -m "Add asymptomatic compartment to example model"
git log --oneline         # view your commits
```

**Tips**

- Commit early and often — small, focused commits are easier to review.
- Describe **what changed**.
- Commits stay local until you push your branch (see [Submit a pull request](#submit-a-pull-request)).

---

## Add a new model

Before integrating your model, verify it runs correctly locally.

There are two ways to start. The scaffold command creates the directory layout with a working SIR template that the modeler can alter, or you can [copy an existing model](#copy-an-existing-model-instead) and rename it. Either way, start by choosing the name the model will live under.

### Choose a unique name

The scaffold appends `_model` to the name you pass, so `example_parameter_uncertainty_declarative` becomes `compartment/models/example_parameter_uncertainty_declarative_model/`. Class names and the code that instantiates them are generated to match.

**That directory name must be unique.** No other directory in `compartment/models/` can already use it. The scaffold will not overwrite an existing directory: it stops before creating anything and prints

```
Error: 'pandemic-simulator-compartment/compartment/models/example_parameter_uncertainty_declarative_model' already exists.
```

Neither the model directory name nor the disease type can contain spaces.

### Run the scaffold

```
python -m compartment.new_model <model_directory_name>
    [--label "<display_label>"]
    [--disease-type <MODEL_IDENTIFIER>]
    [--description "<model_description>"]
```

Replace the values inside angle brackets (`< >`). Only the model directory name is required. Options in square brackets (`[ ]`) are optional — **do not type the square brackets**.

The command that produced the declarative example model:

**macOS / Linux**
```shell
python -m compartment.new_model example_parameter_uncertainty_declarative \
    --label "Example Disease with Declarative Parameter Uncertainty" \
    --disease-type example_parameter_uncertainty_declarative \
    --description "A SIR model for an example disease with declarative parameter uncertainty"
```

**Windows Command Prompt**
```shell
python -m compartment.new_model example_parameter_uncertainty_declarative --label "Example Disease with Declarative Parameter Uncertainty" --disease-type example_parameter_uncertainty_declarative --description "A SIR model for an example disease with declarative parameter uncertainty"
```

> That directory already exists in the repository, so running the command verbatim fails the uniqueness rule above. Substitute your own model name, or add `--dry-run` to preview the output without writing files:
>
> ```shell
> python -m compartment.new_model example_parameter_uncertainty_declarative --dry-run
> ```

**Files created**

- **`__init__.py`** — leave empty. It marks the directory as a Python package.
- **`model.py`** — the disease class: parameters, input formatting, and the equation function for a minimal SIR model.
- **`main.py`** — loads the model and executes the simulation.
- **`model.md`** — a template for the write-up users see in the "You should know" section of the results page. See [Documenting your model](#documenting-your-model).
- **`example-config.json`** — configuration for running the simulation locally.

`label` and `description` are shown to users in the Pandemic Simulator UI. 

**Verification run (optional).** Before changing any code, run the scaffolded model to confirm your environment works. Output files land in `results/`, which is in `.gitignore`.

**macOS / Linux**
```shell
python -m compartment.models.example_parameter_uncertainty_declarative_model.main \
    --mode local \
    --config_file compartment/models/example_parameter_uncertainty_declarative_model/example-config.json \
    --output_file results/example-declarative-test.json
```

**Windows Command Prompt**
```shell
python -m compartment.models.example_parameter_uncertainty_declarative_model.main --mode local --config_file compartment\models\example_parameter_uncertainty_declarative_model\example-config.json --output_file results\example-declarative-test.json
```

### Your model is found automatically

With the directory created, the framework already knows about your model. There is no list to sign up for and no shared file to edit: each time it starts, it looks inside every folder in `compartment/models/`, reads the `model.py` it finds there, and picks up whatever model is defined inside. Creating the folder is the whole of "registering" it. Because the folder simply being there is all that matters, this works the same however the folder got there — whether the scaffold made it or you copied and pasted an existing model's folder.

Two things make a model recognizable, and you get both from the scaffold or from any model you copy: it builds on the framework's `Model` class, and it gives itself a `disease_type` in `set_model_info()`. Nothing else is needed, and the same automatic discovery happens when your model is published to the Pandemic Simulator.

A copied folder is found quickly, so give it its own `disease_type` right away: a name claimed by two models no longer identifies either one, and configs that ask for it stop working — for the original as well as the copy. [Copy an existing model instead](#copy-an-existing-model-instead) lists everything a copy needs renamed.

To see all models the framework currently knows about:

```shell
python -m compartment.generate_artifact --list
```

If your `disease_type` appears in that list, the framework has your model. If it is missing, the cause is nearly always one of two things: no `disease_type` was set, or something in `model.py` stops the file from loading. A model that fails to load is passed over quietly rather than flagged, so it is worth running the command above after your first round of edits.

### Copy an existing model instead

Scaffolding always produces the same minimal SIR template. When an existing model is already close to what you want — the same compartment structure with different parameters, or a variation on a model you have written before — copying its directory is often less work. The trade-off is that a copy comes with the original's names baked into it, and until you change all of them the copy identifies itself as the model you copied.

Copy the directory first:

**macOS / Linux**
```shell
cp -r compartment/models/example_parameter_uncertainty_declarative_model \
    compartment/models/my_disease_model
```

**Windows Command Prompt**
```shell
xcopy /E /I compartment\models\example_parameter_uncertainty_declarative_model compartment\models\my_disease_model
```

The new directory name follows the same rule as scaffolding: it must be unique within `compartment/models/`, and the `_model` suffix is the convention. Then work through every name the copy inherited.

| Where | What to change |
| :---- | :---- |
| Directory name | Your new, unique name. Nothing detects a duplicate for you here — you chose the name when you copied. |
| `model.py` | The class name (`ExampleParameterUncertaintyDeclarativeModel`), and the `disease_type`, `label`, and `description` passed to `set_model_info()`. Update the class docstring too — it becomes your model's description on the API reference site. |
| `main.py` | Three places: the module path **and** class name in the `import` line, `model_class=` inside `lambda_handler()`, and `model_class=` in the `drive_simulation()` call at the bottom. |
| `example-config.json` | `Disease.disease_type`, which has to match the value in `set_model_info()`. |
| `model.md` | The title and the whole write-up, which still describes the original model. |
| `artifacts/` | Delete the copied JSON. It is named after the old disease type and describes the old schema; regenerate it once your changes are in place. |
| `__pycache__` | Safe to delete. It is compiled output from the original and is rebuilt on the next run. |
| `__init__.py` | Nothing. It stays empty. |

In the example model that is six occurrences across three files. To catch anything missed, search the new directory for the old names:

```shell
grep -rn "ExampleParameterUncertaintyDeclarative\|example_parameter_uncertainty_declarative" compartment/models/my_disease_model
```

Finally, confirm the registry picked up the copy as its own model. Both disease types should be listed, the old one unchanged:

```shell
python -m compartment.generate_artifact --list
```

Then run it locally, the same way as a scaffolded model.

---

## Writing the model

The template inherits most behavior from the base `Model` class. **Do not modify the base class.** Customize by overriding methods in your own `model.py`.

### How the framework uses your class

If you are new to object-oriented Python, this section explains the handful of ideas the framework relies on. Everything here uses the example models, so you can open the files and follow along. If these ideas are already familiar, skip to [model.py](#modelpy).

#### Classes and instances

A **class** is a blueprint. `ExampleStochasticModel` is a class: it describes what a stochastic example-disease model *is* and what it can *do*, but it isn't running anything by itself.

An **instance** is one actual model built from that blueprint, with real numbers in it. You never build one yourself — the framework does it. It calls your class once with the loaded config, then copies that instance and empties the copy's interventions to produce the control run. That's where the two runs in every output file come from: two instances of your one class.

Inside the class, `self` means "this particular instance." The two runs differ by a single attribute: the framework empties `self.intervention_dict` on the control instance. So when `equation()` applies interventions, the first instance finds `my_intervention` and scales `beta` down, while the control finds nothing and leaves `beta` alone. Same class, same code, two different curves — because `self` points at a different instance each time.

#### Inheritance

**Inheritance** means one class starts with everything another class already has. You write the differences, not the whole thing.

The parentheses in a class definition name the class you are inheriting from:

```python
class ExampleParameterUncertaintyDeclarativeModel(Model):
```

Read that as: "the declarative example model **is a** `Model`, and starts out with everything `Model` can do." The class that gives is called the **parent** (or base class); the class that receives is the **child** (or subclass).

This is why the example model files are so short. `Model` already knows how to load a config, build the population matrix, apply interventions, run the solver, and format the output. `example_parameter_uncertainty_declarative_model/model.py` is about 160 lines because it only has to supply what's specific to this disease — its compartments, its rates, its equation.

All three example models inherit directly from `Model`:

```
Model                                          (the framework's base class)
  ├─ ExampleParameterUncertaintyDeclarativeModel
  ├─ ExampleParameterUncertaintyCustomModel
  └─ ExampleStochasticModel
```

#### Overriding: changing inherited behavior

**Overriding** means writing a method in your class with the same name as one you inherited. Yours is the one that runs. The parent's version still exists, but your version takes precedence for your class.

This is how you customize the framework without touching it. The base `Model` automatically creates a cumulative `<target>_total` compartment for every transmission edge target. `example_stochastic_model` doesn't want that, because its two infectious compartments should share one combined total. So it overrides the method and does nothing:

```python
@classmethod
def _add_total_compartments(cls, schema):
    pass  # I_total and R_total are declared in define_parameters() instead
```

`pass` means "do nothing." Because this version replaces the inherited one, the automatic totals are never created, and the model declares its own instead.

Overriding is also why the guide keeps saying **do not rename these functions**. The framework looks for methods by name — it calls `equation()`, `define_parameters()`, and `prepare_initial_state()` on whatever class you give it. Name yours `equation`, and yours runs. Name it `my_equation`, and the framework never finds it, so the parent's version (or an error) is what you get.

Note the direction of control here: you almost never call these methods yourself. You *provide* them, and the framework calls them at the right moment — `define_parameters()` once when building the schema, `equation()` at every step of the solver. Your job is to fill in the blanks the framework asks for.

#### `super()`: extending instead of replacing

Sometimes you don't want to replace the inherited behavior — you want to run it *and then* add to it. `super()` means "the parent's version of this method."

Almost every model's `__init__` starts this way:

```python
def __init__(self, config):
    super().__init__(config)
    # Model-specific initialisation goes here.
```

The first line runs `Model.__init__`, which does all the shared setup — loading the config, building the population matrix, wiring up interventions. Only after that does your own code run. This is why the attributes listed under [Optional functions to override](#optional-functions-to-override) (`self.population_matrix`, `self.interventions`, and so on) are available *after* the `super()` call and not before: the parent is what creates them.

`example_stochastic_model` uses exactly this pattern to add one thing of its own — a random number generator seed:

```python
def __init__(self, config):
    super().__init__(config)          # let Model do all the standard setup
    seed = config.get("seed") if isinstance(config, dict) else None
    if seed is None:
        seed = int(time.time() * 1000) % (2**31)
    self._key = jax.random.PRNGKey(seed)   # then add what only this model needs
```

Leave out the `super()` call and you replace the parent's behavior entirely instead of building on it — none of the shared setup would run, so `self.population_matrix` and `self.interventions` would never be created. A few models in the repository (dengue, for example) deliberately skip `super().__init__()` to do their own setup, and they have to set `self.compartment_list` by hand.

#### Class attributes: settings the framework reads

Most of what you write lives inside methods. A few things are set directly on the class instead, and apply to every instance of it. These are **class attributes**, and the framework reads them to decide how to treat your model:

```python
class ExampleStochasticModel(Model):
    STOCHASTIC = True                       # use Euler integration, run many trajectories
    COMPARTMENT_DELTA_GROUPING = {          # combine A + Sym into one "Infected" curve
        "S": ["S"],
        "I": ["A", "Sym"],
        "R": ["R"],
    }
```

They are written in capitals by convention, to signal that they are fixed settings rather than values that change during a run. Note that setting `STOCHASTIC = True` is not something you *do* — it's something you *declare*, and the framework changes its behavior in response.

#### `cls` vs. `self`, and `@classmethod`

You'll see two different first arguments in the example models, and the difference is practical.

- **`self`** — an ordinary method, called on one instance. It can read that run's data. `equation(self, y, t, p)` is an instance method, because it needs `self.travel_matrix`, `self.interventions`, and the rest of this run's state.
- **`cls`** with a `@classmethod` decorator above it — a method called on the class itself, before any instance exists. `define_parameters(cls, schema)` is a classmethod because the framework needs your model's compartments and parameters *in order to* validate a config and build an instance. There is no instance yet, so there is no `self` to use.

This is also why `define_parameters` can't read `self.beta`: at the time it runs, no config has been loaded and no rates exist. It only *declares* that `beta` exists and what its bounds are.

### model.py

Implement these two functions at a minimum. **Do not rename them** — the framework calls them by name.

- **`define_parameters(cls, schema)`** — declares model metadata, compartments, transmission edges, parameters, interventions, and demographic groups. Everything declared here is displayed in the UI.
- **`equation(self, y, t, p)`** — implements the model equations in Python using JAX.

`define_parameters` uses the `ParameterSchemaBuilder` ([API reference](https://who-collaboratory.github.io/pandemic-simulator-compartment/api/parameters/#compartment.parameters.ParameterSchemaBuilder)). The scaffold already called `set_model_info()` with the values you passed on the command line; edit them there to change them.

From `example_parameter_uncertainty_declarative_model/model.py`:

```python
schema.set_model_info(
    disease_type="example_parameter_uncertainty_declarative",
    label="Example Disease with Declarative Parameter Uncertainty",
    description="A SIR model for an example disease with declarative parameter uncertainty",
)

# Mark infective=True on every compartment that contributes to the force of infection.
schema.add_compartment("S", "Susceptible", "Population susceptible to infection")
schema.add_compartment("I", "Infected", "Currently infectious population", infective=True)
schema.add_compartment("R", "Recovered", "Recovered and immune")

# The framework auto-generates the I_total / R_total cumulative compartments
# for these edge targets — do not declare them by hand.
schema.add_transmission_parameter(
    source="susceptible",
    target="infected",
    variable_name="beta",
    frequency_dependent=True,
    label="Transmission Rate (S->I)",
    description="Rate at which susceptibles become infected through contact",
    default=0.3,
    default_min=0.1,
    default_max=0.5,
    min_value=0.01,
    max_value=2.0,
    unit="per day",
)
schema.add_transmission_parameter(
    source="infected",
    target="recovered",
    variable_name="gamma",
    label="Recovery Period (I->R)",
    description="Average number of days to recover",
    default=10.0,
    value_type=ValueType.DAYS,
    unit="days",
)

schema.add_intervention(
    id="my_intervention",
    label="My Intervention",
    description="Reduces transmission while active",
    target_rates=["beta"],
    adherence=50.0,
    transmission_reduction=50.0,
)

schema.add_demographic_group("age_0_4",     "Young children", default_weight=6.0,  age_range=(0, 4))
schema.add_demographic_group("age_5_17",    "School-age",     default_weight=16.0, age_range=(5, 17))
schema.add_demographic_group("age_18_49",   "Young adults",   default_weight=42.0, age_range=(18, 49))
schema.add_demographic_group("age_50_64",   "Older adults",   default_weight=19.0, age_range=(50, 64))
schema.add_demographic_group("age_65_plus", "Seniors",        default_weight=17.0, age_range=(65, 120))
```

> `value_type=ValueType.DAYS` means `default=10.0` is a 10-day mean, converted to a `0.1/day` rate at load. Do not pre-divide.

> **`ValueType.DAYS` is whole days on `add_parameter()`.** On a transmission parameter, as above, the value is stored as a float and fractions are fine. On `add_parameter()` it becomes an integer field in the generated config, so a fractional value is rejected outright:
>
> ```
> Input should be a valid integer, got a number with a fractional part
>   [type=int_from_float, input_value=5.9, input_type=float]
> ```
>
> Published estimates are rarely whole numbers, so for a duration that needs decimals, declare it as `ValueType.FLOAT` with `unit="days"` and convert it yourself with `self._to_rate(value, ValueType.DAYS)` in `equation()`. `ebola_seihfr_burial_legrand_model` does this for three durations, one of which is 9.6 days.
>
> A plain parameter's value reaches your model unconverted whichever type you give it — only transmission parameters get the automatic days-to-rate conversion — so `ValueType.DAYS` buys nothing there beyond the integer restriction. Prefer `ValueType.FLOAT` for durations and `ValueType.INTEGER` for values that really are whole numbers; both state plainly what the field accepts.

#### Automated vs. manual equations

There are two ways to write `equation()`.

**Automated (recommended).** Declare each compartment-to-compartment movement as a transmission edge and let `_compute_equations` build the derivatives. Use `skip_edges` to exclude any transition you want to write yourself, and `_apply_flow` to record it.

- **`_compute_equations`** — computes all standard transitions from the declared transmission edges.
- **`skip_edges`** — argument to `_compute_equations` listing transitions to exclude so you can implement them manually.
- **`_apply_flow`** — applies a manually computed transition: subtracts from the source, adds to the target, and updates the associated totals.

`example_parameter_uncertainty_declarative_model/model.py` uses the fully automated form:

```python
def equation(self, y, t, p):
    C = self.COMPARTMENTS
    params = self._unpack_params(p)
    states = {c: y[i] for i, c in enumerate(self.compartment_list)}

    I = states[C.I]
    non_total = [c for c in C if not c.endswith("_total")]
    N_total = sum(states[c] for c in non_total)
    prop_infective = I.sum() / (N_total.sum() + 1e-10)

    # Scales the target rates of any active intervention and returns the
    # updated travel matrix. With none configured it returns both unchanged.
    rates, self.travel_matrix = self._apply_interventions(
        t, {"beta": params["beta"]}, prop_infective
    )
    rates["gamma"] = params["gamma"]

    derivs = self._compute_equations(states, rates)
    return jnp.stack([derivs[c] for c in self.compartment_list])
```

To write one transition by hand and automate the rest, skip that edge and apply the flow yourself:

```python
derivs = self._compute_equations(states, rates, skip_edges={"beta"})
infection = rates["beta"] * S * I / N
self._apply_flow(derivs, "S", "I", infection)   # subtracts from S, adds to I, updates totals
```

**Manual.** Write every transition yourself. `example_parameter_uncertainty_custom_model/model.py` does this, and also replaces `_apply_interventions` with its own `custom_intervention()` that ramps adherence up and down instead of switching instantly:

```python
def equation(self, y, t, p):
    C = self.COMPARTMENTS
    params = self._unpack_params(p)
    states = {c: y[i] for i, c in enumerate(self.compartment_list)}
    ...
    beta = self.custom_intervention(t, params["beta"])
    gamma = params["gamma"]

    new_infections = beta * states[C.S] * prop_infective   # S -> I
    new_recoveries = gamma * states[C.I]                   # I -> R

    # Start every compartment at zero so the auto-generated _total
    # compartments are present before stacking.
    derivs = {c: jnp.zeros_like(states[C.I]) for c in self.compartment_list}
    derivs[C.S] = -new_infections
    derivs[C.I] = new_infections - new_recoveries
    derivs[C.R] = new_recoveries

    # _total compartments accumulate inflows only.
    derivs[f"{C.I}_total"] = new_infections
    derivs[f"{C.R}_total"] = new_recoveries

    return jnp.stack([derivs[c] for c in self.compartment_list])
```

> **Compartment order matters.** `equation()` indexes the state array by position. Always stack with `jnp.stack([derivs[c] for c in self.compartment_list])`; never hardcode an order.

Anything ramped, time-varying, or otherwise conditional on `t` must stay JAX-traceable — use `jnp.clip` and `jnp.where` rather than Python `if` statements on traced values, as `custom_intervention()` does.

#### Optional functions to override

**`__init__(self, config)`**

Most models start with `super().__init__(config)`, which loads the configuration, population matrix, transmission parameters, interventions, and other shared attributes:

```python
def __init__(self, config):
    super().__init__(config)
    # Model-specific initialisation goes here.
```

Afterwards your model can access `self.population_matrix`, `self.compartment_list`, `self.beta` / `self.gamma` and other transmission parameters, `self.interventions`, `self.intervention_statuses`, `self.contact_matrix`, `self.start_date`, `self.admin_units`, and `self.payload` (the local config, or user input from the UI).

`example_stochastic_model` extends `__init__` to seed a PRNG (pseudo-random number generator — the source of the random draws a stochastic model needs). Seeding it from a fixed `seed` makes runs reproducible; falling back to the clock makes each run different:

```python
def __init__(self, config):
    super().__init__(config)
    seed = config.get("seed") if isinstance(config, dict) else None
    if seed is None:
        seed = int(time.time() * 1000) % (2**31)
    self._key = jax.random.PRNGKey(seed)
```

A few models (such as dengue) initialize manually instead of calling `super().__init__(config)`. If you do that, you **must** set `self.compartment_list = list(self.COMPARTMENTS)` yourself — it is required throughout the framework and must match the order of the population matrix.

**`prepare_initial_state(self)`**

Builds the starting population matrix from each administrative region's population (WorldPop) and the initial infected percentage from the config or UI. **Do not rename it.** Override it if your model needs a different starting state — for a custom mobility matrix, override `build_travel_matrix()` instead (see below).

The population matrix is **(K, R)**: K compartments × R regions. With demographic groups declared via `schema.add_demographic_group()`, it should end up as **(K, A, R)** — call `_prepare_demographic_state()` rather than building it by hand. That method expands the default (K, R) matrix, appends zero-valued rows for `_total` compartments, updates `self.population_matrix` and `self.compartment_list` in place, and returns `None`. Models without demographic groups can ignore it.

**`get_initial_population(cls, admin_zones, compartment_list, **kwargs)`**

Override this when the initial infected cannot be seeded into a single `I` compartment. `example_stochastic_model` splits the seed across its asymptomatic (`A`) and symptomatic (`Sym`) compartments:

```python
for i, zone in enumerate(admin_zones):
    infected = round(zone["infected_population"] / 100 * zone["population"], 2)
    initial_population[i, col["S"]] = zone["population"] - infected
    initial_population[i, col["A"]] = infected * asymp_frac
    initial_population[i, col["Sym"]] = infected * (1.0 - asymp_frac)
```

**Mobility matrix.** The framework's default is the **identity matrix** — no inter-zone travel. There is no default gravity model: a model that travels declares its own mobility parameters and overrides `build_travel_matrix(self, admin_zones)`. The framework calls it before `prepare_initial_state()` and stores the result on `self.travel_matrix`, so don't assign that attribute yourself. Building the matrix isn't enough on its own — it only has an effect if `equation()` uses it in the force of infection. See [mobility.md](./mobility.md).

#### Built-in base class functions

- **`_add_total_compartments`** — by default the framework creates a cumulative `<target>_total` compartment for every transmission edge target (for example `I_total`). These track cumulative inflow and are used when computing summary results; you do not declare them, and they are not included in model outputs. To use a different aggregation strategy — for example one `_total` shared across several compartments — override it with `pass` and declare your own totals. `example_stochastic_model` does this because its two infectious compartments share a single `I_total`:

  ```python
  @classmethod
  def _add_total_compartments(cls, schema):
      pass  # I_total and R_total are declared in define_parameters() instead
  ```

  It then declares those totals itself, alongside its other compartments in `define_parameters()`:

  ```python
  schema.add_compartment(
      "I_total",
      "Infected Total",
      "Cumulative infections (asymptomatic + symptomatic combined)",
  )
  schema.add_compartment("R_total", "Recovered Total", "Cumulative recoveries")
  ```

  Because two compartments now feed one total, it also maps them onto the aggregate used for summary results:

  ```python
  COMPARTMENT_DELTA_GROUPING = {
      "S": ["S"],
      "I": ["A", "Sym"],
      "R": ["R"],
  }
  ```

- **`_apply_interventions`** — applies intervention effects by updating transmission rates and the mobility matrix during the simulation. Call it inside `equation()` to enable interventions. Override it, or write your own function as `example_parameter_uncertainty_custom_model` does, for custom logic.

#### Stochastic models

Models are deterministic by default. Declare a stochastic model with a class variable:

```python
class ExampleStochasticModel(Model):
    STOCHASTIC = True
```

The `example_stochastic_model` tells the `SimulationManager` to integrate with fixed-step Euler instead of the adaptive ODE solver, and to run several trajectories and report a median with an interval band. In a stochastic model, `equation()` returns the **per-step change**, since the integrator applies `y_{t+1} = y_t + dt * equation(...)`.


The number of trajectories is a declared parameter, so users can change it in the UI:

```python
schema.add_parameter(
    name="num_runs",
    label="Number of Runs",
    description="Number of stochastic trajectories to simulate.",
    value_type=ValueType.INTEGER,
    default=30,
    min_value=5,
    max_value=50,
    enable_variance=False,
)
```

To combine several compartments into one series for graphing, set `COMPARTMENT_DELTA_GROUPING`:

```python
COMPARTMENT_DELTA_GROUPING = {
    "S": ["S"],
    "I": ["A", "Sym"],   # asymptomatic + symptomatic shown as one "Infected" curve
    "R": ["R"],
}
```

#### Parameter uncertainty

There is no flag for parameter uncertainty. The framework detects it automatically when a config assigns a value **range** to a parameter, then draws Latin Hypercube samples and emits a median with a 95% simulation-based interval.

Give the parameter a `default_min` / `default_max` in `define_parameters()`:

```python
schema.add_transmission_parameter(
    source="susceptible",
    target="infected",
    variable_name="beta",
    default=0.3,
    default_min=0.1,
    default_max=0.5,
    ...
)
```

…then set `has_variance` in the config (see [example-config.json](#example-configjson)):

```json
{
    "field_key": "value",
    "has_variance": true,
    "distribution_type": "UNIFORM",
    "disease_param": "BETA",
    "min": 0.2,
    "max": 0.4
}
```

Intervention fields support variance too. `example_parameter_uncertainty_declarative_model/example-config.json` varies `adherence_min` between 40% and 60%.

### main.py

You should not need to change this file. The scaffold generates it with your class name wired in:

```python
import logging
import argparse
from compartment.driver import drive_simulation
from compartment.models.example_parameter_uncertainty_declarative_model.model import (
    ExampleParameterUncertaintyDeclarativeModel,
)

logging.getLogger("jax").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def lambda_handler(event, context):
    drive_simulation(
        model_class=ExampleParameterUncertaintyDeclarativeModel,
        args={"mode": "cloud", "simulation_job_id": event["simulation_job_id"]},
    )
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["local", "cloud"], default="local")
    parser.add_argument("--config_file")
    parser.add_argument("--output_file", nargs="?", default=None)
    parser.add_argument("--simulation_job_id", nargs="?", default=None)
    args = parser.parse_args()
    drive_simulation(
        model_class=ExampleParameterUncertaintyDeclarativeModel, args=vars(args)
    )
```

### Additional files

To organize your code further, put extra functions in separate files inside your model directory and import them into `model.py`.

---

## Documenting your model

A finished model needs three pieces of documentation. Each has a different audience and a different home:

| What you write | Where it lives | Who reads it, and where |
| :---- | :---- | :---- |
| A write-up of how the model behaves | `model.md`, beside `model.py` | Users, in the **"You should know"** panel on the results page. |
| `schema.set_model_metadata()` | inside `define_parameters()` in `model.py` | Users, on the **Simulation Configuration** page. |
| Docstrings | throughout your Python code | Other modelers, on the [technical API](#technical-api) site. |

The first two travel with your model into the Pandemic Simulator: both are collected into the artifact JSON the UI reads. The third is published separately, from the source code itself.

### model.md

Created by the scaffold as a template listing suggested sections — replace those with your own write-up. Use [Markdown syntax](https://www.markdownguide.org/basic-syntax/). Delete the file and the "You should know" panel is omitted.

Cover anything users should be aware of. The suggestions in the template are:

- **Model overview** — Brief plain-language description of how the model works.
- **Compartment and state definitions** — Meaning of each compartment, state,
  or population group.
- **Inputs and parameters** — Required inputs, definitions, units, valid ranges,
  and defaults.
- **Initial conditions** — How the starting population is distributed and any
  required initialization rules.
- **Outputs** — Available results, units, aggregation levels, and how each
  output should be interpreted.
- **Model nuances** — Subtle behaviors, implementation details, special
  conventions, or interpretation considerations that users should understand
  when configuring the model or evaluating its results.
- **Known edge cases** — Inputs or scenarios that may produce unstable,
  unrealistic, or invalid results.
- **Differences from the source model** — Any deliberate implementation changes
  or numerical approximations.
- **Related models** — When users might choose another model in the repository.

All three example models include one. From `example_stochastic_model/model.md`:

```markdown
# Example Disease (Stochastic) — Model Summary

A teaching example of a **stochastic SIR** model. It splits the infectious
population into **asymptomatic** and **symptomatic** compartments and uses
**tau-leaping** to add demographic randomness...

## Model structure

- **Compartments:** `S`, `A` (asymptomatic, infectious), `Sym` (symptomatic,
  infectious), `R`, plus a cumulative `I_total` tracker and `R_total`.
```

### Model metadata

Where `model.md` is prose, `schema.set_model_metadata()` records provenance and scope — who wrote the model, what it assumes, what it is for — as structured fields the UI can lay out on its own. Call it in `define_parameters()`, as `example_stochastic_model` does:

```python
schema.set_model_metadata(
    authors=[{"name": "Jenny Blase", "email": "jblase@ruvos.com", "affiliation": "Ruvos"}],
    license="MIT",
    model_type="Compartmental",
    diseases=["Example disease"],
    transmission_routes=["Airborne"],
    questions_answered=["How much does demographic stochasticity change outbreak size and timing?"],
    key_assumptions=["Closed population — no births or deaths."],
)
```

Every field is optional, and none of them affect how the simulation runs. Beyond those shown above you can also set `citations`, `applicability`, `not_for`, `constraints`, `biases`, and `validation` — see [`set_model_metadata`](https://who-collaboratory.github.io/pandemic-simulator-compartment/api/parameters/#compartment.parameters.ParameterSchemaBuilder.set_model_metadata) for what each one expects.

### Docstrings for the technical API

The [technical API](#technical-api) rebuilds whenever code is merged to `main`, and your model gets a page there with no extra files or configuration. The only requirement is that docstrings follow [Google style](https://google.github.io/styleguide/pyguide.html#383-functions-and-methods), using section headers like `Args:`, `Returns:`, and `Raises:` with indented descriptions:

```python
def equation(self, y, t, p):
    """Compute the compartment derivatives for one integration step.

    Args:
        y (jnp.ndarray): Current compartment values, ordered by ``compartment_list``.
        t (float): Current time in days since the simulation start date.
        p (tuple): Packed parameter tuple, unpacked via ``_unpack_params``.

    Returns:
        jnp.ndarray: The stacked per-compartment derivatives (dy/dt).
    """
```

Give each argument a type in parentheses and start the `Returns:` line with the type followed by a colon. Because these signatures carry no type annotations, the types you write in the docstring are the only thing that fills the **Type** column in the rendered tables.

To see the result, compare the docstring above with the published page for [`ExampleStochasticModel.equation`][equation-rendered]: the summary becomes the description, the arguments become a Parameters table, and the return line becomes a Returns table.

[equation-rendered]: https://who-collaboratory.github.io/pandemic-simulator-compartment/models/example-stochastic/#compartment.models.example_stochastic_model.model.ExampleStochasticModel.equation

Any public class or method without a docstring is omitted from the site.

---

## example-config.json

Running a model locally requires the information the Pandemic Simulator would normally supply: back-end data (population counts, administrative region coordinates) and user inputs from the UI. Each model directory contains an `example-config.json` you can edit, or regenerate from the schema.

**Required fields**

- **`Disease.disease_type`** — must match `set_model_info(disease_type=...)`. Determines which model is loaded.
- **`start_date`** and **`end_date`** — `YYYY-MM-DD`.
- **`admin_zones`** (or `case_file.admin_zones`) — each zone needs `name`, `population`, `center_lat`, `center_lon`, `infected_population` (a percentage — see below), and any fields declared with `add_admin_zone_field()`.
- **`TransmissionEdges.items`** — one entry per transmission edge declared in `define_parameters()`.

Optional: `Interventions`, `travel_volume`, `demographics`, `contact_matrix_overrides`, `demographic_rate_overrides`, and any additional disease parameters declared in the schema.

If a required field is missing, the framework raises a `ValidationError`, logs which field is missing, and exits.

`example_parameter_uncertainty_declarative_model/example-config.json`, abbreviated — note `has_variance: true` on `beta`, which puts the run in uncertainty mode:

```json
{
    "Disease": { "disease_type": "example_parameter_uncertainty_declarative" },
    "start_date": "2026-01-01",
    "end_date": "2026-12-31",
    "TransmissionEdges": {
        "items": [
            {
                "transmission_edge": { "source": "susceptible", "target": "infected", "value_type": "RATE" },
                "value": 0.3,
                "FieldConfigs": {
                    "items": [
                        { "field_key": "value", "has_variance": true, "distribution_type": "UNIFORM",
                          "disease_param": "BETA", "min": 0.2, "max": 0.4 }
                    ]
                }
            },
            {
                "transmission_edge": { "source": "infected", "target": "recovered", "value_type": "DAYS" },
                "value": 10.0
            }
        ]
    },
    "Interventions": {
        "items": [
            {
                "Intervention": { "name": "MY_INTERVENTION", "display_name": "My intervention" },
                "adherence_min": 50.0,
                "transmission_percentage": 50.0,
                "start_threshold": 2.0,
                "end_threshold": 1.0
            }
        ]
    },
    "admin_zones": [
        {
            "name": "Example Zone",
            "center_lat": 40.7128,
            "center_lon": -74.006,
            "population": 1000000,
            "infected_population": 0.01
        }
    ]
}
```

> **`infected_population` is a percentage, not a case count.** The framework seeds the initial infected as `infected_population / 100 * population`, so the `0.01` above is 0.01% of 1,000,000 — 100 people, not 1 person and not 1%. Valid values run from 0 to 100.
>
> To start from a known number of cases, you must convert it: `cases / population * 100`. To seed 25 cases in the zone above, use `25 / 1000000 * 100 = 0.0025`. Entering `25` instead would start the run with 25% of the zone infected — 250,000 cases.

Model-specific parameters declared with `add_parameter()` go in the `Disease` block. `example_parameter_uncertainty_custom_model` passes its ramp settings this way:

```json
"Disease": {
    "disease_type": "example_parameter_uncertainty_custom",
    "ramp_up_days": 14,
    "ramp_down_days": 21
}
```

> Top-level `admin_zones` and `demographics` are automatically wrapped into `case_file` when the config loads. Do not double-nest them in a hand-written config.

### Generating artifacts

A model artifact is a saved representation of the model configuration that can be used to reproduce the model. Regenerate `example-config.json` from the schema after changing `define_parameters()`:

**macOS / Linux**
```shell
python -m compartment.generate_artifact example_parameter_uncertainty_declarative \
    --example-config \
    --config-output compartment/models/example_parameter_uncertainty_declarative_model/example-config.json
```

**Windows Command Prompt**
```shell
python -m compartment.generate_artifact example_parameter_uncertainty_declarative --example-config --config-output compartment\models\example_parameter_uncertainty_declarative_model\example-config.json
```

> Regenerating **overwrites** the file. If you have hand-edited it — added variance ranges, intervention dates, or real admin zones — write to a new path first and merge, or re-apply your edits afterwards.

Other artifact commands:

```shell
# List every model that supports artifact generation
python -m compartment.generate_artifact --list

# Print artifact JSON to stdout
python -m compartment.generate_artifact example_stochastic

# Write the artifact to a file
python -m compartment.generate_artifact example_stochastic --output artifact.json
```

---

## Install additional packages

You may need to install additonal pacakges for your model. This project uses **uv** for dependency management. From the repository root:

```shell
uv add <package-name>
```

For example, `uv add scikit-learn` adds the dependency to `pyproject.toml`, updates `uv.lock`, and installs it into the virtual environment.

Verify, review, and commit:

```shell
uv run python -c "import sklearn; print(sklearn.__version__)"
git add pyproject.toml uv.lock
git commit -m "Add scikit-learn dependency"
```

Variations:

```shell
uv add --dev pytest          # development-only dependency
uv add "pandas>=2.2,<3.0"    # version range
uv add pandas==2.3.1         # pinned version
```

> **Never edit `uv.lock` manually.** Treat it as generated output, updated through `uv add`, `uv remove`, or `uv lock`.

---

## Run the model locally

Each run executes two simulations in parallel — one with interventions and one without (the control run). Both are written to the output JSON.

**macOS / Linux**
```shell
python -m compartment.models.example_parameter_uncertainty_declarative_model.main \
    --mode local \
    --config_file compartment/models/example_parameter_uncertainty_declarative_model/example-config.json \
    --output_file results/example-declarative-test.json
```

**Windows Command Prompt**
```shell
python -m compartment.models.example_parameter_uncertainty_declarative_model.main --mode local --config_file compartment\models\example_parameter_uncertainty_declarative_model\example-config.json --output_file results\example-declarative-test.json
```

The last lines of a successful run look like:

```
[INFO] compartment.run_simulation: Results saved to: results/example-declarative-test.json
[INFO] root: Elapsed time: 4.12 seconds
```

The other two examples follow the same pattern:

**macOS / Linux**
```shell
# Custom equation + ramped intervention
python -m compartment.models.example_parameter_uncertainty_custom_model.main \
    --mode local \
    --config_file compartment/models/example_parameter_uncertainty_custom_model/example-config.json \
    --output_file results/example-custom-test.json

# Stochastic
python -m compartment.models.example_stochastic_model.main \
    --mode local \
    --config_file compartment/models/example_stochastic_model/example-config.json \
    --output_file results/example-stochastic-test.json
```

**Windows Command Prompt**
```shell
python -m compartment.models.example_parameter_uncertainty_custom_model.main --mode local --config_file compartment\models\example_parameter_uncertainty_custom_model\example-config.json --output_file results\example-custom-test.json

python -m compartment.models.example_stochastic_model.main --mode local --config_file compartment\models\example_stochastic_model\example-config.json --output_file results\example-stochastic-test.json
```

> `--mode cloud` is for the Pandemic Simulator web app and is not supported for local use.

### Visualize model results

To view charts similar to the UI's results page:

**macOS / Linux**
```shell
python tools/view_results.py results/example-declarative-test.json
```

**Windows Command Prompt**
```shell
python tools\view_results.py results\example-declarative-test.json
```

This plots the whole-population time series (`parent_admin_total`) for the with-interventions and control runs side by side, and:

- draws a green dashed line at each intervention start date and a red dotted line at each end date;
- shades uncertainty bands for uncertainty and stochastic runs;
- prints a table of cumulative compartment totals using the same compartment names as the frontend.

Flags: `-c/--compartments`, `--log`, `--no-deltas`, `--title`, `-o/--save`. Run with `-h` for details, or see [tools/README.md](https://github.com/WHO-Collaboratory/pandemic-simulator-compartment/tree/main/tools).

```shell
# Only some compartments, log y-axis
python tools/view_results.py results/example-declarative-test.json -c S,I,R --log

# Save an image instead of opening a window
python tools/view_results.py results/example-stochastic-test.json -o run.png
```

---

## Run tests

Integration tests verify that your model contains everything needed to run in the Pandemic Simulator. Run all of them before opening a pull request.

`tests/test_smoke.py` auto-discovers any model directory containing both `model.py` and `example-config.json` — no test registration is needed.

```shell
# Smoke-test all three example models
python -m pytest tests/test_smoke.py -v -m integration -k example

# Smoke-test one model
python -m pytest tests/test_smoke.py -v -m integration -k example_stochastic
```

Additional test files:

```shell
# Model can be discovered, loaded, and used by the framework
python -m pytest tests/test_artifact.py -v

```

The `-k` flag is a pytest keyword filter: it runs every discovered test whose name contains the given substring, so `-k example` matches all three example models (any directory whose name contains `example`). Replace `example` with your own model's directory name to narrow it to just your model.

---

## Submit a pull request

Once your model runs and passes tests, open a pull request so WHO Collaboratory reviewers can review and approve it.

1. **Check what changed**

   ```shell
   git status
   git diff
   ```

2. **Stage and commit**

   ```shell
   git add .
   git commit -m "Add example disease model with parameter uncertainty"
   ```

3. **Push your branch**

   ```shell
   git push origin add-example-disease-model
   ```

4. **Open the pull request.** After `git push`, the terminal output usually includes a `Create a pull request` link (a GitHub URL for your branch) — Cmd-click (or Ctrl-click) it to open the PR form directly in your browser. Otherwise go to the repository on GitHub, where a yellow banner reading **"Compare & pull request"** usually appears at the top of the page just after pushing — click it to jump straight to the PR form. If neither shows up, open the **Pull requests** tab, click **New pull request**, and pick your branch manually. Either way, confirm **base** is set to `main` and **compare** is set to your feature branch, then click **Create pull request**.

5. **Write a clear description** covering what changed, why, how it was tested, and any known limitations or follow-up work.

6. **Request review** from the appropriate reviewer or team. Respond to comments and push additional commits if changes are requested.

7. **Merge after approval**, once checks pass, following the project's standard process.

---

## Approve a model

Everything above is the modeler's job. This section is the **approver's** — the reviewer who takes a submitted model from an open pull request to a model users can actually pick in the simulator.

Models are published to UAT, at <https://uat.pandemic-simulator.com/>. Every step below happens there.

The short version: review the PR, merge it, tag a release, then publish the model from the **Model Approvals** dashboard. Merging alone puts nothing in front of users — the tag is what builds and ships the model, and the Publish button is what reveals it.

### 1. Review the pull request

Read the diff the way a user will meet the model: **anything declared on the `schema` becomes UI.**

- **The schema is a contract.** Every `add_parameter()`, `add_transmission_parameter()`, and `add_intervention()` turns into a control on the Simulation Configuration page. Check that labels read plainly, descriptions say what the number actually does, units are right, and `min_value` / `max_value` bound the parameter to a physically sensible range. A slider is only as safe as its limits, and widening one later widens it for every existing user.
- **The defaults have to run.** `example-config.json` is what the smoke test executes and what seeds a user's first simulation. Confirm the defaults produce a plausible epidemic curve rather than a flat line or a blow-up.
- **The model documents itself.** `model.md` and the `schema.set_model_metadata()` block — authors, license, citations, key assumptions, applicability, `not_for`, known biases, validation — are rendered verbatim in the approvals preview and on the model's page. Missing provenance is a legitimate reason to request changes.
- **CI is green.** `smoke-tests` runs each model's integration smoke test plus `tests/test_<model>.py` where one exists. Don't approve on a red or skipped matrix leg.
- **Run it yourself** when the diff touches `equation()`, mobility, or interventions:

  ```shell
  python -m compartment.models.<their_model>.main \
      --mode local \
      --config_file compartment/models/<their_model>/example-config.json \
      --output_file results/<their_model>-review.json

  python tools/view_results.py results/<their_model>-review.json
  ```

Approve, or request changes with specifics. Then merge to `main`.

### 2. Tag a release

Merging builds nothing. The `disease-pipeline` workflow fires on a semver tag, and you create that tag from GitHub — no terminal needed.

1. **Open Releases.** From the repository's **Code** tab, click **Releases** in the right-hand sidebar (or go to the repository URL with `/releases` on the end).

2. **Click "Draft a new release."**

3. **Create the tag.** Click the **Choose a tag** dropdown, type the new version — `v1.4.0` — and click **"+ Create new tag: v1.4.0 on publish."** The tag must start with `v` and be three numbers separated by dots, or the pipeline will not fire.

4. **Confirm the target is `main`.** The **Target** dropdown sits next to the tag dropdown and defaults to `main`. If it shows a branch or commit, set it back to `main` — the tag is cut from whatever is selected here.

5. **Add a title and notes.** Use the version as the title. Click **Generate release notes** to pull in the merged pull requests since the last release, then add a line naming the model that changed and what changed about it.

6. **Click "Publish release."** This is the step that creates and pushes the tag, which is what starts the build. **Saving a draft does nothing** — a draft release holds no tag, so the pipeline never runs. If you tick **Set as a pre-release**, the tag is still created and the pipeline still runs; the label is cosmetic.

The tag covers **every** model in `compartment/models/` that contains an `example-config.json`, not just the one that changed — the pipeline auto-discovers them and fans out. Re-publishing unchanged models is harmless (identical artifacts are content-hashed and skipped), but the version number applies repo-wide, so pick it accordingly.

**Troubleshooting**

- **Nothing happens after publishing** — check the tag's spelling on the **Releases** page. `1.4.0`, `v1.4`, and `release-1.4.0` all fail to match the workflow's tag pattern and are silently ignored.
- **The tag already exists** — GitHub will not let you reuse one. Cut the next patch version rather than deleting and recreating a tag; deleting a published tag breaks the artifact-to-version mapping for anything already provisioned from it.

### 3. Watch the pipeline

Open the **Actions** tab and follow the `disease-pipeline` run for your tag. In order, it:

1. Runs the smoke tests.
2. Builds and pushes a container image per model to ECR, tagged `{model_directory}-{tag}`.
3. Generates each model's artifact JSON, uploads it to S3 under its SHA-256, and records the image-to-artifact mapping.
4. Emits a `ModelVersionPublished` event per artifact.

That event is what provisions the model's Lambda and registers the artifact with the API. Downstream, the artifact processor seeds the model's transmission edges, interventions, custom fields, and demographic groups so the UI has something to render.

**Troubleshooting**

- **Job fails at "Stage datasets"** — a file declared in the model's `datasets.yaml` is missing from the bucket. Fix the dataset, then tag again; the pipeline deliberately fails here rather than letting the simulation fail later with no data.
- **Pipeline is green but nothing appears in Model Approvals** — provisioning is asynchronous and takes a few minutes. If the artifact still hasn't landed, check the provisioner's dead-letter queue.
- **Never move or delete a tag to "redo" a release.** Cut the next patch version instead.

### 4. Review the model in Model Approvals

Go to **Model Approvals** (<https://uat.pandemic-simulator.com/model-approvals>). The dashboard is visible to admins, super admins, and disease modelers; **only a super admin can change a model's status.**

The new version arrives with status **NEW** and is invisible to ordinary users. Find it by name, disease type, or version, and select it to open the preview panel on the right. Read the preview as the user-facing surface it is: compartments, transmission edges, interventions, custom fields, demographic groups, contact-matrix overrides, and the model's authorship and assumptions. Anything that reads wrong here reads wrong to the user.

Statuses run **NEW → PREPUBLISH → PUBLISHED → ARCHIVED**. Only `PUBLISHED` models reach the disease-model dropdown on the simulation page.

### 5. Simulate before publishing

Click **Simulate**. This opens the simulation form preloaded with the model even though it isn't published yet, which is the point: a real run against the deployed Lambda, in UAT, before any user can reach it.

Confirm that:

- The configuration form renders every parameter with the labels, units, and bounds you reviewed in the PR.
- The run completes rather than erroring or timing out.
- Results are plausible, and the intervention and control curves differ in the direction the intervention claims.
- Interventions and admin-zone selection behave as documented.

If anything is wrong, leave the model unpublished and go back to the modeler. An unpublished model costs nothing; a published broken one is in front of every user.

### 6. Publish

With the model selected in the preview panel, click **Publish** (super admin only). Status flips to `PUBLISHED` and the model appears in the disease-model dropdown for all UAT users.

Verify the result: open the simulation page as a normal user would, confirm the model is listed under the right name and version, and run it once from a clean form.

### Unpublishing and archiving

- **Unpublish** returns a published model to `PREPUBLISH`. It disappears from the dropdown immediately while staying available in Model Approvals — this is the fast lever if a problem surfaces after release.
- **Archive** retires a model version for good. Use it for superseded versions, not for a model you intend to fix and re-publish.

Neither action deletes anything, and neither touches simulations users have already run.
