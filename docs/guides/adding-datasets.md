# Adding Your Own Data

Some models require external data, such as a contact matrix, population table,
or time-varying parameter schedule. This guide explains how to add a data file
to your model and upload it using your Pandemic Simulator login. Files can use
any format, including CSV or JSON, and can be up to 500 MB per dataset.
All datasets are visible to all modelers using the Pandemic Simulator, so do not upload
confidential information or personal data. See the
[MPOX example](#a-complete-example) for a complete implementation. 
Follow the steps below to upload a dataset and use it in your model.

## Local runs and cloud runs

Your model reads its data the same way in both cases — `self.dataset("name")`,
looked up in the `datasets.yaml` you write in Step 2. Only the way the file
arrives differs:

- **Running locally**, you fetch the file yourself with `pull`, once per clone.
  Data files are not stored in git, so a fresh clone has your `datasets.yaml` but
  not the data it points to.
- **Running in the cloud**, nothing to do at run time. When your model is
  released, everything listed in `datasets.yaml` is packaged up and sent along
  with it. Each dataset must already have been uploaded with `push`, or the
  release fails.

So `push` is what publishes your data for the cloud and for other modelers, and
`pull` is what puts a published file back on your own machine.

---

## Step 1 — Navigate to your model directory and add the data file

From the repository root, navigate to your model's directory. For example, the
MPOX model uses:

**macOS / Linux**
```shell
cd ~/Desktop/pandemic-simulator-compartment/compartment/models/mpox_jax_model
```

**Windows Command Prompt**
```shell
cd %USERPROFILE%\Desktop\pandemic-simulator-compartment\compartment\models\mpox_jax_model
```

Replace `mpox_jax_model` with your model's directory name. The data file goes in
a `data` directory inside the model directory. Create that directory if it does
not exist yet:

**macOS / Linux**
```shell
mkdir -p data
```

**Windows Command Prompt**
```shell
mkdir data
```

Then copy your data file into it. For example, if the file is in your Downloads
folder:

**macOS / Linux**
```shell
cp ~/Downloads/transition-rate-multipliers.csv data/
```

**Windows Command Prompt**
```shell
copy %USERPROFILE%\Downloads\transition-rate-multipliers.csv data\
```

The MPOX model has this structure:

```text
compartment/models/mpox_jax_model/
    model.py
    datasets.yaml
    data/
        transition-rate-multipliers.csv
```

## Step 2 — Describe your dataset/s in `datasets.yaml`

Create an empty file called `datasets.yaml` in your model's directory, next to
`model.py` — not inside `data/`. From the same directory you were in for Step 1:

**macOS / Linux**
```shell
touch datasets.yaml
```

**Windows Command Prompt**
```shell
type nul > datasets.yaml
```

Open it in your editor and describe each of your data files with these three
fields:

| Field | What it means |
|---|---|
| `name` | What you want to call the dataset. Others will see this name. |
| `version` | Which version of the data this is. |
| `file` | Where the file is, relative to `datasets.yaml`. |

The MPOX model has one data file, `data/transition-rate-multipliers.csv`, so its
`datasets.yaml` is:

```yaml
datasets:
  - name: mpox-transition-multipliers
    version: "1"
    file: data/transition-rate-multipliers.csv
```

You can list as many datasets as you like — add one `- name:` block per file
under `datasets:`. The same name and version cannot appear twice in one file.

**Always quote the version.** Write `version: "1"`, not `version: 1`. Without
quotes, YAML reads `1` as a number and `1.0` as a decimal. The tool accepts
either and converts it, but quoting avoids surprises like `1.10` becoming `1.1`.


## Step 3 — Upload your data

**Every dataset is scanned for malware before anyone can use it.** Uploading is
therefore two steps. `push` copies your file to a holding area and then finishes
— it does not wait for the scan, so the command being done does not mean your
dataset is ready. The scan runs in the background, and Step 4 is how you find
out whether it passed. Until it passes, your dataset is not published and cannot
be pulled; a file that fails the scan is deleted rather than made available.

From your model's directory — the same place as `datasets.yaml` — run:

```bash
python -m compartment.datasets push
```

The first time you run it, you are asked to sign in:

1. A browser opens at your profile page. (If it does not, the address is
   printed — open it yourself.)
2. Click **Copy Session Token**.
3. Paste it into the terminal and press Enter.

You only sign in once. The token is saved and reused for about an hour, so the
other commands will not ask again.

The output looks something like this:

```
Authenticated as you@example.org.
mpox-transition-multipliers@1  upload-id 3eff5b8d-ebe9-4efd-b376-98d79b247faa

Uploads are being scanned for malware. Check progress with:
  python -m compartment.datasets check-status <upload-id>
```

**Copy the upload-id.** Step 4 uses it to check whether the scan passed. One is
printed per dataset listed in your `datasets.yaml`.

**If your `datasets.yaml` is in a different location**, run the `push` command pointing to that location:

```bash
python -m compartment.datasets push --manifest <path/to>/datasets.yaml
```

To upload only some of the datasets listed in the file, name them:

```bash
python -m compartment.datasets push mpox-transition-multipliers
```

## Step 4 — Confirm your data passed the security scan

The scan usually takes **10 to 60 seconds**. Check on the scan's status with the upload-id from Step 3:

```bash
python -m compartment.datasets check-status 3eff5b8d-ebe9-4efd-b376-98d79b247faa
```

```
upload-id : 3eff5b8d-ebe9-4efd-b376-98d79b247faa
dataset   : mpox-transition-multipliers@1 (transition-rate-multipliers.csv)
status    : PROMOTED
detail    : Malware scan clean. Dataset promoted and available via `pull`.
location  : s3://collaboratory-datasets/datasets/mpox-transition-multipliers/1/transition-rate-multipliers.csv
```

Reading the scan result

The `status` line is the one that matters. It has four possible values:

| Status | What it means | What to do |
|---|---|---|
| `SCANNING` | The scan has not finished. Nothing is published yet. | Wait a few seconds and run `check-status` again. |
| `PROMOTED` | The scan was clean and your dataset is published. | Nothing — go on to Step 5. |
| `REJECTED` | Either malware was found, or that name and version were already published. | Read the `detail` line; it says which. For a name and version already in use, bump the version in `datasets.yaml` and push again. |
| `FAILED` | The scan could not finish. Nothing was published. | Run `push` again. |

Whatever the outcome, nothing is left in the holding area: a promoted dataset
moves into permanent storage, and anything else is deleted.

## Step 5 — Use it in your model

Read the file with `self.dataset()`, passing the `name` from your
`datasets.yaml`:

```python
import pandas as pd

class MpoxJaxModel(Model):
    def _load_transition_schedule(self):
        table = pd.read_csv(self.dataset("mpox-transition-multipliers"))
        ...
```

`self.dataset()` returns the path to the file. Open it with pandas, numpy,
`json`, or your package of choice.

**That one line works everywhere.** Locally it points at the file in your `data/`
folder; in the cloud it points at the copy sent along with your model. Never
build the path by hand: `Path(__file__).parent / "data" / "transition-rate-multipliers.csv"`
appears to work, but it ignores `datasets.yaml`. When you later replace the data
— published as a new version, meaning a fresh entry in `datasets.yaml` with
`version:` raised and usually a new filename — `self.dataset()` follows that
entry, while a hand-written path continues opening the old file.

If the file is missing, the error names the command that fixes it:

```
mpox-transition-multipliers@1 is declared in .../datasets.yaml but
.../data/transition-rate-multipliers.csv does not exist. Download it with:
  python -m compartment.datasets pull mpox-transition-multipliers@1 --dest .../data
```

This is normal on a fresh clone: `datasets.yaml` is tracked in git, the data
files are not. Run the `pull` it suggests — once per clone, not once per run.
Only local runs need it; when your model is released, everything listed in
`datasets.yaml` is downloaded and sent along with it.

## A complete example

The [MPOX model](https://github.com/WHO-Collaboratory/pandemic-simulator-compartment/blob/main/compartment/models/mpox_jax_model/model.py)
is a working version of everything above. Its
[`datasets.yaml`](https://github.com/WHO-Collaboratory/pandemic-simulator-compartment/blob/main/compartment/models/mpox_jax_model/datasets.yaml)
is the file shown in Step 2, and `_load_transition_schedule()` reads the CSV
with `self.dataset()`, validates its columns, and interpolates between the
elapsed-day checkpoints to get a multiplier for any simulation day.

The schedule is applied uniformly to every zone, so the model runs against any country, zone names, number
of zones, and start date.

As with every model, its CSV is not in git, so pull it once before running the
model locally. Nothing to do for the cloud — releasing the model packages the
dataset up with it for you:

**macOS / Linux**
```shell
cd compartment/models/mpox_jax_model
python -m compartment.datasets pull mpox-transition-multipliers@1 --dest data/
```

**Windows Command Prompt**
```shell
cd compartment\models\mpox_jax_model
python -m compartment.datasets pull mpox-transition-multipliers@1 --dest data\
```

---

## Finding and downloading published datasets

Every published dataset is available to every modeler. List them all:

```bash
python -m compartment.datasets list
```

Each row gives a name, a version, and the filename you will get:

```
mpox-transition-multipliers  1             transition-rate-multipliers.csv
mpox-transition-multipliers  2             transition-rate-multipliers.csv
uganda-admin-zones           1.0.0         uganda-zones.json
```

Download one into your own model's `data/` directory:

**macOS / Linux**
```shell
cd compartment/models/my_model
python -m compartment.datasets pull mpox-transition-multipliers@1 --dest data/
```

**Windows Command Prompt**
```shell
cd compartment\models\my_model
python -m compartment.datasets pull mpox-transition-multipliers@1 --dest data\
```

Leaving the version off gives you the **most recently uploaded** one:

**macOS / Linux**
```shell
python -m compartment.datasets pull mpox-transition-multipliers --dest data/
```

**Windows Command Prompt**
```shell
python -m compartment.datasets pull mpox-transition-multipliers --dest data\
```

Careful: most recently uploaded is not the same as the highest number. If
someone uploaded version `1` after version `2`, this gives you version `1`.
**Name the version whenever it matters.**

Downloading the file does not by itself make it visible to your model. Add it to
your own `datasets.yaml` as in Step 2, with the same name and version you pulled,
and read it with `self.dataset()` as in Step 5.

---

## Rules to know

**Datasets never change.** Once `mpox-transition-multipliers` version `1`
exists, it is fixed forever. If your data changes, upload it with a new version:

```yaml
datasets:
  - name: mpox-transition-multipliers
    version: "2"          # was "1"
    file: data/transition-rate-multipliers.csv
```

This is deliberate. Results from an old model run must always be reproducible,
so the data behind them cannot be replaced.

If you try to reuse a version, you get a clear message before anything is
uploaded:

> Dataset mpox-transition-multipliers version 1 already exists. Datasets are
> immutable — bump the version in datasets.yaml.

**Naming.** Names and versions may use letters, numbers, dots, dashes and
underscores. They must start with a letter or number, and cannot contain
slashes. Names can be up to 128 characters, filenames up to 256.

**Size.** Up to **500 MB per dataset**. Every dataset your model declares is
packaged up and sent to the cloud with it, so the limit is about how big that
package gets, not what the virus scanner can handle. If your file is bigger:
compress it, drop columns you do not use, coarsen the resolution, or split it
into several smaller datasets. You are told before the upload starts, not after.

**Visibility.** Every dataset is visible to everyone using the simulator. Do not
upload anything confidential or containing personal data.

---

## Your model records what it used

When your model's artifact is generated, the contents of `datasets.yaml` are
copied into it automatically:

```json
{
  "datasets": [
    {
      "name": "mpox-transition-multipliers",
      "version": "1",
      "filename": "transition-rate-multipliers.csv"
    }
  ]
}
```

You do not have to do anything for this. It means anyone looking at a model run
later can see exactly which data it was built from. Models without a
`datasets.yaml` simply do not get this section.

This list is also what puts your data in the cloud. When your model is released,
each dataset in it is downloaded and packaged up with the model — which is why
`self.dataset()` finds the file when the model runs. Two consequences worth
knowing:

- **A dataset must be published before the release.** If `datasets.yaml` names
  something that was never pushed (or was rejected), the release fails with a
  message naming the missing dataset. Better there than in a simulation someone
  is waiting on.
- **Changing data means a new release.** Bumping a version in `datasets.yaml`
  changes nothing until the model is released again.

---

## If something goes wrong

| Message | What it means | Fix |
|---|---|---|
| `No manifest at datasets.yaml` | The tool could not find the file. | Run from the folder holding `datasets.yaml`, or use `--manifest`. |
| `must contain a top-level 'datasets:' list` | The file's shape is wrong. | It must start with `datasets:` and a list of entries under it. |
| `no such file ...` | A `file:` path does not point at anything. | Paths are relative to `datasets.yaml`, not to where you are standing. |
| `... is listed twice` | The same name and version appear twice. | Remove the duplicate or change one version. |
| `over the 500 MB per-dataset limit` | The file is too big to send to the cloud with a model. | Shrink or split it. See **Size** above. |
| `does not declare 'name'` | The name you passed to `self.dataset()` is not in `datasets.yaml`. | The message lists the names that are declared — usually a typo. |
| `is declared ... but ... does not exist` | Your `datasets.yaml` is right but the file is not downloaded. | Run the `pull` command in the message. Normal on a fresh clone. |
| `already exists. Datasets are immutable` | That version is published. | Bump the version. |
| `Session token is invalid or expired` | Your token has run out (about an hour). | Run the command again and paste a fresh token. |
| `was not issued by the expected Cognito user pool` | You copied the token from the wrong site. | Copy it from the same site the CLI opened for you, not a different environment. |
| `not a well-formed JWT` | The token got cut off when copying. | Copy it again, all of it. |

If a command keeps asking you to sign in, delete the saved token and try once
more:

**macOS / Linux**
```shell
rm ~/.pansim/dataset-session.json
```

**Windows Command Prompt**
```shell
del %USERPROFILE%\.pansim\dataset-session.json
```

---

## Command reference

| Command | What it does |
|---|---|
| `python -m compartment.datasets push` | Upload everything in `./datasets.yaml`. |
| `python -m compartment.datasets push NAME` | Upload only the named dataset. |
| `python -m compartment.datasets push --manifest PATH` | Use a `datasets.yaml` somewhere else. |
| `python -m compartment.datasets check-status ID` | Check whether an upload was published. |
| `python -m compartment.datasets list` | List every published dataset. |
| `python -m compartment.datasets pull NAME` | Download the most recently uploaded version. |
| `python -m compartment.datasets pull NAME@VERSION` | Download a specific version. |
| `python -m compartment.datasets pull NAME --dest DIR` | Download into a chosen folder. |

`check-status` exits with a non-zero code if the upload was `REJECTED` or
`FAILED`, so you can use it in a script.
