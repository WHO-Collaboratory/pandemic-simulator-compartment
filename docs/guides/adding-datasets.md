# Adding Your Own Data

Some models need a data file: a contact matrix, a population table, a set of
admin zones. This guide shows you how to upload one and use it in your model.

You do not need an AWS account. You do not need to set any environment
variables. You need your normal Pandemic Simulator login.

## How it works, in one paragraph

You put your file next to your model and list it in a small file called
`datasets.yaml`. You run one command to upload it. The file is scanned for
viruses. If it is clean, it is published and everyone can use it. If it is not
clean, it is deleted. Once published, a dataset never changes — if your data
changes, you upload it again under a new version number.

---

## Step 1 — Put your file with your model

Keep the data file inside your model's folder.

```
compartment/models/my-model/
    model.py
    datasets.yaml        ← you will create this in step 2
    data/
        kenya-contacts.csv
```

## Step 2 — Describe it in `datasets.yaml`

Create a file called `datasets.yaml` in your model's folder:

```yaml
datasets:
  - name: kenya-contact-matrix
    version: "1"
    file: data/kenya-contacts.csv
```

Three things per dataset:

| Field | What it means |
|---|---|
| `name` | What you want to call the dataset. Others will see this name. |
| `version` | Which version of the data this is. **Always put it in quotes.** |
| `file` | Where the file is, relative to `datasets.yaml`. |

!!! warning "Quote the version"
    Write `version: "1"`, not `version: 1`. Without quotes YAML reads `1` as a
    number and `1.0` as a decimal. The tool will accept it and convert it, but
    quoting avoids surprises like `1.10` becoming `1.1`.

You can list as many datasets as you like. You cannot list the same name and
version twice in one file.

## Step 3 — Upload it

From your model's folder:

```bash
python -m compartment.datasets push
```

The first time, it asks you to sign in:

1. A browser opens at your profile page. (If it does not, the address is
   printed — open it yourself.)
2. Click **Copy Session Token**.
3. Paste it into the terminal and press Enter.

You will see something like:

```
Authenticated as you@example.org.
kenya-contact-matrix@1  upload-id 3eff5b8d-ebe9-4efd-b376-98d79b247faa

Uploads are being scanned for malware. Check progress with:
  python -m compartment.datasets check-status <upload-id>
```

**Copy that upload-id.** You need it for the next step.

You only sign in once. The token is saved and reused for about an hour, so the
other commands will not ask again.

!!! tip "If your `datasets.yaml` is somewhere else"
    ```bash
    python -m compartment.datasets push --manifest path/to/datasets.yaml
    ```
    To upload only some of the datasets listed in the file, name them:
    ```bash
    python -m compartment.datasets push kenya-contact-matrix
    ```

## Step 4 — Check it was accepted

The upload command finishes straight away. The virus scan runs afterwards, and
usually takes **10 to 60 seconds**.

```bash
python -m compartment.datasets check-status 3eff5b8d-ebe9-4efd-b376-98d79b247faa
```

```
upload-id : 3eff5b8d-ebe9-4efd-b376-98d79b247faa
dataset   : kenya-contact-matrix@1 (kenya-contacts.csv)
status    : PROMOTED
detail    : Malware scan clean. Dataset promoted and available via `pull`.
location  : s3://collaboratory-datasets/datasets/kenya-contact-matrix/1/kenya-contacts.csv
```

What the statuses mean:

| Status | Meaning | What to do |
|---|---|---|
| `SCANNING` | Still being checked. | Wait a few seconds and run the command again. |
| `PROMOTED` | Clean and published. | Done. |
| `REJECTED` | A virus was found, **or** that name and version already exist. | Read the `detail` line — it says which. |
| `FAILED` | The scan could not finish. Nothing was published. | Run `push` again. |

Your file is deleted from the holding area either way. Nothing unscanned is
ever published.

## Step 5 — Use it in your model

Read the file with `self.dataset()`, passing the `name` from your
`datasets.yaml`:

```python
import pandas as pd

class MyModel(Model):
    def build_travel_matrix(self, admin_zones):
        contacts = pd.read_csv(self.dataset("kenya-contact-matrix"))
        ...
```

`self.dataset(...)` hands back the path to the file. Open it however you like —
pandas, numpy, `json`, anything. The simulator does not care what is in it.

**This one line works everywhere.** On your machine it points at the file in
your `data/` folder. When your model runs in the cloud, it points at the copy
built into your model's image. You never write a path, and you never write an
S3 address.

!!! warning "Do not build the path yourself"
    Writing something like `Path(__file__).parent / "data" / "kenya-contacts.csv"`
    appears to work and then breaks: the moment you bump the version in
    `datasets.yaml`, your model is still reading the old file. Use
    `self.dataset()` and the version follows automatically.

If the file is not where the manifest says it is, you get told exactly what to
run:

```
kenya-contact-matrix@1 is declared in .../datasets.yaml but
.../data/kenya-contacts.csv does not exist. Download it with:
  python -m compartment.datasets pull kenya-contact-matrix@1 --dest .../data
```

This is normal on a fresh clone — data files are not stored in git, only your
`datasets.yaml` is. Run the `pull` it suggests.

### A complete example

The [MPOX model](https://github.com/WHO-Collaboratory/pandemic-simulator-compartment/blob/main/compartment/models/mpox_jax_model/model.py)
uses `self.dataset()` to load a time-varying schedule of infection and recovery
multipliers, interpolating between the elapsed-day checkpoints in the CSV.
Its
[manifest](https://github.com/WHO-Collaboratory/pandemic-simulator-compartment/blob/main/compartment/models/mpox_jax_model/datasets.yaml)
and small
[CSV](https://github.com/WHO-Collaboratory/pandemic-simulator-compartment/blob/main/compartment/models/mpox_jax_model/data/transition-rate-multipliers.csv)
are committed as a runnable example. The schedule is applied uniformly rather
than using administrative-zone lookups, so the model works with any solution's
country, zone names, number of zones, and start date.

That CSV is a deliberate exception to the normal rule above so automated tests
and fresh clones can run the example without downloading anything. Do not
commit ordinary model datasets; publish and pull those with the dataset
commands.

---

## Using a dataset someone else uploaded

See what is available:

```bash
python -m compartment.datasets list
```

```
kenya-contact-matrix  1             kenya-contacts.csv
kenya-contact-matrix  2             kenya-contacts.csv
uganda-admin-zones    1.0.0         uganda-zones.json
```

Download one:

```bash
python -m compartment.datasets pull kenya-contact-matrix@1 --dest data/
```

If you leave off the version, you get the **most recently uploaded** one:

```bash
python -m compartment.datasets pull kenya-contact-matrix --dest data/
```

!!! warning "Most recent is not the same as highest number"
    Without a version, you get whichever was uploaded last — not the biggest
    version number. If someone uploads version `1` after version `2`, you would
    get version `1`. **Name the version you want whenever it matters.**

---

## Rules to know

**Datasets never change.** Once `kenya-contact-matrix` version `1` exists, it is
fixed forever. If your data changes, upload it with a new version:

```yaml
datasets:
  - name: kenya-contact-matrix
    version: "2"          # was "1"
    file: data/kenya-contacts.csv
```

This is deliberate. Results from an old model run must always be reproducible,
so the data behind them cannot be replaced.

If you try to reuse a version, you get a clear message before anything is
uploaded:

> Dataset kenya-contact-matrix version 1 already exists. Datasets are immutable
> — bump the version in datasets.yaml.

**Naming.** Names and versions may use letters, numbers, dots, dashes and
underscores. They must start with a letter or number, and cannot contain
slashes. Names can be up to 128 characters, filenames up to 256.

**Size.** Up to **500 MB per dataset**. Every dataset your model declares is
built into your model's cloud image, so this is a limit on how big that image
gets, not on what the virus scanner can handle. If your file is bigger:
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
      "name": "kenya-contact-matrix",
      "version": "1",
      "filename": "kenya-contacts.csv"
    }
  ]
}
```

You do not have to do anything for this. It means anyone looking at a model run
later can see exactly which data it was built from. Models without a
`datasets.yaml` simply do not get this section.

This list is also what puts your data in the cloud. When your model is released,
the pipeline reads it, downloads each dataset, and builds them into the model's
image — which is why `self.dataset()` finds the file when the model runs. Two
consequences worth knowing:

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
| `over the 500 MB per-dataset limit` | The file is too big to build into a model image. | Shrink or split it. See **Size** above. |
| `does not declare 'name'` | The name you passed to `self.dataset()` is not in `datasets.yaml`. | The message lists the names that are declared — usually a typo. |
| `is declared ... but ... does not exist` | The manifest is right but the file is not downloaded. | Run the `pull` command in the message. Normal on a fresh clone. |
| `already exists. Datasets are immutable` | That version is published. | Bump the version. |
| `Session token is invalid or expired` | Your token has run out (about an hour). | Run the command again and paste a fresh token. |
| `was not issued by the expected Cognito user pool` | You copied the token from the wrong site. | Copy it from the same site the CLI opened for you, not a different environment. |
| `not a well-formed JWT` | The token got cut off when copying. | Copy it again, all of it. |

If a command keeps asking you to sign in, delete the saved token and try once
more:

```bash
rm ~/.pansim/dataset-session.json
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
