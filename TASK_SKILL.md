# TASK_SKILL.md — How to add a new task to PyHealth

**Audience:** an LLM (or engineer) implementing a *new prediction task* in this
repository. This file is a complete, self-contained specification. Read it end
to end before writing code, then follow the checklist in §12.

**Scope:** everything under `pyhealth/tasks/` — cohort definition, feature
extraction, labels, schemas, exports, docs, examples, and tests. It does not
cover adding new *datasets* (`pyhealth/datasets/`), *models*
(`pyhealth/models/`), or *processors* (`pyhealth/processors/`), though §5 tells
you how to pick an existing processor and §5.4 tells you when a new one is
genuinely required.

**Start here if your task is unusual.** §1–14 assume the label fits one of four
modes (`binary`, `multiclass`, `multilabel`, `regression`) under a single output
key. Most tasks do, and for those the task file is the whole job. If yours might
not — segmentation, censoring-aware time-to-event, token classification,
ranking, multi-head — run the three-question diagnostic in **§15.1 first**. It
takes a minute and it changes what `output_schema` should say. §15.3 maps common
archetypes to the work each actually requires.

---

## 1. The mental model

A **task** in PyHealth is a pure function from *one patient's full event
history* to *a list of ML-ready sample dictionaries*, plus a declaration of how
each field in those dictionaries should be turned into tensors.

```
BaseDataset (raw tables, all patients)
        │
        │  task.pre_filter(lazyframe)          ← optional, whole-dataset row filter
        ▼
   per-Patient iteration
        │
        │  task(patient) -> List[Dict]         ← YOUR CODE: cohort + features + label
        ▼
   SampleBuilder.fit(samples)                  ← fits processors from input/output_schema
        │
        │  processor.process(value) per field
        ▼
   SampleDataset (LitData .ld on disk)  →  get_dataloader  →  model  →  Trainer
```

Three consequences that shape every design decision:

1. **The task sees one patient at a time.** It cannot see other patients, and
   it must not depend on global state. Anything cohort-wide (e.g. "keep only
   patients with a T1DM code") goes in `pre_filter`, which operates on a Polars
   `LazyFrame` of the whole dataset, or is re-derived per patient.
2. **The task declares intent, processors do the encoding.** The task emits
   *raw Python values* (lists of code strings, `(timestamps, array)` tuples,
   file paths, floats). It never builds vocabularies, pads, tokenizes, or
   returns tensors. `input_schema` / `output_schema` name the processor that
   does that.
3. **The task defines the cohort.** Inclusion/exclusion criteria, the
   prediction time, the observation window, and the label all live in
   `__call__`. Returning `[]` for a patient silently drops them — that is the
   normal way to express "this patient does not meet inclusion criteria."

---

## 2. The `BaseTask` contract

`pyhealth/tasks/base_task.py`:

```python
class BaseTask(ABC):
    task_name: str
    input_schema: Dict[str, Union[str, Type]]
    output_schema: Dict[str, Union[str, Type]]

    def __init__(self, code_mapping: Optional[Dict[str, Tuple[str, str]]] = None): ...
    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame: return df
    @abstractmethod
    def __call__(self, patient) -> List[Dict]: ...
```

| Member | Required | Notes |
|---|---|---|
| `task_name` | yes | String, **identical to the class name**. Used for the on-disk cache directory. |
| `input_schema` | yes | `{sample_key: processor_spec}` for features. |
| `output_schema` | yes | `{sample_key: processor_spec}` for labels. Usually exactly one entry. |
| `__call__(patient)` | yes | Returns `List[Dict]`; `[]` means "exclude this patient". |
| `__init__` | optional | Only when the task is parameterized (window length, thresholds, padding). |
| `pre_filter(df)` | optional | Whole-dataset row filter applied before patient iteration. |

### 2.1 Class attributes vs. instance attributes

Declare the schemas as **class attributes** when they are fixed. New files must
use PEP 585 builtin generics wrapped in `ClassVar` (see §6.4 — the lint gate
flags `Dict[...]`/`List[...]` and bare mutable class defaults):

```python
task_name: str = "InHospitalMortalityMEDS"
input_schema: ClassVar[dict[str, str]] = {"codes": "sequence"}
output_schema: ClassVar[dict[str, str]] = {"mortality": "binary"}
```

Set them in `__init__` **only** when a constructor argument changes the
processor kwargs (see `MortalityPredictionStageNetMIMIC4`, `DKAPredictionMIMIC4`):

```python
def __init__(self, padding: int = 0):
    self.padding = padding
    self.input_schema = {
        "icd_codes": ("stagenet", {"padding": padding}),
        "labs": ("stagenet_tensor", {}),
    }
    self.output_schema = {"mortality": "binary"}
```

Any other constant lookup table (lab item IDs, ICD code sets, category order)
must also be annotated `ClassVar[...]`, or `RUF012` fires. Prefer immutable
containers (`tuple`, `frozenset`) where the constant is only ever read — it
documents the intent and makes accidental mutation impossible:

```python
AKI_ICD10_PREFIXES: ClassVar[tuple[str, ...]] = ("N17",)
AKI_ICD9_CODES: ClassVar[frozenset[str]] = frozenset({"5845", "5846"})
LAB_CATEGORIES: ClassVar[dict[str, list[str]]] = {...}
LABITEMS: ClassVar[list[str]] = [i for v in LAB_CATEGORIES.values() for i in v]
```

### 2.2 Parameterized tasks and `super().__init__()`

If you define `__init__` and want to keep `BaseTask`'s `code_mapping` support,
accept `code_mapping` explicitly and forward it (an explicit parameter documents
itself in the signature and in the API docs; `**kwargs` does not):

```python
def __init__(
    self,
    window: timedelta = timedelta(days=15),
    exclude_minors: bool = True,
    code_mapping: dict[str, tuple[str, str]] | None = None,
) -> None:
    super().__init__(code_mapping=code_mapping)
    self.window = window
    self.exclude_minors = exclude_minors
```

Validate constructor arguments eagerly and raise `ValueError` with a message
naming the bad value — that failure belongs at construction, not three hours
into a `set_task` run:

```python
if window_hours <= 0:
    raise ValueError(f"window_hours must be positive, got {window_hours}.")
```

Constructor arguments are hashed into the cache key (`set_task` serializes
`vars(task)` plus both schemas), so two differently-parameterized instances get
separate caches automatically. This only works if every constructor argument is
JSON-serializable via `default=str` — keep them to primitives, `timedelta`,
tuples, and small dicts.

### 2.3 `pre_filter`

Use it when the cohort is a small fraction of the dataset and per-patient
filtering would waste most of the work. It receives a `LazyFrame` whose columns
are `patient_id`, `timestamp`, `event_type`, and `"{event_type}/{attribute}"`
for every attribute. Always guard against the column being absent (the same
task may run on a dataset loaded with fewer tables):

```python
def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
    if "diagnoses_icd/icd_code" not in df.collect_schema().names():
        return df
    has_code = pl.col("diagnoses_icd/icd_code").is_in(CODES).fill_null(False)
    return (
        df.with_columns(has_code.alias("__hit"))
          .with_columns(pl.col("__hit").any().over("patient_id").alias("__keep"))
          .filter(pl.col("__keep"))
          .drop(["__hit", "__keep"])
    )
```

Note the `.over("patient_id")` window: filter *patients*, not *rows*, or you
will silently drop the rest of a kept patient's history.

---

## 3. Reading patient data

### 3.1 `Patient.get_events`

```python
patient.get_events(
    event_type: Optional[str] = None,   # e.g. "admissions", "labevents"
    start: Optional[datetime] = None,   # inclusive lower bound on timestamp
    end: Optional[datetime] = None,     # inclusive upper bound on timestamp
    filters: Optional[List[tuple]] = None,   # [(attr, op, value), ...], AND-ed
    return_df: bool = False,
) -> Union[List[Event], pl.DataFrame]
```

- Events come back **sorted by timestamp**.
- `filters` requires `event_type` to be set. Operators: `==`, `!=`, `<`, `<=`,
  `>`, `>=`. The attribute is resolved as the column `"{event_type}/{attr}"`.
- `return_df=False` → list of `Event` objects with attribute access
  (`event.icd_code`, `event.timestamp`).
- `return_df=True` → Polars `DataFrame` whose columns are **prefixed**:
  `"labevents/itemid"`, `"labevents/valuenum"`, plus bare `timestamp`,
  `patient_id`, `event_type`. Use this for anything vectorizable
  (pivots, aggregations, large lab pulls) — it is far faster than looping
  `Event` objects.

### 3.2 Naming rules that bite

- **All column names are lowercased at ingest.** Use `event.icd_code`, never
  `event.ICD_CODE`.
- **Timestamps are normalized to `event.timestamp`**, not the original column
  (`charttime`, `admittime`, `dischtime`). The originating column is named by
  the dataset YAML config, not by the task.
- **Non-timestamp datetimes are strings** and must be parsed:
  `datetime.strptime(admission.dischtime, "%Y-%m-%d %H:%M:%S")`. Keep them
  timezone-naive to match `event.timestamp`. `DTZ007` is enabled, so this
  *will* be flagged — add `# noqa: DTZ007` plus a one-line comment saying the
  source data carries no timezone.
- **The same clinical concept is named differently per dataset.** MIMIC-III
  uses `icd9_code`; MIMIC-IV uses `icd_code` (+ `icd_version`). Check
  `pyhealth/datasets/configs/{dataset}.yaml` — it is the authoritative list of
  event types and attributes for that dataset.

### 3.3 Event-type / timestamp cheat sheet

| Dataset | Config | Key event types | Timestamp source |
|---|---|---|---|
| MIMIC-III | `mimic3.yaml` | `patients`, `admissions`, `icustays`, `diagnoses_icd`, `procedures_icd`, `prescriptions`, `labevents`, `noteevents` | `admittime`, `dischtime` (for ICD), `startdate`, `charttime` |
| MIMIC-IV EHR | `mimic4_ehr.yaml` | `patients`, `admissions`, `icustays`, `diagnoses_icd`, `procedures_icd`, `prescriptions`, `labevents` | `admittime`, `dischtime` (for ICD), `starttime`, `charttime` |
| eICU | `eicu.yaml` | `patient`, `hospital`, `diagnosis`, `medication`, `treatment`, `lab`, `physicalexam`, `admissiondx` | mostly `null` (offset-based) |
| OMOP | `omop.yaml` | `person`, `visit_occurrence`, `death`, `condition_occurrence`, `procedure_occurrence`, `drug_exposure`, `measurement` | `*_start_datetime` / `*_datetime` |
| MEDS | `meds.yaml`, `meds_with_hadm.yaml` | flat `code`/`numeric_value` events | native |

**Critical MIMIC-specific fact:** `diagnoses_icd` and `procedures_icd` are
timestamped at **`dischtime`**, not at the time of care. Codes for an admission
are therefore only *knowable* at or after that admission's discharge. See §7.

---

## 4. Writing `__call__`

### 4.1 Canonical shape

```python
def __call__(self, patient: Any) -> list[dict[str, Any]]:
    samples = []

    # 1. Patient-level inclusion criteria (cheapest checks first)
    demographics = patient.get_events(event_type="patients")
    if not demographics:
        return []
    ...

    # 2. Iterate the unit of prediction (admission / stay / episode / record)
    for admission in patient.get_events(event_type="admissions"):

        # 3. Establish prediction time and observation window
        predict_time = admission.timestamp + timedelta(hours=48)

        # 4. Extract features strictly from [window_start, predict_time]
        ...

        # 5. Visit-level exclusion criteria
        if not conditions or not procedures:
            continue

        # 6. Derive the label from data strictly after predict_time
        label = int(admission.hospital_expire_flag)

        # 7. Emit one sample; keys must match the schemas exactly
        samples.append({
            "patient_id": patient.patient_id,
            "visit_id": admission.hadm_id,
            "conditions": conditions,
            "procedures": procedures,
            "mortality": label,
        })

    return samples
```

### 4.2 Reserved / conventional sample keys

| Key | Purpose |
|---|---|
| `patient_id` | **Always include.** Populates `patient_to_index`; required for `split_by_patient`. |
| `visit_id` or `record_id` | Include for visit-level tasks. Populates `record_to_index`. `record_id` takes precedence if both are present. |
| everything else | Must appear in `input_schema` or `output_schema`, or it is passed through unprocessed. |

`SampleBuilder.fit` asserts that every schema key is a subset of every sample's
keys — a missing key is a hard failure, not a warning. Extra keys are allowed
and passed through untouched (useful for metadata like `admission_id`), but
prefer declaring anything you don't want in the batch as `"ignore"`.

### 4.3 Granularity: one sample per what?

Pick deliberately and document it in the class docstring.

- **Visit-level** (`InHospitalMortalityMIMIC4`, `ReadmissionPrediction*`): one
  sample per qualifying admission. Include `visit_id`.
- **Patient-level** (`MortalityPredictionStageNetMIMIC4`, `DKAPredictionMIMIC4`):
  exactly one sample per patient, aggregating all visits. Return `[sample]`.
- **Sequential/cumulative** (`DrugRecommendation*`): one sample per visit, where
  features are the *nested* history of all prior visits. Build the per-visit
  samples first, then accumulate:

  ```python
  samples[0]["conditions"] = [samples[0]["conditions"]]
  for i in range(1, len(samples)):
      samples[i]["conditions"] = samples[i-1]["conditions"] + [samples[i]["conditions"]]
  ```

- **Sub-record** (`SleepStagingSleepEDF`): many samples per record (one per
  30-second epoch). Fine — just make sure `patient_id` is on each so splits stay
  patient-disjoint.

### 4.4 Robustness rules

- Wrap every string→number and string→datetime conversion the source data does
  not guarantee: `except (ValueError, TypeError, AttributeError): return []` or
  `continue`. Real EHR extracts have nulls everywhere.
- Never raise for ordinary bad data. Raise only for a genuine programming error
  (`assert len(demographics) == 1` is acceptable when the dataset config
  guarantees one row per patient).
- Return `[]` early and often. A task that crashes on 0.1% of patients fails the
  whole `set_task` run hours in.
- Do not mutate the `Patient` object or any class-level constant.

---

## 5. Schemas and processors

### 5.1 Processor spec forms

`input_schema` / `output_schema` values may be:

| Form | Example | When |
|---|---|---|
| string alias | `"sequence"` | default behaviour |
| `(alias, kwargs)` tuple | `("image", {"mode": "RGB"})` | pass constructor kwargs |
| `FeatureProcessor` subclass | `MyProcessor` | custom, no kwargs |
| `(cls, kwargs)` tuple | `(MyProcessor, {"k": 1})` | custom with kwargs |

### 5.2 Registered aliases and the raw value each expects

These are the exact strings accepted (from `@register_processor` in
`pyhealth/processors/`). **The middle column is what your `__call__` must put
in the sample dict.**

| Alias | Raw value your task emits | Result |
|---|---|---|
| `sequence` | `list[str]` of codes | `LongTensor[n]` of vocab indices |
| `nested_sequence` | `list[list[str]]` (visits × codes) | padded `LongTensor[V, max_len]` |
| `nested_sequence_floats` | `list[list[float]]` | padded float tensor |
| `deep_nested_sequence` | `list[list[list[str]]]` | 3-D padded index tensor |
| `deep_nested_sequence_floats` | `list[list[list[float]]]` | 3-D padded float tensor |
| `timeseries` | `(list[datetime], np.ndarray[T, F])` | resampled `FloatTensor[S, F]` |
| `temporal_timeseries` | same as `timeseries` | `{"value": Tensor[S,F], "time": Tensor[S]}` |
| `tensor` | number / nested list of numbers / `torch.Tensor` | `FloatTensor` |
| `stagenet` | `(list[float] times, list[str] or list[list[str]] codes)` | `(time_tensor, value_tensor)` |
| `stagenet_tensor` | `(list[float] times, list[list[float]] vectors)` | `(time_tensor, value_tensor)` |
| `text` | `str` | `str` (pass-through; embedded downstream) |
| `tuple_time_text` | `(list[str] texts, list[float] time_diffs)` | text + time tensors |
| `image` | file path `str` | `FloatTensor[C, H, W]` |
| `time_image` | `(list[path] paths, list[float] time_diffs)` | stacked image tensor + times |
| `audio` | file path `str` | waveform or mel-spectrogram tensor |
| `graph` | `list[str]` of medical codes | PyG `Data` subgraph (needs a `KnowledgeGraph` kwarg) |
| `multi_hot` | `list[str]` (input-side categoricals) | multi-hot `FloatTensor[V]` |
| `cehr` | CEHR-format event tuple | CEHR tensors |
| `raw` | anything | unchanged (avoid floats — LitData serialization issues) |
| `ignore` | anything | field dropped from the batch entirely |
| `binary` | `0/1`, `False/True`, or exactly 2 distinct values | `FloatTensor[1]` |
| `multiclass` | hashable label | `LongTensor` scalar |
| `multilabel` | `list[label]` | multi-hot `FloatTensor[num_classes]` |
| `regression` | number | `FloatTensor[1]` |

> **Tuple order is not uniform.** `stagenet` / `stagenet_tensor` /
> `timeseries` take **`(times, values)`**, while `tuple_time_text` and
> `time_image` take **`(values, times)`**. Check the processor's docstring
> before emitting a tuple.

### 5.3 Choosing the output schema

| Prediction target | Alias |
|---|---|
| yes/no outcome (mortality, readmission, DKA onset) | `binary` |
| one of K mutually exclusive classes (sleep stage, specialty) | `multiclass` |
| a *set* of labels (drug recommendation, ICD coding, multilabel CXR) | `multilabel` |
| continuous value (length of stay in days, risk score) | `regression` |

`binary` raises at fit time if the collected samples do not contain exactly two
distinct label values — if your cohort ends up single-class on a small test
fixture, that is a fixture problem, not a reason to switch to `multiclass`.

### 5.4 Do NOT write a new processor unless you must

Search `pyhealth/processors/` first. A new processor is justified only when the
raw value has a *shape* no existing processor accepts (a new modality, a new
temporal packing). Reformatting your data to fit an existing processor is
almost always correct and always cheaper. If you do add one, register it with
`@register_processor("name")`, import it in `pyhealth/processors/__init__.py`,
and add a docs entry under `docs/api/processors/` — that is a separate change
with its own review surface.

---

## 6. File, naming, and export conventions

### 6.1 File placement and naming

- One file per task family: `pyhealth/tasks/{snake_case_topic}.py`.
- Group closely-related dataset variants of the *same* task in one file
  (`drug_recommendation.py` holds MIMIC3/MIMIC4/eICU/OMOP variants).
- Give a substantially different formulation of the same outcome its own file
  (`mortality_prediction.py` vs `mortality_prediction_stagenet_mimic4.py` vs
  `in_hospital_mortality_mimic4.py`).

### 6.2 Class naming

`{Outcome}{Formulation?}{DATASET}` in PascalCase, with the dataset suffix in the
project's existing casing:

```
MortalityPredictionMIMIC4          LengthOfStayPredictioneICU
InHospitalMortalityMIMIC4          DrugRecommendationOMOP
MortalityPredictionStageNetMIMIC4  SleepStagingSleepEDF
```

`task_name` must equal the class name exactly.

### 6.3 Exporting from `pyhealth/tasks/__init__.py`

Every public task class must be importable as `from pyhealth.tasks import X`.
The module has **no `__all__`**, and the CI lint gate flags `F401` on newly
added import lines — so **new exports must use the redundant `X as X` form**:

```python
from .my_new_task import (
    MyNewTaskMIMIC4 as MyNewTaskMIMIC4,
    MyNewTaskMIMIC3 as MyNewTaskMIMIC3,
)
```

(Existing plain imports predate the gate; do not "fix" them, and do not
reformat neighbouring import blocks — reformatting re-flags pre-existing
`UP035`/`I001` violations on lines you touched.)

If the task's module has heavy or optional imports, use the lazy `__getattr__`
hook at the bottom of `__init__.py` instead:

```python
def __getattr__(name: str):
    if name == "MPFClinicalPredictionTask":
        from .mpf_clinical_prediction import MPFClinicalPredictionTask
        return MPFClinicalPredictionTask
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

### 6.4 Code style the lint gate actually enforces

`pyproject.toml` sets only `line-length = 88` and `target-version = "py313"`
under `[tool.ruff]` — it selects no rules, so the active ruleset comes from the
ruff version in use. CI pins `ruff~=0.15`; recent ruff builds enable a wide set
by default (`UP`, `RUF`, `DTZ`, `B`, `C4`, `SIM`, `I`, `BLE`, `S`, …). Running
`ruff check pyhealth/tasks/` today reports ~640 findings, all pre-existing —
they pass only because `tools/check_pr_rules.py` checks **added and modified
lines only**.

Two rules in that set are confirmed enforced by this repo's own history:
`pyhealth/tasks/__init__.py` carries an inline comment about `F401`/`UP035`/
`I001` on newly added lines, and `mortality_prediction_stagenet_mimic4.py`
carries a `# noqa: DTZ007`.

**Every line of a new file is an added line**, so write to the strict standard —
it is a superset of whatever CI's ruff enables and costs nothing.
`pyhealth/tasks/in_hospital_mortality_meds.py` (the most recently added task) is
the model: `ruff check` on it passes with zero findings. Match it.

What that means in practice:

| Rule | Don't | Do |
|---|---|---|
| `UP035`, `UP006` | `from typing import Dict, List`; `Dict[str, str]` | `dict[str, str]`, `list[str]` — no `typing` generics |
| `UP007`, `UP045` | `Optional[X]`, `Union[A, B]` | `X \| None`, `A \| B` |
| `RUF012` | `input_schema: dict[str, str] = {...}` | `input_schema: ClassVar[dict[str, str]] = {...}` |
| `DTZ007` | bare `datetime.strptime(...)` | keep it naive, add `# noqa: DTZ007` with a comment saying why |
| `BLE001` | `except Exception:` | `except (ValueError, TypeError, AttributeError):` |
| `S110` | `try: ... except ...: pass` | `continue` / `return []` with a comment |
| `I001` | ad-hoc import order | stdlib, then third-party, then local; alphabetized |
| `E501` | lines over 88 chars | wrap |

`from typing import Any, ClassVar` is normally the entire typing import a task
file needs.

Run the gate locally before opening the PR:

```bash
ruff check pyhealth/tasks/my_new_task.py
```

Do **not** reformat or "modernize" neighbouring untouched lines — doing so
converts pre-existing violations into *your* added lines and fails the gate.
This is exactly why the existing `Dict[str, str]` task files must be left
alone.

### 6.5 Optional: make it a dataset's `default_task`

Only for single-task datasets (imaging, signals, single-file datasets):

```python
@property
def default_task(self) -> COVID19CXRClassification:
    return COVID19CXRClassification()
```

EHR datasets support many tasks and have no default; leave them alone.

---

## 7. Data leakage — the non-negotiable section

Leakage is the single most common defect in task definitions, and this repo has
already shipped fixes for several. Treat every one of these as a hard
requirement, and **document the reasoning in the class docstring under a
`Data Leakage Prevention:` heading**.

### 7.1 The rule

> Every feature in a sample must be knowable at the prediction time. Nothing
> recorded at, after, or *as a consequence of* the outcome may appear in the
> features.

### 7.2 Recurring traps in this codebase

1. **Discharge-timestamped codes.** In MIMIC, `diagnoses_icd`/`procedures_icd`
   carry `dischtime`. For any task whose label comes from an admission's own
   outcome, **that admission's codes must be excluded**. Codes from earlier,
   already-resolved admissions are fine. (`MortalityPredictionStageNetMIMIC4`,
   `LengthOfStayStageNetMIMIC4`.)

2. **Terminal-stay measurements.** Labs drawn in the hours before death are
   near-perfect predictors of death and are not available at any realistic
   prediction time. Cap the target admission's feature window (e.g. the first
   48 hours after admission).

3. **Class-asymmetric restriction — the subtle one.** If you window the *death*
   class but leave survivors unrestricted, the model learns "more data →
   survived" from the volume of information alone, with zero clinical signal.
   **Apply the identical window to both classes.** A survivor's last admission
   must be treated exactly like a decedent's terminal admission. This exact bug
   shipped once and is now covered by
   `tests/core/test_stagenet_task_leakage_prevention.py` — read it before
   writing any patient-level EHR task.

4. **Target inside the history.** In drug recommendation, the visit's own drug
   list is the label; it must be blanked out of `drugs_hist`:
   `samples[i]["drugs_hist"][i] = []`.

5. **`storetime` vs `charttime`.** A lab can be *charted* before the prediction
   time but *stored* (i.e. become visible) after it. Filter on both:
   ```python
   df = df.filter(pl.col("labevents/storetime") <= predict_time)
   ```

6. **Post-outcome events of any kind.** Discharge disposition, death date,
   billing codes, discharge summaries — all are post-hoc. If in doubt about an
   attribute, check its `timestamp` column in the dataset YAML.

7. **Cross-patient leakage via splits.** Always emit `patient_id` so users can
   `split_by_patient`. A task emitting many samples per patient without
   `patient_id` forces sample-level splits and leaks the same patient across
   train/test.

### 7.3 Prove it with a test

Any leakage-sensitive decision must be locked in by a regression test on the
demo fixtures asserting that specific codes are **present** (earlier admissions)
and specific codes are **absent** (target admission). See §11.

---

## 8. Docstrings (CI-enforced)

`tools/check_pr_rules.py` fails the PR if any new or modified **module-level
public** class or function in `pyhealth/` lacks a `>>>` example in its
docstring. The check walks `ast.iter_child_nodes(module)` and skips names
starting with `_`, so *methods* (including `__call__`) and private helpers are
exempt — but the task class itself is not. Google style, 88-char lines.

Use this template verbatim:

```python
class MyNewTaskMIMIC4(BaseTask):
    """One-line summary of what is predicted, from what, on which dataset.

    Two or three sentences on the clinical framing and why the task is
    defined this way.

    Cohort:
        - Inclusion / exclusion criteria, one per line.

    Prediction Time:
        Exactly when the prediction is made, relative to the admission or
        episode, and what observation window feeds it.

    Data Leakage Prevention:
        - Which events are excluded and why (cite the timestamp column).
        - Confirm the restriction is applied identically across classes.

    Args:
        window (timedelta): What it controls and its default.
        code_mapping (Optional[Dict[str, Tuple[str, str]]]): Optional vocab
            mapping forwarded to :class:`BaseTask`.

    Raises:
        ValueError: When an argument is out of range.

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): ``conditions`` -- diagnosis codes
            for the visit.
        output_schema (Dict[str, str]): ``label`` -- binary indicator of
            the outcome.

    Examples:
        >>> from pyhealth.datasets import MIMIC4Dataset
        >>> from pyhealth.tasks import MyNewTaskMIMIC4
        >>> dataset = MIMIC4Dataset(
        ...     ehr_root="/path/to/mimic-iv/2.2",
        ...     ehr_tables=["diagnoses_icd", "procedures_icd", "labevents"],
        ... )
        >>> task = MyNewTaskMIMIC4()
        >>> samples = dataset.set_task(task)
    """
```

`__call__` still gets its own docstring with `Args:` / `Returns:` describing
the sample dict keys, and every private helper gets one too — the gate does not
require it, but Sphinx renders `:members:` and reviewers read them.

Note the docstring `Args:`/`Attributes:` types stay in the classic
`Dict[str, str]` / `Optional[...]` prose form used throughout the API docs;
that is documentation text, not annotations, and ruff does not see it. The
*annotations* in the signature must still be PEP 585/604 (§6.4).

---

## 9. Docs (CI-enforced)

**A PR touching `pyhealth/**/*.py` that does not touch `docs/**` fails CI.**

1. Create `docs/api/tasks/pyhealth.tasks.{ModuleOrClass}.rst`.

   Single class:
   ```rst
   pyhealth.tasks.MyNewTaskMIMIC4
   ==============================

   .. autoclass:: pyhealth.tasks.my_new_task.MyNewTaskMIMIC4
       :members:
       :undoc-members:
       :show-inheritance:
   ```

   Several classes in one module — name the file after the *module* and add a
   `Task Classes` section with one `autoclass` per variant (see
   `pyhealth.tasks.drug_recommendation.rst`).

   The underline must be at least as long as the title, or Sphinx warns.

2. Add a line to the `Available Tasks` toctree in `docs/api/tasks.rst`:
   ```rst
       My New Task (MIMIC-IV) <tasks/pyhealth.tasks.MyNewTaskMIMIC4>
   ```

3. If the task introduces a new schema key or a genuinely new pattern, extend
   the `Processor String Keys` table or the `Writing a Custom Task` section in
   `docs/api/tasks.rst`.

---

## 10. Examples (CI-enforced)

**A PR touching `pyhealth/**/*.py` that does not touch `examples/**` fails CI.**

Add a runnable end-to-end script under the matching subdirectory
(`examples/mortality_prediction/`, `examples/drug_recommendation/`,
`examples/length_of_stay/`, `examples/readmission/`, `examples/clinical_tasks/`,
…), or create one for a genuinely new task family.

Follow the existing shape:

```python
"""{Model} on {Dataset}: {task} benchmark.

Usage:
    python examples/{dir}/{name}.py
    python examples/{dir}/{name}.py --quick-test   # dev mode smoke test
"""
import os

from pyhealth.datasets import MIMIC4Dataset, get_dataloader, split_by_patient
from pyhealth.models import RNN
from pyhealth.tasks import MyNewTaskMIMIC4
from pyhealth.trainer import Trainer

EHR_ROOT = os.environ.get("EHR_ROOT", "/srv/local/data/physionet.org/files/mimiciv/2.2/")


def main():
    dataset = MIMIC4Dataset(
        ehr_root=EHR_ROOT,
        ehr_tables=["diagnoses_icd", "procedures_icd", "labevents"],
        dev=True,
    )
    samples = dataset.set_task(MyNewTaskMIMIC4())

    train_ds, val_ds, test_ds = split_by_patient(samples, [0.8, 0.1, 0.1])
    train_dl = get_dataloader(train_ds, batch_size=32, shuffle=True)
    val_dl = get_dataloader(val_ds, batch_size=32, shuffle=False)
    test_dl = get_dataloader(test_ds, batch_size=32, shuffle=False)

    model = RNN(dataset=samples)
    trainer = Trainer(model=model, metrics=["roc_auc", "pr_auc", "f1"])
    trainer.train(train_dataloader=train_dl, val_dataloader=val_dl,
                  epochs=5, monitor="roc_auc")
    print(trainer.evaluate(test_dl))


if __name__ == "__main__":
    main()
```

Keep the `if __name__ == "__main__":` guard — `set_task(num_workers>1)` spawns
processes and will otherwise re-import and re-run the script recursively.
Read paths from env vars with a sensible default; never hardcode a personal
path. Offer a `--quick-test` / `dev=True` path so the script is runnable
without the full dataset.

---

## 11. Tests

Add `tests/core/test_{task_name}.py`, run by
`python -m unittest discover -t tests -s tests/core -p 'test_*.py' -v`
(`make testall` in CI). Use the checked-in demo fixtures — never real PHI:

```
test-resources/core/mimic4demo/{hosp,icu}/*.csv
test-resources/core/mimic3demo/
test-resources/core/eicudemo/
test-resources/core/support2/, chestxray14/, physionet_deid/
```

Cover, at minimum:

1. **Schema declaration** — `task_name`, `input_schema`, `output_schema` are
   class attributes with the expected keys.
2. **Constructor defaults** — every parameter's default value.
3. **Sample schema** — every emitted sample has all declared keys plus
   `patient_id` (and `visit_id` for visit-level tasks).
4. **Cohort logic** — patients/visits that must be excluded are absent
   (minors, single-admission patients, missing-feature visits).
5. **Label correctness** — a hand-verified patient in the fixture yields the
   expected label.
6. **Parameter effects** — different constructor args change the samples
   (e.g. 5-day vs 15-day readmission window).
7. **Leakage regression** — specific codes present, specific codes absent, and
   symmetric treatment across classes (§7.3).

Two useful patterns:

```python
# Fast, no processors, no cache — call the task directly on a Patient:
patient = self.dataset.get_patient("10003")
samples = MyNewTaskMIMIC4()(patient)

# Full path through set_task — use a temp cache dir and close() in teardown:
cls.cache_dir = tempfile.TemporaryDirectory()
dataset = MIMIC4Dataset(ehr_root=..., ehr_tables=[...], cache_dir=cls.cache_dir.name)
cls.samples = dataset.set_task(MyNewTaskMIMIC4())
...
@classmethod
def tearDownClass(cls):
    cls.samples.close()
```

If the fixture lacks a case you need (a second admission, a specific ICD code),
**add rows to the demo CSVs** rather than weakening the assertion — but check
that you do not break the tests that already read those files.

---

## 12. Pre-PR checklist

Implementation
- [ ] Subclasses `BaseTask`; `task_name` == class name.
- [ ] `input_schema` / `output_schema` use registered aliases from §5.2.
- [ ] Schemas and constants annotated `ClassVar[...]`; parameterized schemas built in `__init__`.
- [ ] PEP 585/604 typing only (`dict[str, str]`, `X | None`); `from typing import Any, ClassVar` is usually the whole import.
- [ ] `super().__init__(code_mapping=code_mapping)` called if `__init__` is defined; arguments validated with `ValueError`.
- [ ] `__call__` returns `list[dict]`; `[]` for excluded patients; never raises on bad data.
- [ ] Every sample carries `patient_id` (+ `visit_id`/`record_id` where applicable).
- [ ] Sample keys are a superset of both schemas' keys.
- [ ] Bulk extraction uses `return_df=True` with `"{event_type}/{col}"` column names.
- [ ] Datetime strings parsed explicitly; naive throughout.

Leakage
- [ ] Prediction time and observation window stated in the docstring.
- [ ] Target admission's discharge-timestamped codes excluded.
- [ ] Feature window capped identically for **all** label classes.
- [ ] `storetime`-style visibility columns respected.
- [ ] `Data Leakage Prevention:` section written in the class docstring.

Integration
- [ ] Exported from `pyhealth/tasks/__init__.py` using the `X as X` form.
- [ ] `docs/api/tasks/pyhealth.tasks.*.rst` added.
- [ ] Entry added to the `Available Tasks` toctree in `docs/api/tasks.rst`.
- [ ] Runnable example added under `examples/`.
- [ ] `tests/core/test_*.py` added, covering §11's seven areas.

Gates
- [ ] `>>>` example in every new/modified public class and function docstring.
- [ ] `ruff check pyhealth/tasks/my_new_task.py` reports **zero** findings (a new file is all added lines — see §6.4).
- [ ] No untouched neighbouring lines reformatted (that would re-flag pre-existing violations as yours).
- [ ] `python tools/check_pr_rules.py --base <base_sha> --head HEAD` passes.
- [ ] `python -m unittest discover -t tests -s tests/core -p 'test_*.py'` passes.

---

## 13. Reference implementation

A complete, convention-compliant task. Use it as a starting template.

```python
"""AKI (acute kidney injury) prediction for MIMIC-IV."""

from datetime import datetime, timedelta
from typing import Any, ClassVar

import polars as pl

from .base_task import BaseTask


class AKIPredictionMIMIC4(BaseTask):
    """Predicts acute kidney injury within a hospital admission (MIMIC-IV).

    One sample is produced per qualifying hospital admission. Features are
    the diagnosis history from prior admissions plus the creatinine values
    observed in the first 24 hours; the label is whether an AKI diagnosis
    code is recorded for that admission.

    Cohort:
        - Adults (``anchor_age >= 18``).
        - Admissions lasting longer than the observation window.
        - Admissions with at least one in-window creatinine measurement.
        - Admissions with at least one prior admission (history is required).

    Prediction Time:
        ``observation_hours`` after admission. Only events occurring --
        and visible, per ``storetime`` -- at or before that instant are used.

    Data Leakage Prevention:
        - ``diagnoses_icd`` events are timestamped at ``dischtime`` (see
          ``pyhealth/datasets/configs/mimic4_ehr.yaml``), so an admission's
          own codes are only known at its discharge. They supply the label
          and are excluded from that admission's features; they enter the
          history only for *subsequent* admissions.
        - Labs are filtered on ``storetime <= predict_time`` as well as
          charttime, since a result charted in-window may only become
          visible afterwards.
        - The window is applied to every admission regardless of the label,
          so AKI and non-AKI cases have identical amounts of information
          available and the model cannot learn "more data -> no AKI" from
          feature volume alone.

    Args:
        observation_hours (int): Length of the observation window in hours.
            Defaults to 24.
        code_mapping (Optional[Dict[str, Tuple[str, str]]]): Optional vocab
            mapping forwarded to :class:`BaseTask`.

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): ``conditions`` -- diagnosis codes
            from prior admissions; ``creatinine`` -- in-window creatinine
            timeseries.
        output_schema (Dict[str, str]): ``aki`` -- binary AKI indicator.

    Raises:
        ValueError: If ``observation_hours`` is not positive.

    Examples:
        >>> from pyhealth.datasets import MIMIC4Dataset
        >>> from pyhealth.tasks import AKIPredictionMIMIC4
        >>> dataset = MIMIC4Dataset(
        ...     ehr_root="/path/to/mimic-iv/2.2",
        ...     ehr_tables=["diagnoses_icd", "labevents"],
        ... )
        >>> samples = dataset.set_task(AKIPredictionMIMIC4())
        >>> early = AKIPredictionMIMIC4(observation_hours=12)
    """

    task_name: str = "AKIPredictionMIMIC4"
    input_schema: ClassVar[dict[str, str]] = {
        "conditions": "sequence",
        "creatinine": "timeseries",
    }
    output_schema: ClassVar[dict[str, str]] = {"aki": "binary"}

    AKI_ICD10_PREFIXES: ClassVar[tuple[str, ...]] = ("N17",)
    AKI_ICD9_CODES: ClassVar[frozenset[str]] = frozenset(
        {"5845", "5846", "5847", "5848", "5849"}
    )
    CREATININE_ITEMIDS: ClassVar[tuple[str, ...]] = ("50912", "52546")

    def __init__(
        self,
        observation_hours: int = 24,
        code_mapping: dict[str, tuple[str, str]] | None = None,
    ) -> None:
        if observation_hours <= 0:
            raise ValueError(
                f"observation_hours must be positive, got {observation_hours}."
            )
        super().__init__(code_mapping=code_mapping)
        self.observation_hours = observation_hours

    def _is_aki(self, code: Any, version: Any) -> bool:
        """Returns True if an ICD code denotes acute kidney injury.

        Args:
            code (Any): Raw ``icd_code`` value; may be ``None``.
            version (Any): Raw ``icd_version`` value ("9" or "10").

        Returns:
            bool: Whether the code is an AKI code for its ICD version.
        """
        if not code:
            return False
        normalized = str(code).replace(".", "").strip().upper()
        if str(version) == "10":
            return normalized.startswith(self.AKI_ICD10_PREFIXES)
        if str(version) == "9":
            return normalized in self.AKI_ICD9_CODES
        return False

    def _creatinine_window(
        self, patient: Any, start: datetime, end: datetime
    ) -> pl.DataFrame:
        """Returns creatinine events charted and stored within a window.

        Args:
            patient (Any): Patient exposing ``get_events``.
            start (datetime): Inclusive window start.
            end (datetime): Inclusive window end and prediction time.

        Returns:
            pl.DataFrame: Columns ``timestamp`` and the numeric value,
            sorted by timestamp; empty if nothing qualifies.
        """
        labs = patient.get_events(
            event_type="labevents", start=start, end=end, return_df=True
        )
        labs = labs.filter(
            pl.col("labevents/itemid").is_in(list(self.CREATININE_ITEMIDS))
        )
        if labs.height == 0:
            return labs
        labs = labs.with_columns(
            pl.col("labevents/storetime").str.strptime(
                pl.Datetime, "%Y-%m-%d %H:%M:%S"
            )
        ).filter(pl.col("labevents/storetime") <= end)
        if labs.height == 0:
            return labs
        return labs.select(
            pl.col("timestamp"),
            pl.col("labevents/valuenum").cast(pl.Float64),
        ).sort("timestamp")

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Generates AKI prediction samples for a single patient.

        Args:
            patient (Any): Patient exposing ``get_events``.

        Returns:
            list[dict[str, Any]]: One sample per qualifying admission with
            ``patient_id``, ``visit_id``, ``conditions``, ``creatinine``
            and ``aki``. Empty if no admission qualifies.
        """
        demographics = patient.get_events(event_type="patients")
        if not demographics:
            return []
        try:
            if int(demographics[0].anchor_age) < 18:
                return []
        except (ValueError, TypeError, AttributeError):
            return []

        samples: list[dict[str, Any]] = []
        history: list[str] = []

        for admission in patient.get_events(event_type="admissions"):
            try:
                # MIMIC-IV timestamps are timezone-naive; kept naive to
                # compare against the (also naive) event timestamps.
                dischtime = datetime.strptime(  # noqa: DTZ007
                    admission.dischtime, "%Y-%m-%d %H:%M:%S"
                )
            except (ValueError, TypeError, AttributeError):
                continue

            predict_time = admission.timestamp + timedelta(
                hours=self.observation_hours
            )
            if dischtime <= predict_time:
                # Stay is shorter than the observation window.
                continue

            # Label: this admission's own discharge-timestamped codes.
            visit_codes = patient.get_events(
                event_type="diagnoses_icd",
                filters=[("hadm_id", "==", admission.hadm_id)],
            )
            label = int(
                any(
                    self._is_aki(
                        getattr(event, "icd_code", None),
                        getattr(event, "icd_version", None),
                    )
                    for event in visit_codes
                )
            )

            labs = self._creatinine_window(
                patient, admission.timestamp, predict_time
            )
            if labs.height > 0 and history:
                samples.append(
                    {
                        "patient_id": patient.patient_id,
                        "visit_id": admission.hadm_id,
                        # History only: this admission's codes are the label.
                        "conditions": list(history),
                        "creatinine": (
                            labs["timestamp"].to_list(),
                            labs.drop("timestamp").to_numpy(),
                        ),
                        "aki": label,
                    }
                )

            # Codes become knowable at this admission's discharge, so they
            # are safe history for every *subsequent* admission only.
            history.extend(
                event.icd_code
                for event in visit_codes
                if getattr(event, "icd_code", None)
            )

        return samples
```

---

## 14. Anti-patterns

| Don't | Do |
|---|---|
| Return tensors, build vocabularies, or pad inside `__call__` | Emit raw Python values; let the processor encode |
| Loop `Event` objects to pull thousands of labs | `return_df=True` and use Polars |
| Raise on malformed rows | `continue` / `return []` |
| Use `event.CHARTTIME` or `event.ICD_CODE` | Lowercase attributes; `event.timestamp` for the time |
| Window only the positive class | Apply the identical window to every class |
| Include the target admission's discharge-timestamped codes | Exclude them; use prior admissions |
| Omit `patient_id` | Always include it, so splits stay patient-disjoint |
| Add a plain `from .x import Y` to `tasks/__init__.py` | Use `from .x import Y as Y` |
| Skip docs/examples "for a small change" | CI hard-fails; both are required |
| Hardcode `/srv/local/...` in an example | `os.environ.get("EHR_ROOT", <default>)` |
| Invent a new processor for a reshapeable value | Reshape to fit an existing alias |
| Copy a lab-item list into a new file | Import or reference the existing constant where practical |
| Use `Dict`/`List`/`Optional` in a new file | `dict`/`list`/`X \| None` — the lint gate flags the old forms |
| `except Exception:` to be safe | Catch the specific exceptions; `BLE001` is on |

---

## 15. Task archetypes — when the task file is not the whole job

Sections 1–14 assume your label fits one of four **modes** (`binary`,
`multiclass`, `multilabel`, `regression`). Most tasks do, and for those the task
file is the entire deliverable. This section is the diagnostic for the ones that
don't, and the recipe for what else to build when the answer is "it doesn't".

Read it **before** writing the task, not after — the answer changes what
`output_schema` should say.

### 15.1 The three-question diagnostic

Ask these of *one sample's label*:

1. **Is there exactly one output key?**
2. **Does the label fit a scalar or a fixed-length vector** — shape `(1,)` or
   `(C,)` — with `C` identical for every sample?
3. **Is the objective plain BCE, cross-entropy, or MSE?**

- **All three yes** → Tier 1. Write the task file, stop. §1–14 is complete.
- **Any no** → Tier 2 or Tier 3. Continue below.

### 15.2 How PyHealth degrades — the `mode` contract

This is the single most important mechanic for non-standard tasks, and it is
not obvious from the code. The framework does **not** hard-fail on an
unrecognized output type; it silently falls back to a reduced contract.

`BaseModel.__init__` ([base_model.py:32](pyhealth/models/base_model.py:32))
resolves `self.mode` only when **both** conditions hold:

```python
if len(self.label_keys) == 1:                       # exactly one output key
    m = self._resolve_mode(dataset.output_schema[...])
    if m in {"binary", "multiclass", "multilabel", "regression"}:
        self.mode = m                               # ... and it is one of the four
# otherwise self.mode stays None (the try/except swallows failures)
```

`Trainer.evaluate` ([trainer.py:316](pyhealth/trainer.py:316)) then branches on
it:

```python
if self.model.mode is not None:
    ... metrics_fn = get_metrics_fn(mode); scores = metrics_fn(...)
else:
    ... scores = {"loss": loss_mean}          # loss-only evaluation
```

So `mode is None` is a **supported state**, not an error. What it costs you:

| Capability | `mode` set | `mode is None` |
|---|---|---|
| Training loop | works | works |
| Loss | `BaseModel.get_loss_function()` | **you define it in the model** |
| Output activation | `prepare_y_prob()` | **you define it** |
| Output size | `get_output_size()` | **you define it** |
| Evaluation | full metric suite | **loss only** |
| `monitor=` | any metric name | **must be `"loss"`** |

Two consequences to internalize:

- The four "blocking" helpers — `get_output_size`, `get_loss_function`,
  `prepare_y_prob` ([base_model.py:113](pyhealth/models/base_model.py:113),
  [:129](pyhealth/models/base_model.py:129),
  [:158](pyhealth/models/base_model.py:158)) and `get_metrics_fn`
  ([trainer.py:40](pyhealth/trainer.py:40)) — **raise only if you call them**.
  A model that computes its own loss and returns `{"loss": ..., "y_prob": ...}`
  from `forward` never touches them and trains fine.
- With `mode is None`, `Trainer.train(..., monitor="roc_auc")` raises
  `KeyError` at [trainer.py:233](pyhealth/trainer.py:233), because `scores`
  contains only `"loss"`. Use:

  ```python
  trainer.train(..., monitor="loss", monitor_criterion="min")
  ```

This is exactly how `DeIDNERTask` and `patient_linkage_mimic3` already operate.

### 15.3 Archetype map

Find your task here first.

| Archetype | Tier | `output_schema` | What you must build beyond the task file | Precedent |
|---|---|---|---|---|
| Binary outcome (mortality, readmission, AKI) | 1 | `{"y": "binary"}` | nothing | `InHospitalMortalityMIMIC4` |
| Multiclass (sleep stage, specialty) | 1 | `{"y": "multiclass"}` | nothing | `SleepStagingSleepEDF` |
| Multilabel (drug rec, ICD coding) | 1 | `{"y": "multilabel"}` | nothing | `DrugRecommendationMIMIC3` |
| Scalar regression (LOS in days) | 1 | `{"y": "regression"}` | nothing | `LengthOfStayPredictionMIMIC4` |
| **Image classification** | 1 | `{"y": "multiclass"}` | nothing — `("image", {...})` input is fully supported | `COVID19CXRClassification` |
| Image multilabel (CXR findings) | 1 | `{"y": "multilabel"}` | nothing | `ChestXray14MultilabelClassification` |
| Multimodal (image + codes + labs) | 1 | one label key | nothing; declare several input keys | `MultimodalMortalityPredictionMIMIC4` |
| Time-to-event, **fixed horizon** | 2 | `{"y": "binary"}` | nothing, but you must justify the reduction (§15.4a) | `SurvivalPreprocessSupport2` |
| Token classification / NER | 2 | `{"labels": TextProcessor}` | model owns alignment + loss; no metrics | `DeIDNERTask` |
| Ranking / linkage / retrieval | 2 | `{}` (empty) | model owns scoring entirely | `patient_linkage_mimic3` |
| Sequence generation | 2–3 | pass-through | model owns decoding; generative metrics exist | `EHRGeneration` |
| **Time-to-event, censoring-aware** | 3 | two keys or custom | Cox/AFT loss + C-index metric (§15.7) | none |
| **Semantic segmentation** | 3 | custom mask processor | processor + loss + Dice/IoU (§15.6) | none |
| Dense/spatial regression (depth, density map) | 3 | custom | same shape as segmentation | none |
| Multi-task (2+ prediction heads) | 3-lite | two+ keys | `mode` auto-`None`; model owns combined loss | none |

### 15.4 Tier 2 — reductions and escape hatches

Three legitimate ways to stay in the task file. Each costs something. **Pick one
only if you can name the cost, and write that cost in the class docstring.**

**(a) Reduce to a fixed horizon.** Turn "time until event" into "event within
`H`". This is what `SurvivalPreprocessSupport2` does, with the horizon baked
into the constructor (`time_horizon="2m"` / `"6m"`), and what
`CancerSurvivalPrediction` does more bluntly by predicting `vital_status` as a
binary and discarding time entirely.

The cost is **censoring**. A patient last seen at month 4 with no event is not a
negative for a 6-month horizon — their outcome is unknown. Your options, in
descending order of honesty:

- Drop subjects censored before `H` from the cohort. Unbiased for the retained
  population; shrinks it, and makes the cohort depend on follow-up length.
- Treat censored-before-`H` as negative. **Biases toward the null** and inflates
  apparent performance in cohorts with short follow-up. Only acceptable if you
  say so explicitly.
- Use several horizons as separate task instances and report the curve.

Whichever you pick, state it under a `Censoring:` heading in the docstring
alongside `Data Leakage Prevention:`.

**(b) Pass-through output.** Declare the label with `TextProcessor`, `raw`, or a
custom processor that returns the value unchanged; the model handles encoding,
alignment, and loss. `DeIDNERTask` uses
`output_schema = {"labels": TextProcessor}` for exactly this. Cost: `mode` is
`None`, so evaluation is loss-only and you compute real metrics yourself outside
the Trainer.

**(c) Empty output schema.** `patient_linkage_mimic3` declares
`output_schema = {}` — the task only builds candidate pairs, and the model
produces matching scores. Legal, and the cleanest option when "the label" isn't
a per-sample value at all. Same cost as (b).

### 15.5 Tier 3 — the four companion changes

A genuinely new output modality is **one task file plus up to four other
changes**. Each is a separate review surface; scope the PR accordingly.

| # | Change | Lands in | Required? |
|---|---|---|---|
| 1 | Output processor (raw label → tensor) | `pyhealth/processors/{name}_processor.py` + `@register_processor` + import in `processors/__init__.py` + `docs/api/processors/*.rst` | yes |
| 2 | Loss function | the model's `forward`, returning `{"loss": ...}` | yes |
| 3 | Metrics | `pyhealth/metrics/{name}.py` + export in `metrics/__init__.py` | yes, if you want more than loss |
| 4 | Mode threading (`get_loss_function`, `prepare_y_prob`, `get_metrics_fn`) | `models/base_model.py`, `trainer.py` | **optional** — see §15.2 |

Item 4 is the one people over-scope. You do **not** need to modify `BaseModel`
or `Trainer` to ship a Tier 3 task: leave `mode` as `None`, own the loss in your
model, and call your metric functions directly on `trainer.inference(...)`
outputs. Thread the new mode through only when the archetype is common enough to
deserve first-class Trainer support — and make that its own PR.

### 15.6 Worked example A — semantic segmentation

**Task file (easy).** Two file paths per sample:

```python
input_schema: ClassVar[dict] = {"image": ("image", {"mode": "RGB"})}
output_schema: ClassVar[dict] = {"mask": ("segmentation_mask", {"n_classes": 4})}
```

**Companion 1 — the mask processor.** You cannot reuse `ImageProcessor` for the
label. Two concrete defects:

- [image_processor.py:68](pyhealth/processors/image_processor.py:68) resizes
  with `transforms.Resize`, whose default interpolation is **bilinear**. On a
  label mask that invents class IDs that exist nowhere in the ground truth,
  along every object boundary. Masks require `InterpolationMode.NEAREST`, which
  is not exposed as a kwarg.
- [image_processor.py:71](pyhealth/processors/image_processor.py:71) applies
  `ToTensor()`, which divides 8-bit values by 255. Class ID `3` becomes
  `0.0118`. Masks require `PILToTensor` (or `torch.as_tensor`) so IDs survive
  intact as `int64`.

Sketch:

```python
@register_processor("segmentation_mask")
class SegmentationMaskProcessor(FeatureProcessor):
    def __init__(self, n_classes: int, size: int | None = 224) -> None:
        self.n_classes = n_classes
        self.size = size

    def process(self, value: str | Path) -> torch.Tensor:
        with Image.open(value) as img:
            img.load()
            if self.size is not None:
                img = img.resize(
                    (self.size, self.size), resample=Image.NEAREST
                )
            return torch.as_tensor(np.array(img), dtype=torch.long)

    def size(self) -> int:
        return self.n_classes          # so get_output_size() is meaningful
```

**Companion 2 — paired geometry.** Image and mask run through *independent*
processors, so nothing keeps a random crop or flip consistent between them. Do
geometric augmentation **inside the model or a collate wrapper**, where you hold
both tensors, not in the processors. Keep the processors deterministic.

**Companion 3 — metrics.** `pyhealth/metrics/` has no Dice or IoU (grepping the
package for either returns nothing). Add `pyhealth/metrics/segmentation.py` with
`dice_score` / `iou_score` and export from
[metrics/__init__.py](pyhealth/metrics/__init__.py).

**Companion 4 — the model.** A U-Net returning
`{"loss": dice_ce_loss(logits, y), "y_prob": logits.softmax(1), "y_true": y}`.
`mode` resolves to `"segmentation_mask"`, which is not in the four-set, so it
falls to `None` — training works, evaluation is loss-only, and you call
`dice_score` on `trainer.inference(...)` yourself. Use
`monitor="loss", monitor_criterion="min"`.

**One scaling caveat.** `Trainer.inference`
([trainer.py:305](pyhealth/trainer.py:305)) does
`np.concatenate(y_true_all, axis=0)`, materializing every ground-truth mask in
host RAM. At 512² × float32 that is ~1 MB per sample. Evaluate in chunks for
anything beyond a demo.

### 15.7 Worked example B — time-to-event (CKD progression)

The label is a **pair**: `(duration, event_observed)`. Nothing in the four modes
represents that, and the package currently contains **no** censoring or
concordance machinery — grepping `pyhealth/` for `concordance`, `c_index`,
`partial_likelihood`, or `censor` returns zero hits. Both existing "survival"
tasks are Tier 2 reductions, not survival models.

Two honest routes:

**Route 1 — Tier 2, fixed horizon.** `output_schema = {"ckd_progression": "binary"}`,
label = "reached stage ≥4 within 24 months of the index date". Everything in
§1–14 applies unchanged, plus the `Censoring:` docstring section from §15.4a.
Start here unless you specifically need time-resolved risk.

**Route 2 — Tier 3, censoring-aware.** Emit both parts as separate keys:

```python
output_schema: ClassVar[dict[str, str]] = {
    "duration": "regression",     # months from index date to event or censoring
    "event": "binary",            # 1 = progressed, 0 = right-censored
}
```

Two output keys means `len(self.label_keys) != 1`, so `mode` resolves to `None`
automatically (§15.2) — no framework change needed. The model then owns a Cox
partial-likelihood or AFT loss over the batch, and you add
`pyhealth/metrics/survival.py` with Harrell's / Uno's C-index.

**The leakage traps here are different from §7 and are the real risk.** Three
specific to time-to-event:

1. **Index date (landmark).** Every subject needs an explicit time origin — first
   eGFR below threshold, first CKD code, cohort entry. Features come from
   *before* it; follow-up starts *at* it. A task without a stated index date is
   almost certainly leaking.
2. **Immortal time bias.** If cohort eligibility requires something that can only
   happen after the index date (e.g. "had ≥2 eGFR measurements during
   follow-up"), you have guaranteed those subjects survived long enough to
   qualify. Define eligibility using information available at the index date
   only.
3. **Informative censoring via the extract.** Administrative censoring at the
   dataset's end date is fine. Censoring because a patient stopped appearing —
   which in EHR data often means they died or got sicker elsewhere — is not
   independent of the outcome. State which kind you have.

Document all three under `Index Date:` and `Censoring:` headings, the same way
§7 requires `Data Leakage Prevention:`.

### 15.8 Extension checklist (Tier 3 only)

On top of the §12 checklist:

- [ ] Diagnostic in §15.1 answered explicitly in the class docstring — say which
      tier this task is and why.
- [ ] If Tier 2: the reduction's cost (censoring, lost metrics) named in the
      docstring.
- [ ] New processor registered with `@register_processor`, imported in
      `processors/__init__.py`, `.rst` added under `docs/api/processors/`.
- [ ] Processor's `size()` returns something meaningful, or the model does not
      call `get_output_size()`.
- [ ] Model computes its own loss and never calls `get_loss_function()` /
      `prepare_y_prob()` while `mode is None`.
- [ ] New metrics module added and exported from `metrics/__init__.py`.
- [ ] Example uses `monitor="loss", monitor_criterion="min"` and computes real
      metrics from `trainer.inference(...)`.
- [ ] Test asserts `model.mode is None` (so a future framework change that
      silently starts resolving your mode is caught).
- [ ] Any `BaseModel`/`Trainer` mode threading split into a **separate PR**.
