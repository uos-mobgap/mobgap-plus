# UoS-MobGap data extensions

Optional OpenMovement CWA loading for MobGap. **Not part of upstream MobGap.**

Install the optional extra. [uv](https://docs.astral.sh/uv/) is the recommended tool:

```bash
uv sync --extra uos
```

Or with pip:

```bash
pip install 'mobgap[uos]'
```

Requires Python 3.11+. [omcwa](https://github.com/uos-mobgap/omcwa) is not on PyPI. The extra pins a
git URL to a commit and builds omcwa from source. That needs a C++ toolchain and CMake. To move to a
newer omcwa, bump the sha in the `uos` extra in `pyproject.toml` and re-run `uv lock`.

## Usage

```python
from mobgap.data.uos import load_cwa_as_dataset, load_participant_metadata
from mobgap.pipeline import MobilisedPipelineHealthy

dataset = load_cwa_as_dataset(
    "recording.cwa",
    {"height_m": 1.75, "sensor_height_m": 1.0, "cohort": "HA"},
    recording_metadata={"measurement_condition": "laboratory"},
    resample_hz=100.0,
    drop_invalid=True,
)
pipeline = MobilisedPipelineHealthy().safe_run(dataset)
```

By default, `load_cwa_as_dataset` drops samples that omconvert marked invalid after resampling
(`drop_invalid=True`). Recording metadata includes `cwa_invalid_samples` and
`cwa_invalid_samples_dropped`. Set `drop_invalid=False` to keep every resampled sample, including
startup gaps.

## Feeding the per-hour and per-day aggregation

`mobgap.aggregation.uos` places walking bouts on the wall clock. The loader must provide two things
that are off by default:

```python
dataset = load_cwa_as_dataset(
    "recording.cwa",
    {"height_m": 1.75, "sensor_height_m": 1.0, "cohort": "HA"},
    include_time_index=True,  # RecordingTimeline.from_datapoint raises without this
    timezone="Europe/London",  # only when the logger clock runs in UTC
)
```

`include_time_index=True` puts the Unix time of every sample in the index. After `drop_invalid` has
removed samples, that is the only way to turn a walking bout's sample number into a wall-clock time.
`timezone` is stored as recording metadata and read back by `RecordingTimeline.from_datapoint`. Leave
it out when the logger was set to the local time of the study site. That is how AX3/AX6 devices are
normally configured.

## Participant metadata

The cohort is optional in both schemas. Real Mobilise-D `infoForAlgo.mat` files do not contain it.
Dataset loaders take it from the folder hierarchy. Pass it with `metadata_cohort=`:

```python
load_cwa_as_dataset("recording.cwa", "infoForAlgo.mat", metadata_cohort="HA")
```

A missing cohort stays `None`. It does not become the string `"None"`. That matters because the
Mobilise-D pipelines check for `None` before applying cohort thresholds.

See the [Mobilise-D CWA walkthrough](https://github.com/uos-mobgap/mobilise-d_mobgap_tutorial) for a full notebook.
