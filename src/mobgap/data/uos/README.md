# UoS-MobGap data extensions

Optional OpenMovement CWA loading for MobGap. **Not part of upstream MobGap.**

Install the optional extra (recommended: [uv](https://docs.astral.sh/uv/)):

```bash
uv sync --extra uos
```

Or with pip:

```bash
pip install 'mobgap[uos]'
```

Requires Python 3.11+. [omcwa](https://github.com/uos-mobgap/omcwa) is not on PyPI, so the extra
carries its git URL pinned to a commit and builds it from source. That needs a C++ toolchain and
CMake. To move to a newer omcwa, bump the sha in the `uos` extra in `pyproject.toml` and re-run
`uv lock`.

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

By default, `load_cwa_as_dataset` drops omconvert-invalid resampled samples
(`drop_invalid=True`). Recording metadata includes `cwa_invalid_samples` and
`cwa_invalid_samples_dropped`. Set `drop_invalid=False` to keep the full
uniform timeline (including startup gaps).

## Feeding the per-hour and per-day aggregation

`mobgap.aggregation.uos` places walking bouts on the wall clock, and it needs two things from the
loader that are off by default:

```python
dataset = load_cwa_as_dataset(
    "recording.cwa",
    {"height_m": 1.75, "sensor_height_m": 1.0, "cohort": "HA"},
    include_time_index=True,     # RecordingTimeline.from_datapoint refuses anything else
    timezone="Europe/London",    # only when the logger clock runs in UTC
)
```

`include_time_index=True` puts the Unix time of every sample in the index, which is the only way a
walking bout's sample number can be turned into a wall-clock time after `drop_invalid` has removed
samples. `timezone` is stored as recording metadata and read back by
`RecordingTimeline.from_datapoint`. Leave it out when the logger was configured with the local time
of the study site, which is how AX3/AX6 devices are normally set up.

## Participant metadata

The cohort is optional in both schemas. Real Mobilise-D `infoForAlgo.mat` files do not carry it,
since the dataset loaders take it from the folder hierarchy, so pass it with `metadata_cohort=`:

```python
load_cwa_as_dataset("recording.cwa", "infoForAlgo.mat", metadata_cohort="HA")
```

A missing cohort stays `None` rather than becoming the string `"None"`, which matters because the
Mobilise-D pipelines check for `None` before applying cohort thresholds.

See the [Mobilise-D CWA walkthrough](https://github.com/uos-mobgap/mobilise-d_mobgap_tutorial) for a full notebook.
