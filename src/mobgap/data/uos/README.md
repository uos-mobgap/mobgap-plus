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

Requires Python 3.11+ (pulls in [omcwa](https://github.com/uos-mobgap/omcwa)).

## Usage

```python
from mobgap.data.uos import load_cwa_as_dataset, load_participant_metadata
from mobgap.pipeline import MobilisedPipelineHealthy

dataset = load_cwa_as_dataset(
    "recording.cwa",
    {"height_m": 1.75, "sensor_height_m": 1.0, "cohort": "HA"},
    recording_metadata={"measurement_condition": "laboratory"},
    resample_hz=100.0,
)
pipeline = MobilisedPipelineHealthy().safe_run(dataset)
```

See the [Mobilise-D CWA walkthrough](https://github.com/uos-mobgap/mobilise-d_mobgap_tutorial) for a full notebook.
