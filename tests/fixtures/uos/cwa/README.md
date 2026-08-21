# CWA test fixtures

`cal_success.cwa` is copied from the committed synthetic fixture in
[omcwa](https://github.com/uos-mobgap/omcwa) (`tests/fixtures/golden/`).
It contains deterministic accel and gyro data with a successful omconvert
auto-calibration fit.

Provenance and regeneration steps are documented in the omcwa repository.

Run UoS CWA tests (recommended: [uv](https://docs.astral.sh/uv/)):

```bash
uv sync --extra uos
uv run pytest tests/test_data/uos/ -q
```
