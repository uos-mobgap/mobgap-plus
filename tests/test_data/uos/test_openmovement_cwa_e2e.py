"""End-to-end smoke for the MobGap CWA adapter with a real omcwa build."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("omcwa")

from mobgap.consts import SF_SENSOR_COLS
from mobgap.data.uos import load_cwa_as_dataset

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="omcwa requires Python 3.11+",
)

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "uos" / "cwa"
_CAL_SUCCESS = _FIXTURE_DIR / "cal_success.cwa"
_PARTICIPANT_METADATA = {
    "height_m": 1.75,
    "sensor_height_m": 1.0,
    "cohort": "HA",
}


@pytest.fixture(scope="module")
def cal_success_cwa() -> Path:
    if not _CAL_SUCCESS.is_file():
        pytest.skip(f"CWA fixture not found: {_CAL_SUCCESS}")
    return _CAL_SUCCESS


def test_load_cwa_as_dataset_processes_golden_fixture(cal_success_cwa: Path) -> None:
    """One real-file smoke: adapter loads, converts units, and attaches metadata."""
    dataset = load_cwa_as_dataset(
        cal_success_cwa,
        _PARTICIPANT_METADATA,
        recording_metadata={"measurement_condition": "laboratory"},
        resample_hz=100.0,
    )

    datapoint = dataset[0]
    assert datapoint.group_label.recording_id == "cal_success"
    assert datapoint.sampling_rate_hz == 100.0
    assert list(datapoint.data_ss.columns) == SF_SENSOR_COLS
    assert len(datapoint.data_ss) == 8_000
    assert datapoint.participant_metadata == _PARTICIPANT_METADATA
    assert datapoint.recording_metadata["measurement_condition"] == "laboratory"
    assert datapoint.recording_metadata["cwa_source_path"].endswith("cal_success.cwa")
    assert datapoint.recording_metadata["cwa_calibration_success"] is True
    assert datapoint.recording_metadata["cwa_calibration_error_code"] == 0
    # Acc was converted from g to m/s^2 (would stay ~1 if left in g).
    assert datapoint.data_ss[["acc_x", "acc_y", "acc_z"]].to_numpy().max() > 1.0
