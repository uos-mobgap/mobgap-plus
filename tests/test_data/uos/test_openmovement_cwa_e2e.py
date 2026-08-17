"""End-to-end smoke for the MobGap CWA adapter with a real omcwa build."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

pytest.importorskip("omcwa")

from mobgap.consts import SF_SENSOR_COLS
from mobgap.data.uos import load_cwa_as_dataset

if TYPE_CHECKING:
    from omcwa.types import ProcessedRecording

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


def test_drop_invalid_samples_with_real_processed_recording(cal_success_cwa: Path) -> None:
    """``_drop_invalid_samples`` must handle a real ``omcwa.ProcessedRecording``, not just the test stub.

    ``ProcessedRecording`` cannot be reconstructed with a masked ``time`` array (omcwa derives
    ``time`` from ``start_time``/``n_samples`` internally) -- this only exercises that contract
    when omconvert actually marks samples invalid, which the golden fixture never does on its own.
    """
    from omcwa import process_cwa  # noqa: PLC0415

    def _process_cwa_with_forced_invalid(*args: object, **kwargs: object) -> ProcessedRecording:
        out = process_cwa(*args, **kwargs)
        out.valid[:5] = False
        return out

    with patch(
        "mobgap.data.uos.openmovement_cwa._import_process_cwa",
        return_value=_process_cwa_with_forced_invalid,
    ):
        dataset = load_cwa_as_dataset(cal_success_cwa, _PARTICIPANT_METADATA, resample_hz=100.0)

    datapoint = dataset[0]
    assert len(datapoint.data_ss) == 8_000 - 5
    assert datapoint.recording_metadata["cwa_invalid_samples"] == 5
    assert datapoint.recording_metadata["cwa_invalid_samples_dropped"] == 5
