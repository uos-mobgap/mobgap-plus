"""Unit tests for the MobGap CWA adapter (omcwa mocked)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from mobgap.consts import GRAV_MS2, SF_SENSOR_COLS
from mobgap.data.uos.openmovement_cwa import _recording_to_dataframe, load_cwa_as_dataset

_PARTICIPANT_METADATA = {"height_m": 1.75, "sensor_height_m": 1.0, "cohort": "HA"}


class _StubProcessedRecording:
    """Minimal stand-in for :class:`omcwa.types.ProcessedRecording`."""

    def __init__(
        self,
        *,
        has_gyro: bool = True,
        calibration_success: bool = True,
        calibration_error_code: int = 0,
    ) -> None:
        self.sample_rate_hz = 100.0
        self.time = np.array([1_700_000_000.0, 1_700_000_000.01])
        self.acc = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        self.gyr = np.array([[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]]) if has_gyro else None
        self.calibration = MagicMock(
            success=calibration_success,
            error_code=calibration_error_code,
        )


class TestRecordingToDataframe:
    def test_column_names_and_units(self):
        df, sampling_rate_hz = _recording_to_dataframe(_StubProcessedRecording(), include_time_index=False)

        assert list(df.columns) == SF_SENSOR_COLS
        assert sampling_rate_hz == 100.0
        assert isinstance(df.index, pd.RangeIndex)
        assert df.index.name == "samples"
        np.testing.assert_allclose(df["acc_x"], [GRAV_MS2, 0.0])
        np.testing.assert_allclose(df["acc_y"], [0.0, GRAV_MS2])
        np.testing.assert_allclose(df["gyr_x"], [10.0, 11.0])

    def test_time_index(self):
        df, _ = _recording_to_dataframe(_StubProcessedRecording(), include_time_index=True)

        assert df.index.name == "time"
        assert df.index.dtype == np.float64
        assert not isinstance(df.index, pd.DatetimeIndex)
        np.testing.assert_allclose(df.index, [1_700_000_000.0, 1_700_000_000.01])

    def test_missing_gyro_raises(self):
        with pytest.raises(ValueError, match="no gyroscope data"):
            _recording_to_dataframe(_StubProcessedRecording(has_gyro=False), include_time_index=False)


class TestLoadCwaAsDataset:
    def test_missing_participant_metadata_raises(self):
        with pytest.raises(ValueError, match="height_m"):
            load_cwa_as_dataset(
                "recording.cwa",
                {"sensor_height_m": 1.0, "cohort": "HA"},
            )

    def test_missing_cwa_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="CWA file not found"):
            load_cwa_as_dataset(tmp_path / "missing.cwa", _PARTICIPANT_METADATA)

    @patch("mobgap.data.uos.openmovement_cwa._import_process_cwa")
    def test_builds_dataset_and_forwards_resample(self, mock_import_process_cwa, tmp_path):
        mock_process_cwa = MagicMock(
            return_value=_StubProcessedRecording(
                calibration_success=False,
                calibration_error_code=-1,
            )
        )
        mock_import_process_cwa.return_value = mock_process_cwa
        cwa_path = tmp_path / "recording.cwa"
        cwa_path.write_bytes(b"")

        load_cwa_as_dataset(cwa_path, _PARTICIPANT_METADATA)
        assert mock_process_cwa.call_args.kwargs["sample_rate_hz"] == 0.0

        dataset = load_cwa_as_dataset(
            cwa_path,
            _PARTICIPANT_METADATA,
            recording_metadata={"measurement_condition": "laboratory"},
            include_time_index=True,
            resample_hz=50.0,
        )
        assert mock_process_cwa.call_args.kwargs["sample_rate_hz"] == 50.0

        datapoint = dataset[0]
        assert datapoint.group_label.recording_id == "recording"
        assert datapoint.sampling_rate_hz == 100.0
        assert list(datapoint.data_ss.columns) == SF_SENSOR_COLS
        assert datapoint.participant_metadata == _PARTICIPANT_METADATA
        assert datapoint.recording_metadata["measurement_condition"] == "laboratory"
        assert datapoint.recording_metadata["cwa_source_path"].endswith("recording.cwa")
        assert datapoint.recording_metadata["cwa_start_time"] == 1_700_000_000.0
        assert datapoint.recording_metadata["cwa_calibration_success"] is False
        assert datapoint.recording_metadata["cwa_calibration_error_code"] == -1
