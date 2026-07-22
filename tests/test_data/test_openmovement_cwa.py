"""Unit tests for the MobGap CWA adapter."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from mobgap.consts import GRAV_MS2, SF_SENSOR_COLS
from mobgap.data._openmovement_cwa import _recording_to_dataframe, load_cwa_as_dataset

_PARTICIPANT_METADATA = {"height_m": 1.75, "sensor_height_m": 1.0, "cohort": "HA"}


class _StubProcessedRecording:
    """Minimal stand-in for :class:`omcwa.types.ProcessedRecording`.

    Uses omcwa physical units: ``acc`` in g, ``gyr`` in deg/s. Defaults to two
    samples at 100 Hz with both acc and gyro present.
    """

    def __init__(
        self,
        *,
        has_gyro: bool = True,
        time: np.ndarray | None = None,
        calibration_success: bool = True,
        calibration_error_code: int = 0,
        metadata: dict | None = None,
    ) -> None:
        self.sample_rate_hz = 100.0
        self.time = (
            np.array([1_700_000_000.0, 1_700_000_000.01])
            if time is None
            else time
        )
        self.acc = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        self.gyr = (
            np.array(
                [
                    [10.0, 20.0, 30.0],
                    [11.0, 21.0, 31.0],
                ]
            )
            if has_gyro
            else None
        )
        self.calibration = MagicMock(
            success=calibration_success,
            error_code=calibration_error_code,
        )
        self.metadata = {} if metadata is None else metadata


class TestRecordingToDataframe:
    """Tests for ``_recording_to_dataframe`` unit/column/index conversion."""

    def test_column_names_and_units(self):
        df, sampling_rate_hz = _recording_to_dataframe(_StubProcessedRecording(), include_time_index=False)

        assert list(df.columns) == SF_SENSOR_COLS
        assert sampling_rate_hz == 100.0
        assert isinstance(df.index, pd.RangeIndex)
        assert df.index.name == "samples"
        np.testing.assert_allclose(df["acc_x"], [GRAV_MS2, 0.0])
        np.testing.assert_allclose(df["acc_y"], [0.0, GRAV_MS2])
        np.testing.assert_allclose(df["gyr_x"], [10.0, 11.0])
        np.testing.assert_allclose(df["gyr_y"], [20.0, 21.0])
        np.testing.assert_allclose(df["gyr_z"], [30.0, 31.0])

    def test_time_index(self):
        df, _ = _recording_to_dataframe(_StubProcessedRecording(), include_time_index=True)

        assert isinstance(df.index, pd.Index)
        assert not isinstance(df.index, pd.DatetimeIndex)
        assert df.index.name == "time"
        assert df.index.dtype == np.float64
        assert len(df.index) == 2
        np.testing.assert_allclose(df.index, [1_700_000_000.0, 1_700_000_000.01])

    def test_missing_gyro_raises(self):
        with pytest.raises(ValueError, match="no gyroscope data"):
            _recording_to_dataframe(_StubProcessedRecording(has_gyro=False), include_time_index=False)


class TestLoadCwaAsDataset:
    """Tests for ``load_cwa_as_dataset`` validation, wiring, and assembly."""

    def test_missing_participant_metadata_raises(self):
        with pytest.raises(ValueError, match="height_m"):
            load_cwa_as_dataset(
                "recording.cwa",
                {"sensor_height_m": 1.0, "cohort": "HA"},
            )

    @patch("mobgap.data._openmovement_cwa._import_process_cwa")
    def test_resample_hz_maps_to_sample_rate_hz(self, mock_import_process_cwa):
        mock_process_cwa = MagicMock(return_value=_StubProcessedRecording())
        mock_import_process_cwa.return_value = mock_process_cwa

        load_cwa_as_dataset("recording.cwa", _PARTICIPANT_METADATA)
        _, default_kwargs = mock_process_cwa.call_args
        assert default_kwargs["sample_rate_hz"] == 0.0

        mock_process_cwa.reset_mock()
        load_cwa_as_dataset("recording.cwa", _PARTICIPANT_METADATA, resample_hz=50.0)
        _, resample_kwargs = mock_process_cwa.call_args
        assert resample_kwargs["sample_rate_hz"] == 50.0

    @patch("mobgap.data._openmovement_cwa._import_process_cwa")
    def test_builds_dataset_and_recording_metadata(self, mock_import_process_cwa):
        mock_import_process_cwa.return_value = MagicMock(
            return_value=_StubProcessedRecording(
                calibration_success=False,
                calibration_error_code=-1,
            )
        )
        cwa_path = "/data/participant/recording.cwa"

        dataset = load_cwa_as_dataset(
            cwa_path,
            _PARTICIPANT_METADATA,
            recording_metadata={"measurement_condition": "laboratory"},
            sensor_position="LowerBack",
            include_time_index=True,
        )

        assert len(dataset) == 1
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
