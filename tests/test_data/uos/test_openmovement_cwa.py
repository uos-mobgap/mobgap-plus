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
        self.n_samples = len(self.time)
        self.acc = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        self.gyr = np.array([[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]]) if has_gyro else None
        self.valid = np.array([True, True], dtype=np.bool_)
        self.clipped = np.array([False, False], dtype=np.bool_)
        self.metadata: dict[str, object] = {}
        self.calibration = MagicMock(
            success=calibration_success,
            error_code=calibration_error_code,
            num_axes=6,
            mean_svm_error=0.01,
        )

    @property
    def first_sample_time(self) -> float:
        return float(self.time[0])


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


class TestLoadCwaAsDataset:
    def test_missing_cwa_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="CWA file not found"):
            load_cwa_as_dataset(tmp_path / "missing.cwa", _PARTICIPANT_METADATA)

    @patch("mobgap.data.uos.openmovement_cwa._import_process_cwa")
    def test_missing_gyro_raises_before_masking_invalid_samples(self, mock_import_process_cwa, tmp_path):
        # AX3 recordings have no gyroscope; the check must fire before dropping invalid
        # samples wastes time masking acc/time for a recording that gets rejected anyway.
        stub = _StubProcessedRecording(has_gyro=False)
        stub.valid = np.array([False, True], dtype=np.bool_)
        mock_import_process_cwa.return_value = MagicMock(return_value=stub)
        cwa_path = tmp_path / "recording.cwa"
        cwa_path.write_bytes(b"")

        with pytest.raises(ValueError, match="no gyroscope data"):
            load_cwa_as_dataset(cwa_path, _PARTICIPANT_METADATA)

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
        # defaults match omcwa's own defaults, so a bare call picks up its behaviour unchanged
        assert mock_process_cwa.call_args.kwargs["calibration_source"] == "data"
        assert mock_process_cwa.call_args.kwargs["dtype"] == "float64"

        dataset = load_cwa_as_dataset(
            cwa_path,
            _PARTICIPANT_METADATA,
            recording_metadata={"measurement_condition": "laboratory"},
            include_time_index=True,
            resample_hz=50.0,
            calibration_source="player",
            dtype="float32",
        )
        assert mock_process_cwa.call_args.kwargs["sample_rate_hz"] == 50.0
        assert mock_process_cwa.call_args.kwargs["calibration_source"] == "player"
        assert mock_process_cwa.call_args.kwargs["dtype"] == "float32"

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
        assert datapoint.recording_metadata["cwa_calibration_num_axes"] == 6
        assert datapoint.recording_metadata["cwa_calibration_mean_svm_error"] == 0.01
        assert datapoint.recording_metadata["cwa_invalid_samples"] == 0
        assert datapoint.recording_metadata["cwa_invalid_samples_dropped"] == 0

    @patch("mobgap.data.uos.openmovement_cwa._import_process_cwa")
    def test_timezone_reaches_the_recording_timeline(self, mock_import_process_cwa, tmp_path):
        # the timezone is only reachable by the aggregator through this metadata key, so the two
        # halves of the extension are wired together by the string "timezone" and nothing else
        from mobgap.aggregation.uos import RecordingTimeline  # noqa: PLC0415

        mock_import_process_cwa.return_value = MagicMock(return_value=_StubProcessedRecording())
        cwa_path = tmp_path / "recording.cwa"
        cwa_path.write_bytes(b"")

        dataset = load_cwa_as_dataset(
            cwa_path,
            _PARTICIPANT_METADATA,
            include_time_index=True,
            timezone="Europe/London",
        )
        datapoint = dataset[0]

        assert datapoint.recording_metadata["timezone"] == "Europe/London"
        assert RecordingTimeline.from_datapoint(datapoint).timezone == "Europe/London"

    @patch("mobgap.data.uos.openmovement_cwa._import_process_cwa")
    def test_no_timezone_key_when_the_clock_is_already_local(self, mock_import_process_cwa, tmp_path):
        mock_import_process_cwa.return_value = MagicMock(return_value=_StubProcessedRecording())
        cwa_path = tmp_path / "recording.cwa"
        cwa_path.write_bytes(b"")

        dataset = load_cwa_as_dataset(cwa_path, _PARTICIPANT_METADATA)

        assert "timezone" not in dataset[0].recording_metadata

    @patch("mobgap.data.uos.openmovement_cwa._import_process_cwa")
    def test_empty_processed_recording_raises(self, mock_import_process_cwa, tmp_path):
        stub = _StubProcessedRecording()
        stub.time = np.array([], dtype=np.float64)
        stub.n_samples = 0
        stub.acc = np.empty((0, 3))
        stub.gyr = np.empty((0, 3))
        stub.valid = np.array([], dtype=np.bool_)
        stub.clipped = np.array([], dtype=np.bool_)
        mock_import_process_cwa.return_value = MagicMock(return_value=stub)
        cwa_path = tmp_path / "recording.cwa"
        cwa_path.write_bytes(b"")

        with pytest.raises(ValueError, match="no samples after processing"):
            load_cwa_as_dataset(cwa_path, _PARTICIPANT_METADATA)

    @patch("mobgap.data.uos.openmovement_cwa._import_process_cwa")
    def test_all_invalid_with_drop_invalid_raises(self, mock_import_process_cwa, tmp_path):
        stub = _StubProcessedRecording()
        stub.valid = np.array([False, False], dtype=np.bool_)
        mock_import_process_cwa.return_value = MagicMock(return_value=stub)
        cwa_path = tmp_path / "recording.cwa"
        cwa_path.write_bytes(b"")

        with pytest.raises(ValueError, match="no valid samples"):
            load_cwa_as_dataset(cwa_path, _PARTICIPANT_METADATA, drop_invalid=True)

    @patch("mobgap.data.uos.openmovement_cwa._import_process_cwa")
    def test_drop_invalid_removes_invalid_samples_by_default(self, mock_import_process_cwa, tmp_path):
        stub = _StubProcessedRecording()
        stub.valid = np.array([False, True], dtype=np.bool_)
        mock_import_process_cwa.return_value = MagicMock(return_value=stub)
        cwa_path = tmp_path / "recording.cwa"
        cwa_path.write_bytes(b"")

        dataset = load_cwa_as_dataset(cwa_path, _PARTICIPANT_METADATA)
        datapoint = dataset[0]

        assert len(datapoint.data_ss) == 1
        assert datapoint.recording_metadata["cwa_invalid_samples"] == 1
        assert datapoint.recording_metadata["cwa_invalid_samples_dropped"] == 1
        np.testing.assert_allclose(datapoint.data_ss["acc_y"], GRAV_MS2)
        np.testing.assert_allclose(datapoint.data_ss["gyr_x"], 11.0)

    @patch("mobgap.data.uos.openmovement_cwa._import_process_cwa")
    def test_drop_invalid_false_keeps_invalid_samples(self, mock_import_process_cwa, tmp_path):
        stub = _StubProcessedRecording()
        stub.valid = np.array([False, True], dtype=np.bool_)
        mock_import_process_cwa.return_value = MagicMock(return_value=stub)
        cwa_path = tmp_path / "recording.cwa"
        cwa_path.write_bytes(b"")

        dataset = load_cwa_as_dataset(
            cwa_path,
            _PARTICIPANT_METADATA,
            drop_invalid=False,
        )
        datapoint = dataset[0]

        assert len(datapoint.data_ss) == 2
        assert datapoint.recording_metadata["cwa_invalid_samples"] == 1
        assert datapoint.recording_metadata["cwa_invalid_samples_dropped"] == 0
