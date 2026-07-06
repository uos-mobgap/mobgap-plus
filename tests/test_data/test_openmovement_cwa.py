import numpy as np
import pandas as pd
import pytest

from mobgap.consts import GRAV_MS2, SF_SENSOR_COLS
from mobgap.data._openmovement_cwa import _cwa_to_dataframe, load_cwa_as_dataset


class _StubCwaData:
    def __init__(self) -> None:
        self.labels = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
        self._sample_values = np.array(
            [
                [1.0, 0.0, 0.0, 10.0, 20.0, 30.0],
                [0.0, 1.0, 0.0, 11.0, 21.0, 31.0],
            ]
        )

    def get_sample_values(self) -> np.ndarray:
        return self._sample_values

    def get_sample_rate(self) -> float:
        return 100.0

    def get_start_time(self) -> float:
        return 1_700_000_000.0


class TestCwaAdapter:
    def test_cwa_to_dataframe_column_names_and_units(self):
        df, sampling_rate_hz = _cwa_to_dataframe(_StubCwaData(), include_time_index=False)

        assert list(df.columns) == SF_SENSOR_COLS
        assert sampling_rate_hz == 100.0
        assert isinstance(df.index, pd.RangeIndex)
        assert df.index.name == "samples"
        np.testing.assert_allclose(df["acc_x"], [GRAV_MS2, 0.0])
        np.testing.assert_allclose(df["acc_y"], [0.0, GRAV_MS2])
        np.testing.assert_allclose(df["gyr_x"], [10.0, 11.0])
        np.testing.assert_allclose(df["gyr_y"], [20.0, 21.0])
        np.testing.assert_allclose(df["gyr_z"], [30.0, 31.0])

    def test_missing_participant_metadata_raises(self):
        with pytest.raises(ValueError, match="height_m"):
            load_cwa_as_dataset(
                "recording.cwa",
                {"sensor_height_m": 1.0, "cohort": "HA"},
            )
