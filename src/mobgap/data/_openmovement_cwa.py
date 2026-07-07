"""Load Open Movement CWA files into MobGap datasets."""

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from mobgap.consts import GRAV_MS2, SF_SENSOR_COLS
from mobgap.data._dataset_from_data import GaitDatasetFromData
from mobgap.data_transform import Resample

_REQUIRED_PARTICIPANT_METADATA_KEYS = ("height_m", "sensor_height_m", "cohort")
_COLUMN_RENAME = {
    "accel_x": "acc_x",
    "accel_y": "acc_y",
    "accel_z": "acc_z",
    "gyro_x": "gyr_x",
    "gyro_y": "gyr_y",
    "gyro_z": "gyr_z",
}


def _import_cwa_data() -> type[Any]:
    try:
        from openmovement.load import CwaData
    except ImportError as exc:
        raise ImportError(
            "The 'openmovement' package is required to load CWA files. "
            "Install it with: pip install 'mobgap[cwa]'"
        ) from exc
    return CwaData


def _validate_participant_metadata(participant_metadata: dict[str, Any]) -> None:
    missing = [key for key in _REQUIRED_PARTICIPANT_METADATA_KEYS if key not in participant_metadata]
    if missing:
        raise ValueError(f"participant_metadata is missing required keys for MobilisedPipelineHealthy: {missing}")


def _cwa_to_dataframe(cwa_data: Any, *, include_time_index: bool) -> tuple[pd.DataFrame, float]:
    sample_values = cwa_data.get_sample_values()
    df = pd.DataFrame(sample_values, columns=cwa_data.labels)
    df = df.rename(columns=_COLUMN_RENAME)

    for col in ("acc_x", "acc_y", "acc_z"):
        if col in df.columns:
            df[col] = df[col] * GRAV_MS2

    df = df[[col for col in SF_SENSOR_COLS if col in df.columns]]

    sampling_rate_hz = float(cwa_data.get_sample_rate())

    if include_time_index:
        # use utc unix seconds as a numeric index. On some hpc stacks
        # pandas DatetimeIndex and pd.to_datetime(..., unit="s") segfaults, float indices do not.
        start_time = cwa_data.get_start_time()
        df.index = pd.Index(
            start_time + np.arange(len(df)) / sampling_rate_hz,
            dtype=np.float64,
            name="time",
        )
    else:
        df.index = pd.RangeIndex(len(df), name="samples")

    return df, sampling_rate_hz


def load_cwa_as_dataset(
    path: Union[str, Path],
    participant_metadata: dict[str, Any],
    recording_metadata: Optional[dict[str, Any]] = None,
    *,
    sensor_position: str = "LowerBack",
    include_time_index: bool = False,
    resample_hz: Optional[float] = None,
) -> GaitDatasetFromData:
    """Load a CWA file into a :class:`~mobgap.data.GaitDatasetFromData`.

    Parameters
    ----------
    path
        Path to the ``.cwa`` file.
    participant_metadata
        Participant metadata required by :class:`~mobgap.pipeline.MobilisedPipelineHealthy`.
        Must contain ``height_m``, ``sensor_height_m``, and ``cohort``.
    recording_metadata
        Optional recording metadata merged with CWA-derived fields.
    sensor_position
        Sensor position label used as the key in the dataset sensor dictionary.
    include_time_index
        If True, use a UTC Unix-time index in seconds derived from the CWA start time.
        If False, use a :class:`~pandas.RangeIndex` (default).
    resample_hz
        Optional target sampling rate in Hz. When provided, sensor data is resampled
        before constructing the dataset.

    Returns
    -------
    GaitDatasetFromData
        Dataset keyed by the recording id (filename stem).

    Raises
    ------
    ImportError
        If the optional ``cwa`` dependency is not installed (``pip install 'mobgap[cwa]'``).
    ValueError
        If required participant metadata keys are missing.

    """
    _validate_participant_metadata(participant_metadata)

    cwa_data_cls = _import_cwa_data()
    path = Path(path)
    cwa_data = cwa_data_cls(
        str(path),
        include_time=False,
        include_accel=True,
        include_gyro=True,
        include_mag=False,
        include_light=False,
        include_temperature=False,
    )

    sensor_df, sampling_rate_hz = _cwa_to_dataframe(cwa_data, include_time_index=include_time_index)

    if resample_hz is not None:
        resampler = Resample(target_sampling_rate_hz=resample_hz, attempt_index_resample=include_time_index)
        sensor_df = resampler.transform(sensor_df, sampling_rate_hz=sampling_rate_hz).transformed_data_
        sampling_rate_hz = resample_hz

    recording_id = path.stem
    recording_meta = dict(recording_metadata or {})
    recording_meta.setdefault("cwa_source_path", str(path.resolve()))
    recording_meta.setdefault("cwa_start_time", cwa_data.get_start_time())

    return GaitDatasetFromData(
        {recording_id: {sensor_position: sensor_df}},
        sampling_rate_hz,
        _participant_metadata={recording_id: participant_metadata},
        _recording_metadata={recording_id: recording_meta},
        single_sensor_name=sensor_position,
        index_cols="recording_id",
    )
