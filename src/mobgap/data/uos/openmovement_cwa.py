"""Load Open Movement CWA files into MobGap datasets.

UoS-MobGap extension. Not part of upstream MobGap.
Requires ``omcwa`` package, which is installed by ``mobgap[uos]`` extra.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import numpy as np
import pandas as pd

from mobgap.consts import GRAV_MS2, SF_SENSOR_COLS
from mobgap.data._dataset_from_data import GaitDatasetFromData
from mobgap.data.uos.participant_metadata import (
    ParticipantMetadataSource,
    load_participant_metadata,
)

if TYPE_CHECKING:
    from omcwa.types import ProcessedRecording

CalibrationFailurePolicy = Literal["raise", "identity"]


@dataclass(frozen=True)
class _MaskedRecording:
    """The subset of ``ProcessedRecording`` fields still needed once invalid samples are dropped.

    ``ProcessedRecording`` is not reconstructable with a boolean-masked ``time`` array (omcwa
    derives ``time`` from ``start_time``/``n_samples`` internally), and ``valid``/``clipped``/
    ``metadata`` are never read again after masking, so this avoids both problems.
    """

    sample_rate_hz: float
    time: np.ndarray
    acc: np.ndarray
    gyr: Optional[np.ndarray]
    calibration: Any


def _import_process_cwa() -> Any:
    try:
        from omcwa import process_cwa  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "The 'omcwa' package is required to load CWA files. "
            "Install it with: uv sync --extra uos  (or pip install 'mobgap[uos]')"
        ) from exc
    return process_cwa


def _drop_invalid_samples(out: "ProcessedRecording") -> tuple[Union["ProcessedRecording", _MaskedRecording], int]:
    """Return a copy of ``out`` with omconvert-invalid samples removed."""
    invalid_count = int((~out.valid).sum())
    if invalid_count == 0:
        return out, 0

    if not np.any(out.valid):
        raise ValueError("CWA recording has no valid samples after processing.")

    mask = out.valid
    filtered = _MaskedRecording(
        sample_rate_hz=out.sample_rate_hz,
        time=out.time[mask],
        acc=out.acc[mask],
        gyr=None if out.gyr is None else out.gyr[mask],
        calibration=out.calibration,
    )
    return filtered, invalid_count


def _recording_to_dataframe(
    out: Union["ProcessedRecording", _MaskedRecording],
    *,
    include_time_index: bool,
) -> tuple[pd.DataFrame, float]:
    """Convert an ``omcwa`` recording into MobGap sensor columns and units."""
    if out.gyr is None:
        raise ValueError("CWA recording has no gyroscope data; MobGap requires acc + gyr.")

    df = pd.DataFrame(
        {
            "acc_x": out.acc[:, 0] * GRAV_MS2,
            "acc_y": out.acc[:, 1] * GRAV_MS2,
            "acc_z": out.acc[:, 2] * GRAV_MS2,
            "gyr_x": out.gyr[:, 0],
            "gyr_y": out.gyr[:, 1],
            "gyr_z": out.gyr[:, 2],
        },
        columns=SF_SENSOR_COLS,
    )

    sampling_rate_hz = float(out.sample_rate_hz)

    if include_time_index:
        # use utc unix seconds as a numeric index. On some hpc stacks
        # pandas DatetimeIndex and pd.to_datetime(..., unit="s") segfaults, float indices do not.
        df.index = pd.Index(out.time.astype(np.float64), name="time")
    else:
        df.index = pd.RangeIndex(len(df), name="samples")

    return df, sampling_rate_hz


def load_cwa_as_dataset(
    path: Union[str, Path],
    participant_metadata: ParticipantMetadataSource,
    recording_metadata: Optional[dict[str, Any]] = None,
    *,
    sensor_position: str = "LowerBack",
    include_time_index: bool = False,
    resample_hz: Optional[float] = None,
    calibrate: bool = True,
    on_calibration_failure: CalibrationFailurePolicy = "raise",
    drop_invalid: bool = True,
    time_range: Optional[tuple[float, float]] = None,
    metadata_time_measure: Optional[str] = None,
    metadata_cohort: Optional[str] = None,
) -> GaitDatasetFromData:
    """Load a CWA file into a :class:`mobgap.data.GaitDatasetFromData`.

    This is a thin adapter over :func:`omcwa.process_cwa`. Decoding,
    auto-calibration, and resampling are handled by ``omcwa``. This function
    converts the result into MobGap column names/units and wraps it in a
    :class:`mobgap.data.GaitDatasetFromData` ready for mobgap pipelines.

    Parameters
    ----------
    path
        Path to the ``.cwa`` file.
    participant_metadata
        Participant metadata required by mobgap pipelines. Accepted forms:

        - mapping / ``dict`` with MobGap keys ``height_m``, ``sensor_height_m``,
          ``cohort`` (meters), or Mobilise-D keys ``Height``, ``SensorHeight``,
          ``Cohort`` (centimetres)
        - :class:`pandas.Series` / :class:`pandas.DataFrame` (first row)
        - path to a Mobilise-D ``infoForAlgo.mat`` or a ``.csv`` with either schema

        All forms are normalised by :func:`mobgap.data.uos.load_participant_metadata`.
    recording_metadata
        Optional recording metadata merged with CWA-derived fields listed in
        ``Notes`` below. User-supplied keys are preserved. Adapter keys are
        added only when absent.
    sensor_position
        Sensor position label used as the key in the dataset sensor dictionary.
    include_time_index
        If True, use a float Unix-time index in seconds from the processed
        recording. If False, use a :class:`pandas.RangeIndex` (default).
    resample_hz
        Optional target sampling rate in Hz. When provided, ``omcwa`` resamples
        before constructing the dataset. When omitted, the file default rate is used.
    calibrate
        When True (default), run omconvert auto-calibration before resampling.
        When False, skip calibration and use identity coefficients.
    on_calibration_failure
        Policy when auto-calibration fails. ``"raise"`` (default) raises
        :class:`omcwa.CalibrationError`. ``"identity"`` continues with
        omconvert's identity fallback.
    drop_invalid
        When True (default), remove resampled samples that omconvert marks as
        invalid (``ProcessedRecording.valid`` is False). This matches
        omconvert/OmGUI analytics, which skip invalid gaps rather than using
        them in summaries. Set to False to keep the full uniform timeline,
        including startup gaps and other periods without sensor data.
    time_range
        Optional half-open ``(start, stop)`` window in Unix seconds forwarded to
        :func:`omcwa.process_cwa`. Calibration and resampling still process the
        full session. The range trims the completed output.
    metadata_time_measure
        Optional first-level key when ``participant_metadata`` is a Mobilise-D
        ``.mat`` file. Defaults to the first entry in the file.
    metadata_cohort
        Optional cohort override for ``participant_metadata``, taking
        precedence over any cohort present in ``participant_metadata``.
        Mobilise-D dataset loaders take cohort from the folder/dataset index
        rather than ``infoForAlgo.mat`` itself, so a real ``.mat`` source
        usually needs this to end up with a non-``None`` cohort.

    Returns
    -------
    GaitDatasetFromData
        Dataset with one recording, keyed by the filename stem.
        Sensor data are available via ``datapoint.data_ss``.

    Raises
    ------
    ImportError
        If the optional ``uos`` dependency is not installed (``uv sync --extra uos`` or ``pip install 'mobgap[uos]'``).
    FileNotFoundError
        If the ``.cwa`` path or a metadata file path does not exist.
    ValueError
        If required participant metadata keys are missing, gyroscope data is absent,
        or processing produced no samples.
    CalibrationError
        If ``calibrate=True``, ``on_calibration_failure="raise"``, and omconvert
        auto-calibration fails. See :func:`omcwa.process_cwa`.

    Notes
    -----
    Install (recommended: uv — https://docs.astral.sh/uv/):

        ``uv sync --extra uos``

    Or: ``pip install 'mobgap[uos]'``

    Output units: accelerometer columns are in m/s^2, gyroscope columns are 
    in deg/s. Column names follow :data:`mobgap.consts.SF_SENSOR_COLS`.

    Sensor requirement: MobGap requires accelerometer and gyroscope data.
    Recordings without gyroscope channels (AX3 recordings) raise
    ``ValueError``.

    Recording metadata added by this adapter (unless already present in
    ``recording_metadata``):

    - ``cwa_source_path`` — absolute path to the source ``.cwa`` file
    - ``cwa_start_time`` — Unix timestamp in seconds of the first processed sample
    - ``cwa_calibration_success`` — omconvert auto-calibration success flag
    - ``cwa_calibration_error_code`` — omconvert error code (0 on success)
    - ``cwa_invalid_samples`` — count of invalid samples before optional dropping
    - ``cwa_invalid_samples_dropped`` — invalid samples removed when
      ``drop_invalid=True`` (0 when ``drop_invalid=False``)

    For resampling algorithm details, calibration behaviour, and failure codes,
    see the ``omcwa`` documentation.

    Examples
    --------
    >>> from mobgap.data.uos import load_cwa_as_dataset
    >>> from mobgap.pipeline import MobilisedPipelineHealthy
    >>> participant = {
    ...     "height_m": 1.75,
    ...     "sensor_height_m": 1.0,
    ...     "cohort": "HA",
    ... }
    >>> dataset = load_cwa_as_dataset(
    ...     "recording.cwa",
    ...     participant,
    ...     resample_hz=100,
    ... )
    >>> pipeline = MobilisedPipelineHealthy().safe_run(dataset)

    """
    participant_metadata = load_participant_metadata(
        participant_metadata,
        time_measure=metadata_time_measure,
        cohort=metadata_cohort,
    )

    process_cwa = _import_process_cwa()
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"CWA file not found: {path}")

    out = process_cwa(
        path,
        sample_rate_hz=resample_hz if resample_hz is not None else 0.0,
        calibrate=calibrate,
        on_calibration_failure=on_calibration_failure,
        time_range=time_range,
    )

    if len(out.time) == 0:
        raise ValueError("CWA recording produced no samples after processing.")

    invalid_count = int((~out.valid).sum())
    if drop_invalid:
        out, dropped_invalid = _drop_invalid_samples(out)
    else:
        dropped_invalid = 0

    sensor_df, sampling_rate_hz = _recording_to_dataframe(out, include_time_index=include_time_index)

    recording_id = path.stem
    recording_meta = dict(recording_metadata or {})
    recording_meta.setdefault("cwa_source_path", str(path.resolve()))
    recording_meta.setdefault("cwa_start_time", float(out.time[0]))
    recording_meta.setdefault("cwa_invalid_samples", invalid_count)
    recording_meta.setdefault("cwa_invalid_samples_dropped", dropped_invalid)

    recording_meta.setdefault("cwa_calibration_success", out.calibration.success)
    recording_meta.setdefault("cwa_calibration_error_code", out.calibration.error_code)

    return GaitDatasetFromData(
        {recording_id: {sensor_position: sensor_df}},
        sampling_rate_hz,
        _participant_metadata={recording_id: participant_metadata},
        _recording_metadata={recording_id: recording_meta},
        single_sensor_name=sensor_position,
        index_cols="recording_id",
    )
