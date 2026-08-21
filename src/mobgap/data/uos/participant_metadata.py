"""Load and normalise participant metadata for MobGap pipelines.

UoS-MobGap extension (optional: ``mobgap[uos]``). Not part of upstream MobGap.

Accepts already-normalised MobGap dicts, Mobilise-D ``infoForAlgo`` raw fields,
CSV / DataFrame / Series tables, or paths to ``.mat`` / ``.csv`` sidecars.
All paths produce the same :class:`mobgap.data.base.ParticipantMetadata` shape
(plus optional Mobilise-D extras when present).
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Union

import pandas as pd

# The cohort keys are deliberately excluded from both schema probes.
# Mobilise-D dataset loaders take cohort from the folder/index, not from infoForAlgo.mat.
# Upstream types it Optional[str], so it arrives through the cohort argument or not at all.
_MOBGAP_HEIGHT_KEYS = ("height_m", "sensor_height_m")
_MOBILISED_HEIGHT_KEYS = ("Height", "SensorHeight")

ParticipantMetadataSource = Union[str, Path, Mapping[str, Any], pd.Series, pd.DataFrame]


def convert_mobilised_info_to_participant_metadata(
    meta_data: Mapping[str, Any],
    *,
    cohort: str | None = None,
) -> dict[str, Any]:
    """Convert raw Mobilise-D ``infoForAlgo`` fields to MobGap participant metadata.

    Parameters
    ----------
    meta_data
        Raw mapping with Mobilise-D keys such as ``Height``, ``SensorHeight``,
        and ``Cohort`` (heights in centimetres).
    cohort
        Optional cohort override. When omitted, ``meta_data["Cohort"]`` is used.

    Returns
    -------
    dict
        Normalised metadata with at least ``height_m``, ``sensor_height_m``, and
        ``cohort``, plus optional Mobilise-D extras when present in ``meta_data``.
    """
    missing = [key for key in _MOBILISED_HEIGHT_KEYS if key not in meta_data]
    if missing:
        raise ValueError(
            "Mobilise-D participant metadata is missing required keys "
            f"{missing}. Expected Height and SensorHeight (cm)."
        )

    # Mobilise-D dataset loaders often take cohort from the index instead, so
    # meta_data typically has no Cohort key at all. pd.isna also catches an
    # explicit None and an empty-CSV-cell NaN. None of those should become "None"/"nan".
    raw_cohort = cohort if cohort is not None else meta_data.get("Cohort")
    cohort_value = None if pd.isna(raw_cohort) else str(raw_cohort)

    walking_aid_raw = meta_data.get("WalkingAid_01", -1)
    try:
        walking_aid_used = {0: False, 1: True}.get(int(walking_aid_raw))
    except (TypeError, ValueError):
        walking_aid_used = None

    result: dict[str, Any] = {
        "cohort": cohort_value,
        "foot_length_cm": meta_data.get("FootSize"),
        "handedness": {"L": "left", "R": "right"}.get(meta_data.get("Handedness")),
        "height_m": float(meta_data["Height"]) / 100.0,
        "indip_data_used": meta_data.get("INDIP_DataUsed"),
        "sensor_attachment_su": meta_data.get("SensorAttachment_SU"),
        "sensor_height_m": float(meta_data["SensorHeight"]) / 100.0,
        "sensor_type_su": meta_data.get("SensorType_SU"),
        "walking_aid_used": walking_aid_used,
        "weight_kg": meta_data.get("Weight"),
    }
    return dict(sorted(result.items()))


def normalize_participant_metadata(raw: Mapping[str, Any], *, cohort: str | None = None) -> dict[str, Any]:
    """Normalise a single-participant mapping to MobGap participant metadata.

    Accepts either:

    - MobGap keys: ``height_m``, ``sensor_height_m`` (metres)
    - Mobilise-D keys: ``Height``, ``SensorHeight`` (centimetres)

    Both schemas treat the cohort as optional, in either its ``cohort`` or its
    ``Cohort`` spelling. Real ``infoForAlgo.mat`` files do not carry it, and
    upstream types it ``Optional[str]``. A missing, blank, or ``NaN`` cohort
    comes back as ``None`` rather than as the string ``"None"``/``"nan"``.

    Extra keys are preserved. Mobilise-D optional fields are mapped to the same
    names used by :class:`mobgap.data.MobilisedParticipantMetadata`.

    Parameters
    ----------
    raw
        Single-participant mapping in either schema.
    cohort
        Optional cohort override, used when ``raw`` has no (or a blank)
        ``Cohort``/``cohort`` value. Takes precedence over ``raw`` when given.
    """
    keys = set(raw)
    has_mobgap = set(_MOBGAP_HEIGHT_KEYS) <= keys
    has_mobilised = set(_MOBILISED_HEIGHT_KEYS) <= keys

    # Prefer the MobGap schema when present so mixed tables do not convert twice.
    if has_mobgap:
        if has_mobilised:
            warnings.warn(
                "participant_metadata contains both MobGap keys "
                f"{list(_MOBGAP_HEIGHT_KEYS)} and Mobilise-D keys "
                f"{list(_MOBILISED_HEIGHT_KEYS)}. "
                "Using the MobGap values (metres) and ignoring Mobilise-D "
                "Height/SensorHeight/Cohort (centimetres).",
                UserWarning,
                stacklevel=2,
            )
        result = dict(raw)
        result["height_m"] = float(result["height_m"])
        result["sensor_height_m"] = float(result["sensor_height_m"])
        raw_cohort = cohort if cohort is not None else result.get("cohort")
        result["cohort"] = None if pd.isna(raw_cohort) else str(raw_cohort)
        return result

    if has_mobilised:
        return convert_mobilised_info_to_participant_metadata(raw, cohort=cohort)

    missing_mobgap = [key for key in _MOBGAP_HEIGHT_KEYS if key not in keys]
    missing_mobilised = [key for key in _MOBILISED_HEIGHT_KEYS if key not in keys]
    raise ValueError(
        "participant_metadata must use MobGap keys "
        f"{list(_MOBGAP_HEIGHT_KEYS)} (metres) or Mobilise-D keys "
        f"{list(_MOBILISED_HEIGHT_KEYS)} (centimetres). "
        f"Missing MobGap keys: {missing_mobgap}. "
        f"Missing Mobilise-D keys: {missing_mobilised}."
    )


def _mapping_from_table(table: pd.Series | pd.DataFrame) -> dict[str, Any]:
    """Convert a pandas Series or DataFrame to a mapping."""
    if isinstance(table, pd.Series):
        return table.to_dict()

    if table.empty:
        raise ValueError("participant metadata table is empty.")

    return table.iloc[0].to_dict()


def _load_from_path(path: Path, *, time_measure: str | None, cohort: str | None) -> dict[str, Any]:
    """Load participant metadata from a Mobilise-D ``infoForAlgo.mat`` or a ``.csv`` file."""
    if not path.is_file():
        raise FileNotFoundError(f"Participant metadata file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".mat":
        from mobgap.data._mobilised_matlab_loader import (  # noqa: PLC0415
            load_mobilised_participant_metadata_file,
        )

        raw = load_mobilised_participant_metadata_file(path)
        if not raw:
            raise ValueError(f"No participant metadata found in {path}")

        if time_measure is None:
            selected = next(iter(raw.values()))
        elif time_measure not in raw:
            raise KeyError(f"time_measure {time_measure!r} not found in {path}. Available: {sorted(raw)}")
        else:
            selected = raw[time_measure]

        return normalize_participant_metadata(selected, cohort=cohort)

    if suffix == ".csv":
        return normalize_participant_metadata(_mapping_from_table(pd.read_csv(path)), cohort=cohort)

    raise ValueError(
        f"Unsupported participant metadata file type {suffix!r} for {path}. "
        "Use a Mobilise-D .mat (infoForAlgo) or a .csv file."
    )


def load_participant_metadata(
    source: ParticipantMetadataSource,
    *,
    time_measure: str | None = None,
    cohort: str | None = None,
) -> dict[str, Any]:
    """Load participant metadata from a path, table, or mapping.

    Parameters
    ----------
    source
        One of:

        - ``dict`` / mapping with MobGap or Mobilise-D keys
        - :class:`pandas.Series` or :class:`pandas.DataFrame` (first row)
        - path to a Mobilise-D ``infoForAlgo.mat`` or a ``.csv``

        CSV / DataFrame columns may use either schema. Mobilise-D height fields
        are converted from centimetres to metres, matching ``.mat`` behaviour.
    time_measure
        Optional first-level key when ``source`` is a Mobilise-D ``.mat`` file
        (usually ``TimeMeasure1``). Defaults to the first entry in the file.
    cohort
        Optional cohort override. Mobilise-D dataset loaders typically take
        cohort from the folder/dataset index rather than ``infoForAlgo.mat``
        itself, so real ``.mat`` sources usually need this to end up with a
        non-``None`` cohort. Takes precedence over any cohort present in
        ``source``.

    Returns
    -------
    dict
        Normalised participant metadata suitable for MobGap pipelines and
        :func:`mobgap.data.uos.load_cwa_as_dataset`.
    """
    if isinstance(source, (str, Path)):
        return _load_from_path(Path(source).expanduser(), time_measure=time_measure, cohort=cohort)

    if isinstance(source, (pd.Series, pd.DataFrame)):
        return normalize_participant_metadata(_mapping_from_table(source), cohort=cohort)

    if isinstance(source, Mapping):
        return normalize_participant_metadata(source, cohort=cohort)

    raise TypeError(
        "participant_metadata must be a mapping, pandas Series/DataFrame, "
        f"or path to a .mat/.csv file. Got {type(source)!r}."
    )
