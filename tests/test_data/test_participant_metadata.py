"""Tests for MobGap-owned participant metadata normalisation."""

from pathlib import Path

import pandas as pd
import pytest

from mobgap.data import (
    convert_mobilised_info_to_participant_metadata,
    load_participant_metadata,
    normalize_participant_metadata,
)

_MOBGAP = {"height_m": 1.75, "sensor_height_m": 1.0, "cohort": "HA"}
_MOBILISED = {"Height": 175.0, "SensorHeight": 100.0, "Cohort": "HA"}


class TestConvertMobilisedInfo:
    def test_converts_cm_to_m_and_maps_extras(self):
        raw = {
            **_MOBILISED,
            "Handedness": "L",
            "FootSize": 26.0,
            "Weight": 70.0,
            "WalkingAid_01": 0,
            "INDIP_DataUsed": "all",
            "SensorAttachment_SU": "belt",
            "SensorType_SU": "MM+",
        }

        out = convert_mobilised_info_to_participant_metadata(raw)

        assert out["height_m"] == pytest.approx(1.75)
        assert out["sensor_height_m"] == pytest.approx(1.0)
        assert out["cohort"] == "HA"
        assert out["handedness"] == "left"
        assert out["foot_length_cm"] == 26.0
        assert out["weight_kg"] == 70.0
        assert out["walking_aid_used"] is False
        assert out["indip_data_used"] == "all"
        assert out["sensor_attachment_su"] == "belt"
        assert out["sensor_type_su"] == "MM+"

    def test_cohort_override(self):
        out = convert_mobilised_info_to_participant_metadata(_MOBILISED, cohort="PD")
        assert out["cohort"] == "PD"


class TestNormalizeParticipantMetadata:
    def test_mobgap_schema(self):
        assert normalize_participant_metadata(_MOBGAP) == _MOBGAP

    def test_prefers_mobgap_when_both_present(self):
        mixed = {**_MOBGAP, **_MOBILISED}
        with pytest.warns(UserWarning, match="both MobGap keys"):
            out = normalize_participant_metadata(mixed)
        assert out["height_m"] == 1.75
        assert out["sensor_height_m"] == 1.0
        assert out["cohort"] == "HA"
        assert out["Height"] == 175.0

    @pytest.mark.parametrize(
        ("fn", "payload"),
        [
            (normalize_participant_metadata, {"height_m": 1.75}),
            (
                convert_mobilised_info_to_participant_metadata,
                {"SensorHeight": 100.0, "Cohort": "HA"},
            ),
        ],
    )
    def test_missing_keys_raises(self, fn, payload):
        with pytest.raises(ValueError):
            fn(payload)


class TestLoadParticipantMetadata:
    def test_from_dataframe_mobilised_schema(self):
        out = load_participant_metadata(pd.DataFrame([_MOBILISED]))
        assert out == convert_mobilised_info_to_participant_metadata(_MOBILISED)

    def test_from_csv_mobilised_schema(self, tmp_path: Path):
        csv_path = tmp_path / "meta.csv"
        pd.DataFrame([_MOBILISED]).to_csv(csv_path, index=False)
        assert load_participant_metadata(csv_path) == convert_mobilised_info_to_participant_metadata(
            _MOBILISED
        )

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_participant_metadata(tmp_path / "missing.csv")

    def test_unsupported_suffix_raises(self, tmp_path: Path):
        path = tmp_path / "meta.json"
        path.write_text("{}")
        with pytest.raises(ValueError, match="Unsupported"):
            load_participant_metadata(path)

    def test_empty_dataframe_raises(self):
        with pytest.raises(ValueError, match="empty"):
            load_participant_metadata(pd.DataFrame())
