"""UoS-MobGap extensions for MobGap (optional: ``mobgap[uos]``).

This subpackage is maintained by UoS-MobGap and is not part of upstream MobGap.

Install (recommended: uv — https://docs.astral.sh/uv/):

    uv sync --extra uos

Or with pip: ``pip install 'mobgap[uos]'``
"""

from mobgap.data.uos.openmovement_cwa import (
    CalibrationFailurePolicy,
    CalibrationSource,
    CwaDtype,
    load_cwa_as_dataset,
)
from mobgap.data.uos.participant_metadata import (
    ParticipantMetadataSource,
    convert_mobilised_info_to_participant_metadata,
    load_participant_metadata,
    normalize_participant_metadata,
)

__all__ = [
    "CalibrationFailurePolicy",
    "CalibrationSource",
    "CwaDtype",
    "ParticipantMetadataSource",
    "convert_mobilised_info_to_participant_metadata",
    "load_cwa_as_dataset",
    "load_participant_metadata",
    "normalize_participant_metadata",
]
