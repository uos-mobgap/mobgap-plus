"""Tests for the UoS-MobGap per-hour, per-day, and per-week DMO aggregation."""

import numpy as np
import pandas as pd
import pytest

from mobgap.aggregation import MobilisedAggregator
from mobgap.aggregation.uos import (
    TOTAL_COLUMNS,
    MultiGranularAggregator,
    RecordingTimeline,
)

_SAMPLING_RATE_HZ = 10.0
# Wednesday 2024-03-06, so the ISO week starts on 2024-03-04
_START = pd.Timestamp("2024-03-06 00:00:00").timestamp()


def _timeline(hours: float = 48.0) -> RecordingTimeline:
    return RecordingTimeline.from_uniform(
        _START, int(hours * 3600 * _SAMPLING_RATE_HZ), _SAMPLING_RATE_HZ, timezone=None
    )


def _wb_dmos(offsets_h, cadence_spm, duration_s=20.0) -> pd.DataFrame:
    """Build a walking bout table with one bout at each given hour offset."""
    starts = (np.asarray(offsets_h) * 3600 * _SAMPLING_RATE_HZ).astype(int)
    return pd.DataFrame(
        {
            "start": starts,
            "duration_s": np.full(len(starts), duration_s, dtype=float),
            "cadence_spm": np.asarray(cadence_spm, dtype=float),
        },
        index=pd.Index(range(len(starts)), name="wb_id"),
    )


class TestWeighting:
    # two bouts in the first hour, one in the second, so the two weightings must disagree
    wb_dmos = _wb_dmos([0.1, 0.2, 1.1], [100.0, 110.0, 130.0])

    def test_equal_weighting_averages_the_hours(self):
        result = MultiGranularAggregator(time_bins=("day",), weighting="equal").aggregate(
            self.wb_dmos, timeline=_timeline()
        )

        day = result.aggregated_data_.loc["day"]
        assert day.loc[pd.Timestamp("2024-03-06"), "wb_all__cadence_spm__avg"] == pytest.approx((105.0 + 130.0) / 2)

    def test_pooled_weighting_averages_the_walking_bouts(self):
        result = MultiGranularAggregator(time_bins=("day",), weighting="pooled").aggregate(
            self.wb_dmos, timeline=_timeline()
        )

        day = result.aggregated_data_.loc["day"]
        assert day.loc[pd.Timestamp("2024-03-06"), "wb_all__cadence_spm__avg"] == pytest.approx(340.0 / 3)

    def test_totals_are_independent_of_the_weighting(self):
        wb_dmos = _wb_dmos(np.arange(0, 40, 0.7), np.linspace(90, 130, 58), duration_s=45.0)
        timeline = _timeline()

        equal = MultiGranularAggregator(weighting="equal").aggregate(wb_dmos, timeline=timeline).aggregated_data_
        pooled = MultiGranularAggregator(weighting="pooled").aggregate(wb_dmos, timeline=timeline).aggregated_data_

        totals = [column for column in equal.columns if column in TOTAL_COLUMNS]
        assert totals
        pd.testing.assert_frame_equal(equal[totals], pooled[totals])

    def test_equal_weighting_climbs_from_the_hours(self):
        timeline = _timeline()

        only_days = MultiGranularAggregator(time_bins=("day",)).aggregate(self.wb_dmos, timeline=timeline)
        all_bins = MultiGranularAggregator().aggregate(self.wb_dmos, timeline=timeline)

        pd.testing.assert_frame_equal(only_days.aggregated_data_, all_bins.aggregated_data_.loc[["day"]])


class TestOutputShape:
    def test_every_bin_is_reported_including_the_empty_ones(self):
        result = MultiGranularAggregator().aggregate(_wb_dmos([0.5], [100.0]), timeline=_timeline())

        aggregated = result.aggregated_data_
        assert aggregated.index.names == ["time_bin", "bin_start"]
        assert aggregated.groupby("time_bin").size().to_dict() == {"hour": 48, "day": 2, "week": 1}

        # an hour without walking is a real zero rather than a hole in the table
        empty_hour = ("hour", pd.Timestamp("2024-03-06 05:00:00"))
        assert aggregated.loc[empty_hour, "wb_all__count"] == 0
        assert aggregated.loc[empty_hour, "total_walking_duration_min"] == 0
        assert pd.isna(aggregated.loc[empty_hour, "wb_all__cadence_spm__avg"])

    def test_drop_partial_keeps_only_fully_covered_bins(self):
        # the recording starts at 12:00 and runs for 24 hours, so no day is fully covered
        timeline = RecordingTimeline.from_uniform(
            pd.Timestamp("2024-03-06 12:00:00").timestamp(), int(24 * 3600 * _SAMPLING_RATE_HZ), _SAMPLING_RATE_HZ
        )
        wb_dmos = _wb_dmos([0.5, 13.0], [100.0, 110.0])

        kept = MultiGranularAggregator(time_bins=("hour", "day"), drop_partial=True).aggregate(
            wb_dmos, timeline=timeline
        )
        dropped = MultiGranularAggregator(time_bins=("hour", "day"), drop_partial=False).aggregate(
            wb_dmos, timeline=timeline
        )

        assert kept.aggregated_data_.groupby("time_bin").size().to_dict() == {"hour": 24}
        assert dropped.aggregated_data_.groupby("time_bin").size().to_dict() == {"hour": 24, "day": 2}


class TestNestedAggregator:
    def test_mask_removes_implausible_walking_bouts(self):
        wb_dmos = _wb_dmos([0.1, 0.2], [100.0, 110.0])
        mask = pd.DataFrame(True, index=wb_dmos.index, columns=wb_dmos.columns)
        mask.loc[1, "cadence_spm"] = False

        result = MultiGranularAggregator(time_bins=("hour",)).aggregate(
            wb_dmos, timeline=_timeline(), wb_dmos_mask=mask
        )

        first_hour = result.aggregated_data_.loc[("hour", pd.Timestamp("2024-03-06 00:00:00"))]
        assert first_hour["wb_all__cadence_spm__avg"] == pytest.approx(100.0)
        # the bout itself is still counted, only its cadence is dropped
        assert first_hour["wb_all__count"] == 2

    def test_original_names_are_summed_and_averaged_alike(self):
        # the original names are the other half of TOTAL_COLUMNS and would otherwise never be exercised
        aggregator = MobilisedAggregator(**MobilisedAggregator.PredefinedParameters.cvs_dmo_data)
        wb_dmos = _wb_dmos([0.1, 0.2, 1.1], [100.0, 110.0, 130.0])

        result = MultiGranularAggregator(time_bins=("day",), aggregator=aggregator).aggregate(
            wb_dmos, timeline=_timeline()
        )

        day = result.aggregated_data_.loc[("day", pd.Timestamp("2024-03-06"))]
        assert day["wb_all_sum"] == 3
        assert day["cadence_all_avg"] == pytest.approx((105.0 + 130.0) / 2)

    def test_every_aggregated_column_is_classified(self):
        # a column that is neither a known total nor a known average would silently be averaged
        full_wb_dmos = _wb_dmos([0.1, 0.2], [100.0, 110.0]).assign(
            n_raw_initial_contacts=30,
            n_turns=2,
            walking_speed_mps=1.1,
            stride_length_m=1.2,
            stride_duration_s=1.0,
        )
        aggregator = MobilisedAggregator(**MobilisedAggregator.PredefinedParameters.single_recording)

        columns = set(aggregator.aggregate(full_wb_dmos).aggregated_data_.columns)

        assert columns == set(MobilisedAggregator.ALTERNATIVE_NAMES.values())
        assert TOTAL_COLUMNS & columns == {
            "wb_all__count",
            "total_walking_duration_min",
            "wb_all__n_raw_initial_contacts__sum",
            "wb_all__n_turns__sum",
            "wb_10_30__count",
            "wb_10__count",
            "wb_30__count",
            "wb_60__count",
        }


def test_unknown_weighting_is_rejected():
    # without the check a misspelled weighting silently falls through to pooled and returns wrong averages
    with pytest.raises(ValueError, match="Unknown weighting"):
        MultiGranularAggregator(weighting="mean").aggregate(_wb_dmos([0.1], [100.0]), timeline=_timeline())
