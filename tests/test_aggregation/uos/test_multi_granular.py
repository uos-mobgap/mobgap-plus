"""Tests for the UoS-MobGap per-hour and per-day DMO aggregation."""

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
_START = pd.Timestamp("2026-08-17 00:00:00").timestamp()


def _timeline(hours: float = 48.0) -> RecordingTimeline:
    return RecordingTimeline.from_uniform(
        _START, int(hours * 3600 * _SAMPLING_RATE_HZ), _SAMPLING_RATE_HZ, timezone=None
    )


def _gapped_timeline(hours: float, gap_h: tuple[float, float]) -> RecordingTimeline:
    """Build a timeline of a recording that lost every sample within ``gap_h``."""
    times = _START + np.arange(int(hours * 3600 * _SAMPLING_RATE_HZ)) / _SAMPLING_RATE_HZ
    lost = (times >= _START + gap_h[0] * 3600) & (times < _START + gap_h[1] * 3600)
    return RecordingTimeline.from_sample_times(times[~lost])


def _wb_dmos(offsets_h, cadence_spm, duration_s=20.0, timeline=None) -> pd.DataFrame:
    """Build a walking bout table with one bout at each given hour offset."""
    if timeline is None or timeline.sample_times is None:
        starts = (np.asarray(offsets_h) * 3600 * _SAMPLING_RATE_HZ).astype(int)
    else:
        starts = np.searchsorted(timeline.sample_times, _START + np.asarray(offsets_h) * 3600)
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
        assert day.loc[pd.Timestamp("2026-08-17"), "wb_all__cadence_spm__avg"] == pytest.approx((105.0 + 130.0) / 2)

    def test_pooled_weighting_averages_the_walking_bouts(self):
        result = MultiGranularAggregator(time_bins=("day",), weighting="pooled").aggregate(
            self.wb_dmos, timeline=_timeline()
        )

        day = result.aggregated_data_.loc["day"]
        assert day.loc[pd.Timestamp("2026-08-17"), "wb_all__cadence_spm__avg"] == pytest.approx(340.0 / 3)

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
        assert aggregated.groupby("time_bin").size().to_dict() == {"hour": 48, "day": 2}

        # an hour without walking is a real zero rather than a hole in the table
        empty_hour = ("hour", pd.Timestamp("2026-08-17 05:00:00"))
        assert aggregated.loc[empty_hour, "wb_all__count"] == 0
        assert aggregated.loc[empty_hour, "total_walking_duration_min"] == 0
        assert pd.isna(aggregated.loc[empty_hour, "wb_all__cadence_spm__avg"])

    def test_a_recording_without_walking_bouts_still_reports_its_bins(self):
        result = MultiGranularAggregator().aggregate(_wb_dmos([], []), timeline=_timeline())

        aggregated = result.aggregated_data_
        assert aggregated.groupby("time_bin").size().to_dict() == {"hour": 48, "day": 2}
        assert (aggregated["wb_all__count"] == 0).all()
        assert aggregated["wb_all__cadence_spm__avg"].isna().all()

    def test_day_start_hour_moves_the_reported_days(self):
        # a bout half an hour after midnight belongs to the previous day when the day starts at 04:00
        wb_dmos = _wb_dmos([0.5, 24.5], [100.0, 120.0])

        result = MultiGranularAggregator(time_bins=("day",), day_start_hour=4).aggregate(wb_dmos, timeline=_timeline())

        days = result.aggregated_data_.loc["day"]
        assert list(days.index) == [
            pd.Timestamp("2026-08-16 04:00:00"),
            pd.Timestamp("2026-08-17 04:00:00"),
            pd.Timestamp("2026-08-18 04:00:00"),
        ]
        assert days["wb_all__count"].to_list() == [1, 1, 0]


class TestCoverage:
    def test_min_coverage_keeps_only_sufficiently_covered_bins(self):
        # the recording starts at 12:00 and runs for 24 hours, so no day is fully covered
        timeline = RecordingTimeline.from_uniform(
            pd.Timestamp("2026-08-17 12:00:00").timestamp(), int(24 * 3600 * _SAMPLING_RATE_HZ), _SAMPLING_RATE_HZ
        )
        wb_dmos = _wb_dmos([0.5, 13.0], [100.0, 110.0])

        strict = MultiGranularAggregator(min_coverage=1.0).aggregate(wb_dmos, timeline=timeline)
        lenient = MultiGranularAggregator(min_coverage=0.0).aggregate(wb_dmos, timeline=timeline)

        assert strict.aggregated_data_.groupby("time_bin").size().to_dict() == {"hour": 24}
        assert lenient.aggregated_data_.groupby("time_bin").size().to_dict() == {"hour": 24, "day": 2}
        assert lenient.aggregated_data_.loc["day", "coverage"].to_list() == [0.5, 0.5]

    def test_a_dropped_hour_still_counts_towards_its_day(self):
        # half of hour 3 is lost, and a walking bout sits in the half that survived
        timeline = _gapped_timeline(hours=48.0, gap_h=(3.5, 4.0))
        wb_dmos = _wb_dmos([1.0, 3.2], [100.0, 120.0], timeline=timeline)

        result = MultiGranularAggregator(min_coverage=0.9).aggregate(wb_dmos, timeline=timeline)

        aggregated = result.aggregated_data_
        assert pd.Timestamp("2026-08-17 03:00:00") not in aggregated.loc["hour"].index
        assert aggregated.loc[("day", pd.Timestamp("2026-08-17")), "wb_all__count"] == 2
        assert aggregated.loc[("day", pd.Timestamp("2026-08-17")), "coverage"] == pytest.approx(1 - 0.5 / 24)


class TestNestedAggregator:
    def test_mask_removes_implausible_walking_bouts(self):
        wb_dmos = _wb_dmos([0.1, 0.2], [100.0, 110.0])
        mask = pd.DataFrame(True, index=wb_dmos.index, columns=wb_dmos.columns)
        mask.loc[1, "cadence_spm"] = False

        result = MultiGranularAggregator(time_bins=("hour",)).aggregate(
            wb_dmos, timeline=_timeline(), wb_dmos_mask=mask
        )

        first_hour = result.aggregated_data_.loc[("hour", pd.Timestamp("2026-08-17 00:00:00"))]
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

        day = result.aggregated_data_.loc[("day", pd.Timestamp("2026-08-17"))]
        assert day["wb_all_sum"] == 3
        assert day["cadence_all_avg"] == pytest.approx((105.0 + 130.0) / 2)

    def test_every_aggregated_column_is_classified(self):
        # a column that is neither a known total nor a known average would be averaged without a warning
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
    # without the check a misspelled weighting falls through to pooled and returns wrong averages
    with pytest.raises(ValueError, match="Unknown weighting"):
        MultiGranularAggregator(weighting="mean").aggregate(_wb_dmos([0.1], [100.0]), timeline=_timeline())


def test_day_start_hour_outside_the_clock_is_rejected():
    # without the check the day bins shift by a whole day and every daily value is wrong
    with pytest.raises(ValueError, match="day_start_hour"):
        MultiGranularAggregator(day_start_hour=24).aggregate(_wb_dmos([0.1], [100.0]), timeline=_timeline())


def test_fractional_day_start_hour_is_rejected():
    # a float boundary breaks the "hourly bins nest exactly 24 per day" invariant equal weighting relies on
    with pytest.raises(ValueError, match="day_start_hour"):
        MultiGranularAggregator(day_start_hour=4.5).aggregate(_wb_dmos([0.1], [100.0]), timeline=_timeline())


class TestOffGridWalkingBouts:
    """A bin label the grid does not carry used to vanish, walking bouts included, in ``_to_grid``."""

    def test_a_timeline_shorter_than_the_data_is_rejected(self):
        # from_uniform takes n_samples on trust and timestamps() computes times past it, so a
        # mismatched timeline parked half the bouts outside the grid without an error
        timeline = _timeline(hours=2.0)
        wb_dmos = _wb_dmos([0.5, 1.5, 2.5, 3.5], [100.0, 110.0, 120.0, 130.0])

        with pytest.raises(ValueError, match="lie outside the recording timeline"):
            MultiGranularAggregator().aggregate(wb_dmos, timeline=timeline)

    def test_a_recording_ending_on_a_dst_fallback_keeps_every_bout(self):
        # the last sample sits at 01:59:59 local while the recording end converts to 01:00 local,
        # so the 01:00 bin holding the final three bouts used to be missing from the grid
        start = pd.Timestamp("2023-10-28 23:00:00", tz="UTC").timestamp()
        n_samples = 2 * 3600
        timeline = RecordingTimeline.from_sample_times(
            start + np.arange(n_samples), sampling_rate_hz=1.0, timezone="Europe/London"
        )
        starts = np.arange(0, n_samples, 1200)
        wb_dmos = pd.DataFrame(
            {
                "start": starts,
                "duration_s": np.full(len(starts), 20.0),
                "cadence_spm": np.full(len(starts), 100.0),
            },
            index=pd.Index(range(len(starts)), name="wb_id"),
        )

        for weighting in ("equal", "pooled"):
            aggregated = (
                MultiGranularAggregator(weighting=weighting).aggregate(wb_dmos, timeline=timeline).aggregated_data_
            )

            assert aggregated.loc["hour", "wb_all__count"].sum() == len(wb_dmos)
            assert aggregated.loc["day", "wb_all__count"].sum() == len(wb_dmos)


def test_empty_time_bins_is_rejected():
    # without the check this reaches TIME_BIN_ORDER.index(requested[-1]) with an empty requested tuple
    with pytest.raises(ValueError, match="time_bins"):
        MultiGranularAggregator(time_bins=()).aggregate(_wb_dmos([0.1], [100.0]), timeline=_timeline())
