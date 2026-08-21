"""Tests for the UoS-MobGap wall-clock binning of walking bouts."""

import numpy as np
import pandas as pd
import pytest

from mobgap.aggregation.uos import (
    RecordingTimeline,
    add_time_bins,
    bin_coverage,
    time_bin_grid,
)
from mobgap.data._dataset_from_data import GaitDatasetFromData

_START = pd.Timestamp("2026-08-17 14:30:00").timestamp()


def _timeline(hours: float = 24.0, sampling_rate_hz: float = 10.0) -> RecordingTimeline:
    return RecordingTimeline.from_uniform(_START, int(hours * 3600 * sampling_rate_hz), sampling_rate_hz)


def _dataset(index: pd.Index, recording_metadata: dict) -> GaitDatasetFromData:
    data = pd.DataFrame(
        np.zeros((len(index), 6)),
        columns=["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"],
        index=index,
    )
    return GaitDatasetFromData(
        {"rec": {"LowerBack": data}},
        10.0,
        _participant_metadata={"rec": {"height_m": 1.7, "sensor_height_m": 1.0, "cohort": "HA"}},
        _recording_metadata={"rec": recording_metadata},
        single_sensor_name="LowerBack",
        index_cols="recording_id",
    )


class TestRecordingTimeline:
    def test_uniform_maps_samples_and_covers_every_sampling_interval(self):
        timeline = _timeline(hours=2)

        assert timeline.start == pd.Timestamp("2026-08-17 14:30:00")
        # the recording covers one sampling interval per sample, so a full two hours
        assert timeline.end == pd.Timestamp("2026-08-17 16:30:00")
        assert list(timeline.timestamps(np.array([0, 18000]))) == [
            pd.Timestamp("2026-08-17 14:30:00"),
            pd.Timestamp("2026-08-17 15:00:00"),
        ]

    def test_sample_times_stay_exact_across_gaps(self):
        # one hour of samples is missing in the middle, as after dropping invalid samples
        sample_times = np.concatenate([_START + np.arange(10), _START + 3600 + np.arange(10)])
        timeline = RecordingTimeline.from_sample_times(sample_times)

        # the uniform mapping would place sample 10 one second after the start
        assert timeline.timestamps(np.array([10]))[0] == pd.Timestamp("2026-08-17 15:30:00")
        assert timeline.end == pd.Timestamp("2026-08-17 15:30:10")
        assert timeline.sampling_rate_hz == pytest.approx(1.0)

    def test_sample_times_infers_rate_from_the_median_interval_despite_a_leading_gap(self):
        # an isolated first sample followed by a dropout would make the first interval
        # sample_times[1] - sample_times[0] look like the whole gap, not the true 100 Hz sampling rate
        rest = _START + 500.0 + np.arange(3 * 60 * 100) / 100.0
        sample_times = np.concatenate([[_START], rest])

        timeline = RecordingTimeline.from_sample_times(sample_times)

        assert timeline.sampling_rate_hz == pytest.approx(100.0, rel=1e-3)

    def test_sample_times_prefers_the_explicit_sampling_rate_over_inference(self):
        rest = _START + 500.0 + np.arange(3 * 60 * 100) / 100.0
        sample_times = np.concatenate([[_START], rest])

        timeline = RecordingTimeline.from_sample_times(sample_times, sampling_rate_hz=100.0)

        assert timeline.sampling_rate_hz == 100.0

    def test_from_datapoint_uses_the_dataset_sampling_rate_not_inference(self):
        # the leading gap in the index would make a two-sample rate estimate wildly wrong
        rest = _START + 500.0 + np.arange(3 * 60 * 100) / 100.0
        sample_times = np.concatenate([[_START], rest])
        dataset = _dataset(pd.Index(sample_times, name="time"), {})

        timeline = RecordingTimeline.from_datapoint(dataset[0])

        assert timeline.sampling_rate_hz == pytest.approx(10.0, rel=1e-3)

    def test_timezone_converts_utc_to_local_wall_clock(self):
        # 2026-03-29 00:59 UTC is one minute before the British clocks jump to summer time
        utc_start = pd.Timestamp("2026-03-29 00:59:00", tz="UTC").timestamp()
        timeline = RecordingTimeline.from_uniform(utc_start, 7200, 1.0, timezone="Europe/London")

        assert timeline.start == pd.Timestamp("2026-03-29 00:59:00")
        # the local clock jumps from 01:00 to 02:00, so two hours of samples end at 03:59
        assert timeline.end == pd.Timestamp("2026-03-29 03:59:00")

    def test_from_datapoint_reads_the_time_index(self):
        # a time-indexed dataset carries the true sample times, gaps included
        sample_times = np.concatenate([_START + np.arange(10), _START + 3600 + np.arange(10)])
        dataset = _dataset(pd.Index(sample_times, name="time"), {"cwa_start_time": _START + 999})

        timeline = RecordingTimeline.from_datapoint(dataset[0])

        # the metadata start time is deliberately wrong, so only the index can give this answer
        assert timeline.start == pd.Timestamp("2026-08-17 14:30:00")
        assert timeline.timestamps(np.array([10]))[0] == pd.Timestamp("2026-08-17 15:30:00")

    def test_from_datapoint_without_a_time_index(self):
        dataset = _dataset(pd.RangeIndex(100, name="samples"), {"cwa_start_time": _START})

        with pytest.raises(ValueError, match="include_time_index=True"):
            RecordingTimeline.from_datapoint(dataset[0])


def test_a_float_time_index_does_not_change_what_the_pipeline_computes():
    """``from_datapoint`` is only safe if this holds. Until this test, only the README said it did.

    ``include_time_index=True`` replaces the sample-number index with float Unix seconds. If any
    step of the pipeline read the index as a value rather than slicing it positionally, the DMOs
    would change and the walking bout ``start`` column would stop being a sample number, which is
    what :meth:`RecordingTimeline.timestamps` looks up.
    """
    from mobgap.data import LabExampleDataset  # noqa: PLC0415
    from mobgap.pipeline import MobilisedPipelineHealthy  # noqa: PLC0415

    reference = LabExampleDataset().get_subset(cohort="HA", participant_id="001", test="Test11", trial="Trial1")[0]
    data = reference.data_ss

    def run(index: pd.Index) -> pd.DataFrame:
        renamed = data.set_axis(index)
        datapoint = GaitDatasetFromData(
            {"rec": {"LowerBack": renamed}},
            reference.sampling_rate_hz,
            _participant_metadata={"rec": reference.participant_metadata},
            _recording_metadata={"rec": {"measurement_condition": "laboratory"}},
            single_sensor_name="LowerBack",
            index_cols="recording_id",
        )[0]
        return MobilisedPipelineHealthy().safe_run(datapoint).per_wb_parameters_

    by_sample = run(pd.RangeIndex(len(data), name="samples"))
    by_time = run(pd.Index(_START + np.arange(len(data)) / reference.sampling_rate_hz, name="time"))

    assert len(by_sample) > 0
    # rule_obj holds algorithm instances that compare by identity, so only the numbers can be compared
    pd.testing.assert_frame_equal(by_sample.select_dtypes("number"), by_time.select_dtypes("number"))
    # start stays a sample number rather than becoming the epoch value of the index
    assert by_time["start"].max() < len(data)


class TestAddTimeBins:
    def test_bins_are_floored_to_the_local_wall_clock(self):
        timeline = _timeline(hours=24)
        # 14:30:00 (start), 23:59:59, and 00:00:01 of the next day
        wb_dmos = pd.DataFrame({"start": [0, 341990, 342010]}, index=pd.Index([0, 1, 2], name="wb_id"))

        binned = add_time_bins(wb_dmos, timeline)

        assert list(binned["start_time"]) == [
            pd.Timestamp("2026-08-17 14:30:00"),
            pd.Timestamp("2026-08-17 23:59:59"),
            pd.Timestamp("2026-08-18 00:00:01"),
        ]
        assert list(binned["hour_start"]) == [
            pd.Timestamp("2026-08-17 14:00:00"),
            pd.Timestamp("2026-08-17 23:00:00"),
            pd.Timestamp("2026-08-18 00:00:00"),
        ]
        assert list(binned["day_start"]) == [
            pd.Timestamp("2026-08-17"),
            pd.Timestamp("2026-08-17"),
            pd.Timestamp("2026-08-18"),
        ]

    def test_day_start_hour_moves_the_day_boundary(self):
        timeline = _timeline(hours=24)
        # 23:59:59 and 00:00:01 both fall before 04:00, so both belong to the day starting 2026-08-17 04:00
        wb_dmos = pd.DataFrame({"start": [0, 341990, 342010]}, index=pd.Index([0, 1, 2], name="wb_id"))

        binned = add_time_bins(wb_dmos, timeline, day_start_hour=4)

        assert list(binned["day_start"]) == [pd.Timestamp("2026-08-17 04:00:00")] * 3


class TestTimeBinGrid:
    def test_grid_covers_every_touched_bin(self):
        timeline = _timeline(hours=24)

        assert list(time_bin_grid(timeline, "day")) == [pd.Timestamp("2026-08-17"), pd.Timestamp("2026-08-18")]
        assert len(time_bin_grid(timeline, "hour")) == 25

    def test_grid_follows_the_day_start_hour(self):
        # the recording runs 14:30 to 14:30, so a day starting at 16:00 is only ever touched twice
        timeline = _timeline(hours=24)

        grid = time_bin_grid(timeline, "day", day_start_hour=16)

        assert list(grid) == [pd.Timestamp("2026-08-16 16:00:00"), pd.Timestamp("2026-08-17 16:00:00")]

    def test_grid_excludes_a_bin_starting_at_the_recording_end(self):
        # the recording ends exactly at midnight, so the following day holds no samples
        timeline = RecordingTimeline.from_uniform(pd.Timestamp("2026-08-17 22:00:00").timestamp(), 7200, 1.0)

        assert list(time_bin_grid(timeline, "day")) == [pd.Timestamp("2026-08-17")]

    def test_grid_keeps_the_repeated_hour_when_the_recording_ends_on_a_dst_fallback(self):
        # 23:00 UTC is 00:00 BST, and the clocks fall back exactly two hours later. The last sample
        # is at 01:59:59 local, but end_epoch_s converts to 01:00 local, so an end-of-recording
        # comparison on local labels with no timezone would cut the 01:00 bin that holds
        # the final hour of data.
        start = pd.Timestamp("2023-10-28 23:00:00", tz="UTC").timestamp()
        timeline = RecordingTimeline.from_sample_times(
            start + np.arange(2 * 3600), sampling_rate_hz=1.0, timezone="Europe/London"
        )

        assert timeline.end == pd.Timestamp("2023-10-29 01:00:00")
        assert list(time_bin_grid(timeline, "hour")) == [
            pd.Timestamp("2023-10-29 00:00:00"),
            pd.Timestamp("2023-10-29 01:00:00"),
        ]

    def test_grid_holds_the_spring_forward_hour_that_never_happened(self):
        # 2023-03-26: British clocks jump 01:00 -> 02:00, so the 01:00 bin exists on the local grid
        # and stays empty. The README documents it as visible rather than missing.
        start = pd.Timestamp("2023-03-26 00:00:00", tz="UTC").timestamp()
        timeline = RecordingTimeline.from_sample_times(
            start + np.arange(4 * 3600), sampling_rate_hz=1.0, timezone="Europe/London"
        )

        grid = time_bin_grid(timeline, "hour")

        assert pd.Timestamp("2023-03-26 01:00:00") in grid
        assert bin_coverage(grid, timeline, "hour")[grid == pd.Timestamp("2023-03-26 01:00:00")] == 0.0


class TestBinCoverage:
    def test_uniform_timeline_only_loses_the_end_bins(self):
        timeline = _timeline(hours=24)
        grid = time_bin_grid(timeline, "hour")

        coverage = bin_coverage(grid, timeline, "hour")

        # the recording starts and ends at half past the hour
        assert coverage[0] == pytest.approx(0.5)
        assert coverage[-1] == pytest.approx(0.5)
        assert coverage[1:-1] == pytest.approx(1.0)

    def test_sample_times_expose_a_gap_in_the_middle(self):
        sampling_rate_hz = 10.0
        start = pd.Timestamp("2026-08-17 00:00:00").timestamp()
        full = start + np.arange(4 * 3600 * sampling_rate_hz) / sampling_rate_hz
        # the second half of hour 1 is lost, as omconvert marks lost sectors invalid
        lost = (full >= start + 5400) & (full < start + 7200)
        timeline = RecordingTimeline.from_sample_times(full[~lost])

        coverage = bin_coverage(time_bin_grid(timeline, "hour"), timeline, "hour")

        assert coverage[0] == pytest.approx(1.0)
        assert coverage[1] == pytest.approx(0.5)
        assert coverage[2] == pytest.approx(1.0)

    def test_gap_is_found_in_the_local_bin_of_a_converted_recording(self):
        # a UTC clock in British summer time, so the local wall clock runs one hour ahead
        start = pd.Timestamp("2026-08-17 09:00:00", tz="UTC").timestamp()
        full = start + np.arange(4 * 3600)
        lost = (full >= start + 2 * 3600) & (full < start + 2.5 * 3600)
        timeline = RecordingTimeline.from_sample_times(full[~lost], timezone="Europe/London")

        grid = time_bin_grid(timeline, "hour")
        coverage = bin_coverage(grid, timeline, "hour")

        assert timeline.start == pd.Timestamp("2026-08-17 10:00:00")
        assert coverage[grid == pd.Timestamp("2026-08-17 12:00:00")] == pytest.approx(0.5)
        assert coverage[grid == pd.Timestamp("2026-08-17 11:00:00")] == pytest.approx(1.0)

    def test_dst_fallback_hour_is_clipped_to_1_not_2(self):
        # 2023-10-29: British clocks fall back at 02:00 BST -> 01:00 GMT, so local 01:00-01:59
        # occurs twice. A gapless recording puts two hours of samples under that one label.
        start = pd.Timestamp("2023-10-29 00:00:00", tz="UTC").timestamp()
        full = start + np.arange(3 * 3600)
        timeline = RecordingTimeline.from_sample_times(full, sampling_rate_hz=1.0, timezone="Europe/London")

        grid = time_bin_grid(timeline, "hour")
        coverage = bin_coverage(grid, timeline, "hour")

        assert coverage[grid == pd.Timestamp("2023-10-29 01:00:00")] == pytest.approx(1.0)
        assert coverage.max() <= 1.0

    def test_gap_reduces_the_coverage_of_the_whole_day(self):
        start = pd.Timestamp("2026-08-17 00:00:00").timestamp()
        full = start + np.arange(24 * 3600)
        # a six hour dropout in the middle of the day
        lost = (full >= start + 6 * 3600) & (full < start + 12 * 3600)
        timeline = RecordingTimeline.from_sample_times(full[~lost])

        coverage = bin_coverage(time_bin_grid(timeline, "day"), timeline, "day")

        assert coverage == pytest.approx([0.75])
