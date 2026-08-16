"""Tests for the UoS-MobGap wall-clock binning of walking bouts."""

import numpy as np
import pandas as pd
import pytest

from mobgap.aggregation.uos import (
    RecordingTimeline,
    add_time_bins,
    is_complete_bin,
    time_bin_grid,
)
from mobgap.data._dataset_from_data import GaitDatasetFromData

# 2024-03-06 is a Wednesday, so the ISO week starts on 2024-03-04
_START = pd.Timestamp("2024-03-06 14:30:00").timestamp()


def _timeline(hours: float = 24.0, sampling_rate_hz: float = 10.0) -> RecordingTimeline:
    return RecordingTimeline.from_uniform(_START, int(hours * 3600 * sampling_rate_hz), sampling_rate_hz)


class TestRecordingTimeline:
    def test_uniform_maps_samples_and_covers_every_sampling_interval(self):
        timeline = _timeline(hours=2)

        assert timeline.start == pd.Timestamp("2024-03-06 14:30:00")
        # the recording covers one sampling interval per sample, so a full two hours
        assert timeline.end == pd.Timestamp("2024-03-06 16:30:00")
        assert list(timeline.timestamps(np.array([0, 18000]))) == [
            pd.Timestamp("2024-03-06 14:30:00"),
            pd.Timestamp("2024-03-06 15:00:00"),
        ]

    def test_sample_times_stay_exact_across_gaps(self):
        # one hour of samples is missing in the middle, as after dropping invalid samples
        sample_times = np.concatenate([_START + np.arange(10), _START + 3600 + np.arange(10)])
        timeline = RecordingTimeline.from_sample_times(sample_times)

        # the uniform mapping would place sample 10 one second after the start
        assert timeline.timestamps(np.array([10]))[0] == pd.Timestamp("2024-03-06 15:30:00")
        assert timeline.end == pd.Timestamp("2024-03-06 15:30:10")

    def test_timezone_converts_utc_to_local_wall_clock(self):
        # 2024-03-31 00:59 UTC is one minute before the British clocks jump to summer time
        utc_start = pd.Timestamp("2024-03-31 00:59:00", tz="UTC").timestamp()
        timeline = RecordingTimeline.from_uniform(utc_start, 7200, 1.0, timezone="Europe/London")

        assert timeline.start == pd.Timestamp("2024-03-31 00:59:00")
        # the local clock jumps from 01:00 to 02:00, so two hours of samples end at 03:59
        assert timeline.end == pd.Timestamp("2024-03-31 03:59:00")

    def test_from_datapoint_reads_the_recording_metadata(self):
        data = pd.DataFrame(np.zeros((100, 6)), columns=["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"])
        dataset = GaitDatasetFromData(
            {"rec": {"LowerBack": data}},
            10.0,
            _participant_metadata={"rec": {"height_m": 1.7, "sensor_height_m": 1.0, "cohort": "HA"}},
            _recording_metadata={"rec": {"cwa_start_time": _START, "timezone": None}},
            single_sensor_name="LowerBack",
            index_cols="recording_id",
        )

        timeline = RecordingTimeline.from_datapoint(dataset[0])

        assert timeline.start == pd.Timestamp("2024-03-06 14:30:00")
        assert timeline.end == pd.Timestamp("2024-03-06 14:30:10")


class TestAddTimeBins:
    def test_bins_are_floored_to_the_local_wall_clock(self):
        timeline = _timeline(hours=24)
        # 14:30:00 (start), 23:59:59, and 00:00:01 of the next day
        wb_dmos = pd.DataFrame({"start": [0, 341990, 342010]}, index=pd.Index([0, 1, 2], name="wb_id"))

        binned = add_time_bins(wb_dmos, timeline)

        assert list(binned["start_time"]) == [
            pd.Timestamp("2024-03-06 14:30:00"),
            pd.Timestamp("2024-03-06 23:59:59"),
            pd.Timestamp("2024-03-07 00:00:01"),
        ]
        assert list(binned["hour_start"]) == [
            pd.Timestamp("2024-03-06 14:00:00"),
            pd.Timestamp("2024-03-06 23:00:00"),
            pd.Timestamp("2024-03-07 00:00:00"),
        ]
        assert list(binned["day_start"]) == [
            pd.Timestamp("2024-03-06"),
            pd.Timestamp("2024-03-06"),
            pd.Timestamp("2024-03-07"),
        ]
        # both days are in the week starting Monday 2024-03-04
        assert list(binned["week_start"]) == [pd.Timestamp("2024-03-04")] * 3


class TestTimeBinGrid:
    def test_grid_covers_every_touched_bin(self):
        timeline = _timeline(hours=24)

        assert list(time_bin_grid(timeline, "day")) == [pd.Timestamp("2024-03-06"), pd.Timestamp("2024-03-07")]
        assert list(time_bin_grid(timeline, "week")) == [pd.Timestamp("2024-03-04")]
        assert len(time_bin_grid(timeline, "hour")) == 25

    def test_grid_excludes_a_bin_starting_at_the_recording_end(self):
        # the recording ends exactly at midnight, so the following day holds no samples
        timeline = RecordingTimeline.from_uniform(pd.Timestamp("2024-03-06 22:00:00").timestamp(), 7200, 1.0)

        assert list(time_bin_grid(timeline, "day")) == [pd.Timestamp("2024-03-06")]

    def test_completeness_needs_full_coverage(self):
        timeline = _timeline(hours=24)
        grid = time_bin_grid(timeline, "day")

        # the recording starts at 14:30 and ends at 14:30, so neither day is complete
        assert not is_complete_bin(grid, timeline, "day").any()
        assert is_complete_bin(time_bin_grid(timeline, "hour"), timeline, "hour").sum() == 23
