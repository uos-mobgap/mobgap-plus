"""Map walking bouts of a single recording onto wall-clock time bins.

UoS-MobGap extension.

Walking bouts carry sample indices. This module turns those sample indices into
local wall-clock timestamps and floors them to hour and day bins.
All timestamps produced here are timezone-naive local wall-clock,
so a day is always exactly 24 hours long.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mobgap.data.base import BaseGaitDataset

TimeBin = Literal["hour", "day"]

# columns added to the walking bout table for every time bin
TIME_BIN_COLUMNS: Final = MappingProxyType({"hour": "hour_start", "day": "day_start"})

# wall-clock width of every time bin
TIME_BIN_WIDTHS: Final = MappingProxyType({"hour": pd.Timedelta(hours=1), "day": pd.Timedelta(days=1)})

_TIME_BIN_FREQ: Final = MappingProxyType({"hour": "h", "day": "D"})

_EPOCH: Final = pd.Timestamp("1970-01-01")


def _to_local_time(epoch_s: np.ndarray, timezone: str | None) -> pd.DatetimeIndex:
    """Convert Unix epoch seconds to timezone-naive local wall-clock timestamps."""
    times = pd.to_datetime(epoch_s, unit="s")
    if timezone is None:
        return times

    return times.tz_localize("UTC").tz_convert(timezone).tz_localize(None)


def _to_epoch_s(times: pd.DatetimeIndex, timezone: str | None) -> np.ndarray:
    """Convert timezone-naive local wall-clock timestamps back to Unix epoch seconds."""
    if timezone is None:
        return ((times - _EPOCH) / pd.Timedelta(1, "s")).to_numpy()

    # a local time that daylight saving skipped moves forward, a repeated one takes its first pass
    utc = times.tz_localize(timezone, ambiguous=True, nonexistent="shift_forward").tz_convert("UTC")
    return ((utc.tz_localize(None) - _EPOCH) / pd.Timedelta(1, "s")).to_numpy()


def _floor_to_time_bin(times: pd.DatetimeIndex, time_bin: TimeBin, day_start_hour: int) -> pd.DatetimeIndex:
    """Floor local timestamps to the start of their hour or day."""
    if time_bin == "hour":
        return times.floor("h")

    # shift the day boundary onto midnight, floor there, and shift back
    offset = pd.Timedelta(hours=day_start_hour)
    return (times - offset).normalize() + offset


@dataclass(frozen=True)
class RecordingTimeline:
    """Wall-clock timeline of a single recording.

    The timeline maps sample indices of a recording to local wall-clock
    timestamps and knows which stretches of wall-clock time the recording holds
    samples for. Use one of the constructors rather than instantiating directly.

    Recordings are stored as Unix epoch seconds, which carry no timezone. The
    ``timezone`` decides how those seconds are read:

    - ``None`` (default): the clock already runs in local time. This is the
      case for OpenMovement AX3/AX6 loggers, which are configured with the
      local time of the study site and never store an offset.
    - An IANA name such as ``"Europe/London"``: the clock runs in UTC and is
      converted to local wall clock, including daylight saving time.

    Parameters
    ----------
    start_epoch_s
        Unix timestamp in seconds of the first sample.
    end_epoch_s
        Unix timestamp in seconds one sampling interval after the last sample.
        Together with ``start_epoch_s`` this is the half-open interval the
        recording spans.
    sampling_rate_hz
        Sampling rate of the recording in Hz.
    timezone
        IANA timezone name of the study site, or ``None`` when the recording
        clock is already local.
    sample_times
        Unix timestamp in seconds of every sample. Set by
        :meth:`from_sample_times`. Stored by reference. When present, gaps
        inside the recording are visible to :func:`bin_coverage`.
    """

    start_epoch_s: float
    end_epoch_s: float
    sampling_rate_hz: float
    timezone: str | None = None
    sample_times: np.ndarray | None = None

    @property
    def start(self) -> pd.Timestamp:
        """Local wall-clock timestamp of the first sample."""
        return _to_local_time(np.array([self.start_epoch_s]), self.timezone)[0]

    @property
    def end(self) -> pd.Timestamp:
        """Local wall-clock timestamp one sampling interval after the last sample."""
        return _to_local_time(np.array([self.end_epoch_s]), self.timezone)[0]

    @classmethod
    def from_uniform(
        cls,
        start_epoch_s: float,
        n_samples: int,
        sampling_rate_hz: float,
        *,
        timezone: str | None = None,
    ) -> RecordingTimeline:
        """Build a timeline for a gapless recording.

        Sample ``i`` is assumed to be recorded at ``start_epoch_s + i / sampling_rate_hz``.
        A recording that lost samples breaks that assumption and this timeline
        cannot detect it, so use :meth:`from_sample_times` for measured data.

        Parameters
        ----------
        start_epoch_s
            Unix timestamp in seconds of the first sample.
        n_samples
            Number of samples in the recording.
        sampling_rate_hz
            Sampling rate of the recording in Hz.
        timezone
            IANA timezone name, or ``None`` when the recording clock is already local.

        Returns
        -------
        RecordingTimeline
            Timeline spanning ``n_samples / sampling_rate_hz`` seconds without gaps.
        """
        start_epoch_s = float(start_epoch_s)
        return cls(
            start_epoch_s=start_epoch_s,
            end_epoch_s=start_epoch_s + n_samples / sampling_rate_hz,
            sampling_rate_hz=float(sampling_rate_hz),
            timezone=timezone,
        )

    @classmethod
    def from_sample_times(
        cls,
        sample_times: np.ndarray,
        *,
        sampling_rate_hz: float | None = None,
        timezone: str | None = None,
    ) -> RecordingTimeline:
        """Build a timeline from the timestamp of every sample.

        This is exact even when samples are missing from the recording, because
        every walking bout is placed by looking up the time of its start sample,
        and it lets :func:`bin_coverage` see the resulting gaps.

        Parameters
        ----------
        sample_times
            Unix timestamps in seconds, one per sample, in recording order.
        sampling_rate_hz
            The recording's nominal sampling rate. Pass this whenever it is
            known -- if omitted, it is estimated from ``sample_times`` as the
            median of consecutive differences, which is wrong whenever the
            leading samples straddle a dropped stretch (the first two
            *surviving* samples then look like one huge interval).
        timezone
            IANA timezone name, or ``None`` when the recording clock is already local.

        Returns
        -------
        RecordingTimeline
            Timeline covering the samples up to one sampling interval after the last one.
        """
        sample_times = np.asarray(sample_times, dtype=float)
        if sampling_rate_hz is None:
            sampling_interval_s = float(np.median(np.diff(sample_times)))
            sampling_rate_hz = 1.0 / sampling_interval_s
        else:
            sampling_rate_hz = float(sampling_rate_hz)
            sampling_interval_s = 1.0 / sampling_rate_hz

        return cls(
            start_epoch_s=float(sample_times[0]),
            end_epoch_s=float(sample_times[-1]) + sampling_interval_s,
            sampling_rate_hz=sampling_rate_hz,
            timezone=timezone,
            sample_times=sample_times,
        )

    @classmethod
    def from_datapoint(cls, datapoint: BaseGaitDataset, *, timezone: str | None = None) -> RecordingTimeline:
        """Build a timeline from a single-recording dataset.

        The dataset must carry the wall-clock time of every sample as its index,
        which :func:`mobgap.data.uos.load_cwa_as_dataset` writes when called with
        ``include_time_index=True``. Sample indices alone cannot be placed on the
        clock exactly, because a recording that lost samples has fewer of them
        than elapsed time suggests.

        The timezone is taken from ``recording_metadata["timezone"]`` unless it
        is passed explicitly.

        Parameters
        ----------
        datapoint
            Dataset with exactly one recording, loaded with ``include_time_index=True``.
        timezone
            IANA timezone name overriding the recording metadata.

        Returns
        -------
        RecordingTimeline
            Timeline of the recording.
        """
        data = datapoint.data_ss

        # the CWA loader names the Unix-time index "time" and the sample index "samples"
        if data.index.name != "time":
            raise ValueError(
                "The dataset carries no wall-clock time index, so walking bouts cannot be placed on the clock. "
                "Reload it with load_cwa_as_dataset(..., include_time_index=True), or build the timeline with "
                "RecordingTimeline.from_uniform() if the recording is known to be gapless."
            )

        if timezone is None:
            timezone = datapoint.recording_metadata.get("timezone")

        return cls.from_sample_times(
            data.index.to_numpy(),
            sampling_rate_hz=datapoint.sampling_rate_hz,
            timezone=timezone,
        )

    def timestamps(self, samples: np.ndarray | pd.Series) -> pd.DatetimeIndex:
        """Return the local wall-clock timestamps of the given sample indices.

        Parameters
        ----------
        samples
            Sample indices into the recording.

        Returns
        -------
        pandas.DatetimeIndex
            Timezone-naive local timestamps, one per sample index.
        """
        samples = np.asarray(samples)

        if self.sample_times is None:
            epoch_s = self.start_epoch_s + samples / self.sampling_rate_hz
        else:
            epoch_s = self.sample_times[samples]

        return _to_local_time(epoch_s, self.timezone)


def add_time_bins(
    wb_dmos: pd.DataFrame,
    timeline: RecordingTimeline,
    *,
    time_bins: Sequence[TimeBin] = ("hour", "day"),
    day_start_hour: int = 0,
) -> pd.DataFrame:
    """Add wall-clock time and time bin columns to a walking bout table.

    Every walking bout is placed in the bin that contains its start, so a
    walking bout crossing a day boundary counts towards the day it started in.
    Its DMOs describe the bout as a whole and cannot be split across bins.

    The added columns turn the standard aggregator into a per-hour or per-day
    aggregator::

        MobilisedAggregator(groupby=["day_start"]).aggregate(
            add_time_bins(wb_dmos, timeline)
        )

    Parameters
    ----------
    wb_dmos
        DMO data per walking bout, with a ``start`` column holding the sample
        index at which the walking bout starts. This is the
        ``per_wb_parameters_`` output of the Mobilise-D pipelines.
    timeline
        Wall-clock timeline of the recording the walking bouts belong to.
    time_bins
        Time bins to add. Each one adds the column named in :data:`TIME_BIN_COLUMNS`.
    day_start_hour
        Hour of the local clock at which a day starts, from 0 to 23. A day is
        always 24 hours long, this only moves where it begins, for example to 4
        so that walking after midnight counts towards the previous day. Whole
        hours only, so that hourly bins always nest exactly 24 per day.

    Returns
    -------
    pandas.DataFrame
        Copy of ``wb_dmos`` with a ``start_time`` column and one column per
        requested time bin, all timezone-naive local wall clock.
    """
    start_times = timeline.timestamps(wb_dmos["start"].to_numpy())
    new_columns = {"start_time": start_times}

    for time_bin in time_bins:
        new_columns[TIME_BIN_COLUMNS[time_bin]] = _floor_to_time_bin(start_times, time_bin, day_start_hour)

    return wb_dmos.assign(**new_columns)


def time_bin_grid(timeline: RecordingTimeline, time_bin: TimeBin, *, day_start_hour: int = 0) -> pd.DatetimeIndex:
    """Return every time bin the recording reaches into.

    Parameters
    ----------
    timeline
        Wall-clock timeline of the recording.
    time_bin
        Time bin to build the grid for.
    day_start_hour
        Hour of the local clock at which a day starts, see :func:`add_time_bins`.

    Returns
    -------
    pandas.DatetimeIndex
        Start of every bin the recording overlaps, including bins without any
        walking bout.
    """
    edges = _floor_to_time_bin(pd.DatetimeIndex([timeline.start, timeline.end]), time_bin, day_start_hour)
    grid = pd.date_range(edges[0], edges[1], freq=_TIME_BIN_FREQ[time_bin], name="bin_start")

    # a bin starting exactly when the recording ends contains no samples at all
    return grid[grid < timeline.end]


def bin_coverage(grid: pd.DatetimeIndex, timeline: RecordingTimeline, time_bin: TimeBin) -> np.ndarray:
    """Return the fraction of every bin that the recording holds samples for.

    A bin fully inside the recording has a coverage of 1. The truncated bins at
    the ends of the recording, and any bin overlapping a stretch of lost
    samples, have less.

    Parameters
    ----------
    grid
        Start of every bin, as returned by :func:`time_bin_grid`.
    timeline
        Wall-clock timeline of the recording. A timeline built by
        :meth:`RecordingTimeline.from_uniform` has no record of lost samples, so
        only the ends of the recording can reduce the coverage.
    time_bin
        Time bin the grid was built for.

    Returns
    -------
    numpy.ndarray
        Coverage of every bin, between 0 and 1. A DST fall-back bin holds two
        wall-clock hours of samples merged under one naive local-time label;
        its coverage is still clipped to 1 rather than reporting ~2.
    """
    width = TIME_BIN_WIDTHS[time_bin]
    edges = grid.append(pd.DatetimeIndex([grid[-1] + width]))

    if timeline.sample_times is None:
        # without measured sample times only the recording ends can cut a bin short
        starts = np.maximum(edges[:-1].to_numpy(), np.datetime64(timeline.start))
        ends = np.minimum(edges[1:].to_numpy(), np.datetime64(timeline.end))
        return np.clip((ends - starts) / np.timedelta64(1, "s") / width.total_seconds(), 0.0, 1.0)

    # counting the samples that fall in each bin catches gaps anywhere in the recording
    samples_per_bin = np.diff(np.searchsorted(timeline.sample_times, _to_epoch_s(edges, timeline.timezone)))
    coverage = samples_per_bin / (width.total_seconds() * timeline.sampling_rate_hz)
    return np.clip(coverage, 0.0, 1.0)
