"""Map walking bouts of a single recording onto wall-clock time bins.

UoS-MobGap extension.

Walking bouts carry sample indices. This module turns those sample indices into 
local wall-clock timestamps and floors them to hour, day, and week bins. 
All timestamps produced here are timezone-naive local wall-clock, 
so a day is always the calendar day from 00:00 to 00:00.
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

TimeBin = Literal["hour", "day", "week"]

# columns added to the walking bout table for every time bin
TIME_BIN_COLUMNS: Final = MappingProxyType({"hour": "hour_start", "day": "day_start", "week": "week_start"})

# wall-clock width of every time bin
TIME_BIN_WIDTHS: Final = MappingProxyType(
    {"hour": pd.Timedelta(hours=1), "day": pd.Timedelta(days=1), "week": pd.Timedelta(days=7)}
)

_TIME_BIN_FREQ: Final = MappingProxyType({"hour": "h", "day": "D", "week": "7D"})


def _to_local_time(epoch_s: np.ndarray, timezone: str | None) -> pd.DatetimeIndex:
    """Convert Unix epoch seconds to timezone-naive local wall-clock timestamps."""
    times = pd.to_datetime(epoch_s, unit="s")
    if timezone is None:
        return times

    return times.tz_localize("UTC").tz_convert(timezone).tz_localize(None)


def _floor_to_time_bin(times: pd.DatetimeIndex, time_bin: TimeBin) -> pd.DatetimeIndex:
    """Floor local timestamps to the start of their hour, day, or week."""
    if time_bin == "hour":
        return times.floor("h")

    days = times.normalize()
    if time_bin == "day":
        return days

    if time_bin == "week":
        # ISO weeks: Monday 00:00 is the first moment of the week
        return days - pd.to_timedelta(times.dayofweek, unit="D")


@dataclass(frozen=True)
class RecordingTimeline:
    """Wall-clock timeline of a single recording.

    The timeline maps sample indices of a recording to local wall-clock
    timestamps and describes which stretch of wall-clock time the recording
    covers. Use one of the constructors rather than instantiating directly.

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
        recording covers.
    timezone
        IANA timezone name of the study site, or ``None`` when the recording
        clock is already local.
    sampling_rate_hz
        Sampling rate used to map sample indices to time. Set by
        :meth:`from_uniform`.
    sample_times
        Unix timestamp in seconds of every sample. Set by
        :meth:`from_sample_times`. Stored by reference.
    """

    start_epoch_s: float
    end_epoch_s: float
    timezone: str | None = None
    sampling_rate_hz: float | None = None
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
        Use :meth:`from_sample_times` instead when samples were dropped from the
        recording, for example by ``drop_invalid`` of
        :func:`mobgap.data.uos.load_cwa_as_dataset`.

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
            Timeline covering ``n_samples / sampling_rate_hz`` seconds.
        """
        start_epoch_s = float(start_epoch_s)
        return cls(
            start_epoch_s=start_epoch_s,
            end_epoch_s=start_epoch_s + n_samples / sampling_rate_hz,
            timezone=timezone,
            sampling_rate_hz=float(sampling_rate_hz),
        )

    @classmethod
    def from_sample_times(cls, sample_times: np.ndarray, *, timezone: str | None = None) -> RecordingTimeline:
        """Build a timeline from the timestamp of every sample.

        This is exact even when samples are missing from the recording, because
        every walking bout is placed by looking up the time of its start sample.

        Parameters
        ----------
        sample_times
            Unix timestamps in seconds, one per sample, in recording order.
        timezone
            IANA timezone name, or ``None`` when the recording clock is already local.

        Returns
        -------
        RecordingTimeline
            Timeline covering the samples up to one sampling interval after the last one.
        """
        sample_times = np.asarray(sample_times, dtype=float)
        sampling_interval_s = sample_times[1] - sample_times[0]
        return cls(
            start_epoch_s=float(sample_times[0]),
            end_epoch_s=float(sample_times[-1]) + sampling_interval_s,
            timezone=timezone,
            sample_times=sample_times,
        )

    @classmethod
    def from_datapoint(cls, datapoint: BaseGaitDataset, *, timezone: str | None = None) -> RecordingTimeline:
        """Build a timeline from a single-recording dataset.

        The start time is taken from ``recording_metadata["recording_start_time"]``,
        falling back to ``recording_metadata["cwa_start_time"]`` written by
        :func:`mobgap.data.uos.load_cwa_as_dataset`. The timezone is taken from
        ``recording_metadata["timezone"]`` unless it is passed explicitly.

        Datasets loaded with a time index (``include_time_index=True``) use the
        exact sample times, everything else assumes a gapless recording.

        Parameters
        ----------
        datapoint
            Dataset with exactly one recording.
        timezone
            IANA timezone name overriding the recording metadata.

        Returns
        -------
        RecordingTimeline
            Timeline of the recording.
        """
        data = datapoint.data_ss
        recording_metadata = datapoint.recording_metadata
        
        if timezone is None:
            timezone = recording_metadata.get("timezone")

        # the CWA loader names the Unix-time index "time" and the sample index "samples"
        if data.index.name == "time":
            return cls.from_sample_times(data.index.to_numpy(), timezone=timezone)

        start_epoch_s = recording_metadata.get("recording_start_time", recording_metadata.get("cwa_start_time"))
        if start_epoch_s is None:
            raise KeyError(
                "The recording metadata must contain 'recording_start_time' (or 'cwa_start_time') as a Unix "
                "timestamp in seconds to place walking bouts on a wall-clock timeline."
            )
        
        return cls.from_uniform(start_epoch_s, len(data), datapoint.sampling_rate_hz, timezone=timezone)

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
    time_bins: Sequence[TimeBin] = ("hour", "day", "week"),
) -> pd.DataFrame:
    """Add wall-clock time and time bin columns to a walking bout table.

    Every walking bout is placed in the bin that contains its start, so a
    walking bout crossing midnight counts towards the day it started in. Its
    DMOs describe the bout as a whole and cannot be split across bins.

    The added columns turn the standard aggregator into a per-hour, per-day, or
    per-week aggregator::

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

    Returns
    -------
    pandas.DataFrame
        Copy of ``wb_dmos`` with a ``start_time`` column and one column per
        requested time bin, all timezone-naive local wall clock.
    """
    start_times = timeline.timestamps(wb_dmos["start"].to_numpy())
    new_columns = {"start_time": start_times}
    
    for time_bin in time_bins:
        new_columns[TIME_BIN_COLUMNS[time_bin]] = _floor_to_time_bin(start_times, time_bin)
    
    return wb_dmos.assign(**new_columns)


def time_bin_grid(timeline: RecordingTimeline, time_bin: TimeBin) -> pd.DatetimeIndex:
    """Return every time bin the recording reaches into.

    Parameters
    ----------
    timeline
        Wall-clock timeline of the recording.
    time_bin
        Time bin to build the grid for.

    Returns
    -------
    pandas.DatetimeIndex
        Start of every bin the recording overlaps, including bins without any
        walking bout.
    """
    edges = _floor_to_time_bin(pd.DatetimeIndex([timeline.start, timeline.end]), time_bin)
    grid = pd.date_range(edges[0], edges[1], freq=_TIME_BIN_FREQ[time_bin], name="bin_start")
    
    # a bin starting exactly when the recording ends contains no samples at all
    return grid[grid < timeline.end]


def is_complete_bin(grid: pd.DatetimeIndex, timeline: RecordingTimeline, time_bin: TimeBin) -> np.ndarray:
    """Flag the bins that the recording covers from their first to their last moment.

    Parameters
    ----------
    grid
        Start of every bin, as returned by :func:`time_bin_grid`.
    timeline
        Wall-clock timeline of the recording.
    time_bin
        Time bin the grid was built for.

    Returns
    -------
    numpy.ndarray
        Boolean mask, ``True`` for fully covered bins.
    """
    return (grid >= timeline.start) & (grid + TIME_BIN_WIDTHS[time_bin] <= timeline.end)
