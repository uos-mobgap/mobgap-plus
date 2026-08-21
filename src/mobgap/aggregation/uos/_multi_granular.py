"""Aggregate DMOs per hour and per day.

UoS-MobGap extension.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, Literal

import pandas as pd
from tpcp import cf
from typing_extensions import Self, Unpack

from mobgap.aggregation import MobilisedAggregator
from mobgap.aggregation.base import BaseAggregator, base_aggregator_docfiller
from mobgap.aggregation.uos._time_bins import (
    TIME_BIN_COLUMNS,
    RecordingTimeline,
    TimeBin,
    _floor_to_time_bin,
    add_time_bins,
    bin_coverage,
    time_bin_grid,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

Weighting = Literal["equal", "pooled"]

TIME_BIN_ORDER: Final = ("hour", "day")

COVERAGE_COLUMN: Final = "coverage"

# counts and totals of the Mobilise-D aggregator, under the original names of
# mobgap.aggregation.MobilisedAggregator (v1.2.0, _mobilised_aggregator.py:201-246)
# every other aggregated column is an average, a percentile, or a coefficient of variation
_ORIGINAL_TOTAL_COLUMNS: Final = (
    "wb_all_sum",
    "walkdur_all_sum",
    "wbsteps_all_sum",
    "turns_all_sum",
    "wb_1030_sum",
    "wb_10_sum",
    "wb_30_sum",
    "wb_60_sum",
)

TOTAL_COLUMNS: Final = frozenset(
    (*_ORIGINAL_TOTAL_COLUMNS, *(MobilisedAggregator.ALTERNATIVE_NAMES[c] for c in _ORIGINAL_TOTAL_COLUMNS))
)


@base_aggregator_docfiller
class MultiGranularAggregator(BaseAggregator):
    """Aggregate walking bout DMOs of one recording per hour and per day.

    Walking bouts are placed on the wall-clock timeline of the recording (see
    :class:`~mobgap.aggregation.uos.RecordingTimeline`) and aggregated within
    each time bin by a nested aggregator, by default
    :class:`~mobgap.aggregation.MobilisedAggregator`. A day is 24 hours long and
    starts at ``day_start_hour`` on the local clock.

    Bins the recording reaches into but that hold no walking bout are part of
    the result, with counts and totals of zero and averages of ``NaN``.

    The ``weighting`` decides how a walking bout is weighted within a day.
    Only averages, percentiles, and coefficients of variation change.
    Counts and totals are the same under both weightings.

    Parameters
    ----------
    time_bins
        Time bins to report, any subset of ``("hour", "day")``.
    weighting
        ``"equal"`` builds every day from its hours, so each hour counts the
        same. An hour of intense walking cannot dominate the daily averages.
        Hours without walking carry no average and are left out of the daily
        average, so a daily average describes the hours in which the participant
        walked.

        ``"pooled"`` aggregates all walking bouts of a day in one go, so hours
        are weighted by how many walking bouts they hold. This is what
        :class:`~mobgap.aggregation.MobilisedAggregator` does on its own and what
        the original Mobilise-D aggregation does.
    min_coverage
        Smallest fraction of a bin that the recording must hold samples for.
        The default of 0 reports every bin the recording reaches into.
        A value of 1 keeps only bins covered from their first to their last
        moment. Values in between tolerate samples lost to logger dropouts,
        like a minimum wear time. The coverage itself is reported in the
        ``coverage`` column. Dropping a bin never removes its walking bouts
        from a coarser bin.
    day_start_hour
        Hour of the local clock at which a day starts, from 0 to 23. Whole hours
        only, so that hourly bins always nest exactly 24 per day.
    aggregator
        Aggregator applied within each time bin. Its ``groupby`` is set by this
        class and ignored.

    Other Parameters
    ----------------
    %(other_parameters)s
    timeline
        The wall-clock timeline passed to the ``aggregate`` method.
    wb_dmos_mask
        The validity mask passed to the ``aggregate`` method, see
        :class:`~mobgap.aggregation.MobilisedAggregator`.

    Attributes
    ----------
    aggregated_data_
        Aggregation results with a ``("time_bin", "bin_start")`` multi-index,
        ordered from the finest to the coarsest requested time bin. The columns
        are the ones the nested aggregator produces, plus ``coverage``.
    binned_wb_dmos_
        ``wb_dmos`` with the ``start_time`` and time bin columns added.

    Notes
    -----
    Under ``"equal"`` weighting a daily statistic is the average of the hourly
    ones. That is the same question only for plain averages. The average of the
    hourly 90th percentiles is not the daily 90th percentile. The average of
    the hourly coefficients of variation describes variability inside each
    hour, not across the day. Request the hourly bin in the same call to see
    how many hours a daily value is computed from. That adds no extra pipeline
    work.

    Counts and totals are zero rather than missing whenever nothing was
    observed. :class:`~mobgap.aggregation.MobilisedAggregator` leaves
    ``wb_1030_sum`` missing in that case. This class reports it as zero so that
    all counts behave the same.
    """

    time_bins: Sequence[TimeBin]
    weighting: Weighting
    min_coverage: float
    day_start_hour: int
    aggregator: BaseAggregator

    timeline: RecordingTimeline
    wb_dmos_mask: pd.DataFrame | None

    binned_wb_dmos_: pd.DataFrame

    def __init__(
        self,
        *,
        time_bins: Sequence[TimeBin] = TIME_BIN_ORDER,
        weighting: Weighting = "equal",
        min_coverage: float = 0.0,
        day_start_hour: int = 0,
        aggregator: BaseAggregator = cf(
            MobilisedAggregator(**MobilisedAggregator.PredefinedParameters.single_recording)
        ),
    ) -> None:
        self.time_bins = time_bins
        self.weighting = weighting
        self.min_coverage = min_coverage
        self.day_start_hour = day_start_hour
        self.aggregator = aggregator

    @base_aggregator_docfiller
    def aggregate(
        self,
        wb_dmos: pd.DataFrame,
        *,
        timeline: RecordingTimeline,
        wb_dmos_mask: pd.DataFrame | None = None,
        **_: Unpack[dict[str, Any]],
    ) -> Self:
        """%(aggregate_short)s.

        Parameters
        ----------
        %(aggregate_para)s
            It further needs a ``start`` column with the sample index at which
            each walking bout starts.
        timeline
            Wall-clock timeline of the recording the walking bouts belong to.
        wb_dmos_mask
            Optional validity mask forwarded to the nested aggregator, see
            :class:`~mobgap.aggregation.MobilisedAggregator`.

        %(aggregate_return)s
        """
        self.wb_dmos = wb_dmos
        self.timeline = timeline
        self.wb_dmos_mask = wb_dmos_mask

        is_whole_hour = isinstance(self.day_start_hour, int) and not isinstance(self.day_start_hour, bool)
        if not (is_whole_hour and 0 <= self.day_start_hour < 24):
            raise ValueError(f"day_start_hour must be a whole hour between 0 and 23, got {self.day_start_hour!r}.")

        requested = self._sorted_time_bins(self.time_bins)
        if not requested:
            raise ValueError("time_bins must not be empty. Expected a non-empty subset of ('hour', 'day').")

        # equal weighting builds every day from the hourly bins, so the hours are always needed
        ladder = TIME_BIN_ORDER[: TIME_BIN_ORDER.index(requested[-1]) + 1] if self.weighting == "equal" else requested

        self.binned_wb_dmos_ = add_time_bins(wb_dmos, timeline, time_bins=ladder, day_start_hour=self.day_start_hour)
        mask = self._add_time_bins_to_mask(wb_dmos_mask, ladder)

        results = self._aggregate_ladder(ladder, mask)
        self.aggregated_data_ = pd.concat(
            {time_bin: self._report(results[time_bin], time_bin) for time_bin in requested}, names=["time_bin"]
        )

        return self

    @staticmethod
    def _sorted_time_bins(time_bins: Sequence[TimeBin]) -> tuple[TimeBin, ...]:
        """Order the requested time bins from the finest to the coarsest one."""
        unknown = set(time_bins) - set(TIME_BIN_ORDER)
        if unknown:
            raise ValueError(f"Unknown time bins {sorted(unknown)}. Expected a subset of {TIME_BIN_ORDER}.")

        return tuple(time_bin for time_bin in TIME_BIN_ORDER if time_bin in time_bins)

    def _add_time_bins_to_mask(
        self, wb_dmos_mask: pd.DataFrame | None, ladder: Sequence[TimeBin]
    ) -> pd.DataFrame | None:
        """Add the time bin columns to the mask, which the nested aggregator groups by."""
        if wb_dmos_mask is None:
            return None

        columns = [TIME_BIN_COLUMNS[time_bin] for time_bin in ladder]
        return wb_dmos_mask.assign(**{column: self.binned_wb_dmos_[column] for column in columns})

    def _aggregate_ladder(self, ladder: tuple[TimeBin, ...], mask: pd.DataFrame | None) -> dict[TimeBin, pd.DataFrame]:
        """Aggregate every time bin of the ladder, from the finest to the coarsest one."""
        if self.weighting == "pooled":
            return {time_bin: self._aggregate_time_bin(time_bin, mask) for time_bin in ladder}

        if self.weighting != "equal":
            raise ValueError(f"Unknown weighting {self.weighting!r}. Expected 'equal' or 'pooled'.")

        results = {}
        finer = None

        for time_bin in ladder:
            finer = self._aggregate_time_bin(time_bin, mask) if finer is None else self._combine(finer, time_bin)
            results[time_bin] = finer

        return results

    def _aggregate_time_bin(self, time_bin: TimeBin, mask: pd.DataFrame | None) -> pd.DataFrame:
        """Aggregate all walking bouts of every bin with the nested aggregator."""
        aggregator = self.aggregator.clone().set_params(groupby=[TIME_BIN_COLUMNS[time_bin]])
        aggregated = aggregator.aggregate(self.binned_wb_dmos_, wb_dmos_mask=mask).aggregated_data_

        return self._to_grid(aggregated, time_bin)

    def _combine(self, finer: pd.DataFrame, time_bin: TimeBin) -> pd.DataFrame:
        """Combine finer bins into coarser ones, weighting every finer bin equally."""
        how = {column: "sum" if column in TOTAL_COLUMNS else "mean" for column in finer.columns}
        keys = _floor_to_time_bin(finer.index, time_bin, self.day_start_hour)

        return self._to_grid(finer.groupby(keys).agg(how), time_bin)

    def _to_grid(self, aggregated: pd.DataFrame, time_bin: TimeBin) -> pd.DataFrame:
        """Put the results on the full bin grid of the recording."""
        grid = time_bin_grid(self.timeline, time_bin, day_start_hour=self.day_start_hour)

        # reindex below would drop these rows and their walking bouts without an error
        off_grid = aggregated.index.difference(grid)
        if len(off_grid):
            raise ValueError(
                f"{len(off_grid)} {time_bin} bin(s) hold walking bouts but lie outside the recording "
                f"timeline. The first is at {off_grid[0]}. The timeline runs from {self.timeline.start} to "
                f"{self.timeline.end}, so it does not describe the recording these walking bouts came from. "
                "Build it with RecordingTimeline.from_datapoint(), or pass the sample count of the very "
                "same recording to RecordingTimeline.from_uniform()."
            )

        total_columns = [column for column in aggregated.columns if column in TOTAL_COLUMNS]
        total_dtypes = aggregated[total_columns].dtypes.to_dict()

        aggregated = aggregated.reindex(grid)
        # Nothing observed in a bin means a count of zero.
        aggregated[total_columns] = aggregated[total_columns].fillna(0).astype(total_dtypes)

        return aggregated

    def _report(self, aggregated: pd.DataFrame, time_bin: TimeBin) -> pd.DataFrame:
        """Add the coverage of every bin and drop the ones below ``min_coverage``.

        Bins are judged on their own coverage only. A dropped hour still
        counts towards its day. The totals stay the same under both
        weightings.
        """
        coverage = bin_coverage(aggregated.index, self.timeline, time_bin)

        return aggregated.assign(**{COVERAGE_COLUMN: coverage})[coverage >= self.min_coverage]
