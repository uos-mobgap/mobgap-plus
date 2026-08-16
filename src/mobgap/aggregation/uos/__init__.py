"""UoS-MobGap aggregation extensions for MobGap.

This extension adds per-hour, per-day, and per-week DMO aggregation on top of the
aggregators of :mod:`mobgap.aggregation`.
"""

from mobgap.aggregation.uos._multi_granular import (
    TIME_BIN_ORDER,
    TOTAL_COLUMNS,
    MultiGranularAggregator,
    Weighting,
)
from mobgap.aggregation.uos._time_bins import (
    TIME_BIN_COLUMNS,
    TIME_BIN_WIDTHS,
    RecordingTimeline,
    TimeBin,
    add_time_bins,
    is_complete_bin,
    time_bin_grid,
)

__all__ = [
    "TIME_BIN_COLUMNS",
    "TIME_BIN_ORDER",
    "TIME_BIN_WIDTHS",
    "TOTAL_COLUMNS",
    "MultiGranularAggregator",
    "RecordingTimeline",
    "TimeBin",
    "Weighting",
    "add_time_bins",
    "is_complete_bin",
    "time_bin_grid",
]