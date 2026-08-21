"""UoS-MobGap aggregation extensions for MobGap.

This extension adds per-hour and per-day DMO aggregation on top of the
aggregators of :mod:`mobgap.aggregation`.
"""

from mobgap.aggregation.uos._multi_granular import (
    COVERAGE_COLUMN,
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
    bin_coverage,
    time_bin_grid,
)

__all__ = [
    "COVERAGE_COLUMN",
    "TIME_BIN_COLUMNS",
    "TIME_BIN_ORDER",
    "TIME_BIN_WIDTHS",
    "TOTAL_COLUMNS",
    "MultiGranularAggregator",
    "RecordingTimeline",
    "TimeBin",
    "Weighting",
    "add_time_bins",
    "bin_coverage",
    "time_bin_grid",
]
