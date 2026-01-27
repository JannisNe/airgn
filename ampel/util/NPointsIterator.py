from typing import Generator
from itertools import pairwise

import pandas as pd

from ampel.base.AmpelABC import AmpelABC


class NPointsIterator(AmpelABC):
    n_points_bins: tuple[int, ...] = (0, 10, 20, 30)
    n_point_cols: list[str]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._min_edge = min(self.n_points_bins)
        self._max_edge = max(self.n_points_bins)

    def is_in_range(self, val: float) -> bool:
        return self._min_edge <= val <= self._max_edge

    def iter_npoints_binned(
        self, df: pd.DataFrame
    ) -> Generator[tuple[pd.DataFrame, float, float], None, None]:
        bins = list(pairwise(self.n_points_bins))
        if all_bin := (min(self.n_points_bins), max(self.n_points_bins)) not in bins:
            bins.append(all_bin)
        for s, e in bins:
            bin_mask = (df[self.n_point_cols] >= s).all(axis=1) & (
                df[self.n_point_cols] < e
            ).any(axis=1)
            yield df[bin_mask], s, e
