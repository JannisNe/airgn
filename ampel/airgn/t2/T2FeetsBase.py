import pandas as pd
import numpy as np
from typing import Optional, Sequence

import feets
from feets import extractor_registry
from ampel.base.LogicalUnit import LogicalUnit


class PearsonsR(feets.Extractor):
    """Pearson's r coefficient"""

    features = ["PearsonsR"]

    def extract(
        self, aligned_magnitude, aligned_magnitude2, aligned_error, aligned_error2
    ):
        mean1 = np.average(aligned_magnitude, weights=1 / aligned_error**2)
        diff1 = aligned_magnitude - mean1
        mean2 = np.average(aligned_magnitude2, weights=1 / aligned_error2**2)
        diff2 = aligned_magnitude2 - mean2
        return {
            "PearsonsR": sum(diff1 * diff2)
            / (np.sqrt(sum(diff1**2)) * np.sqrt(sum(diff2**2)))
        }


extractor_registry.register_extractor(PearsonsR)


class RedderWhenBrighter(feets.Extractor):
    """Linear fit to color vs magnitude"""

    features = ["RedderWhenBrighter1", "RedderWhenBrighter2"]

    def extract(
        self, aligned_magnitude, aligned_magnitude2, aligned_error, aligned_error2
    ):
        color = aligned_magnitude - aligned_magnitude2
        color_e = np.sqrt(aligned_error**2 + aligned_error2**2)

        results = {}
        for i, x in enumerate([aligned_magnitude, aligned_magnitude2]):
            a, b = np.polyfit(x, color, 1, w=1 / color_e)
            results[f"RedderWhenBrighter{i}"] = a

        return results


extractor_registry.register_extractor(RedderWhenBrighter)


class T2FeetsBase(LogicalUnit):
    filters: Sequence[str]
    single_band_features: Sequence[str]
    multi_band_features: Sequence[str]

    # column names
    # if row_per_filter is True then value_col and error_col must ba a formattable string
    # with the band as a variable like "{band}mag"
    time_col: str
    value_col: str
    error_col: str

    # All filters share the same row (like WISE data)
    # or there is one row per datapoint (like ZTF or Rubin).
    # In the second case the filter column has to be given
    row_per_filter: bool
    filter_col: Optional[str] = None

    # requirements for feature extraction
    min_points: int = 5

    # Optional: define explicit band pairs for multi-band features
    band_pairs: Optional[Sequence[tuple[str, str]]] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.row_per_filter and (self.filter_col is None):
            raise ValueError("filter_col must be specified if row_per_filter is True")
        self._single_band_extractor = feets.FeatureSpace(only=self.single_band_features)
        self._multi_band_extractor = feets.FeatureSpace(only=self.multi_band_features)

    # --------------------------------------------------------

    def _prepare_band_df(self, light_curve: pd.DataFrame, band: str) -> pd.DataFrame:
        if self.row_per_filter:
            band_mask = light_curve[self.filter_col] == band
            df = light_curve[band_mask]
            value_col = self.value_col
            error_col = self.error_col
            time_col = self.time_col

        else:
            df = light_curve
            time_col = self.time_col.format(band=band)
            value_col = self.value_col.format(band=band)
            error_col = self.error_col.format(band=band)

        # bring data in feets-required format
        return df[[time_col, value_col, error_col]].rename(
            columns={
                time_col: "time",
                value_col: "magnitude",
                error_col: "error",
            }
        )

    # --------------------------------------------------------

    def _extract_single_band(self, df: pd.DataFrame):
        results = {}

        for band in self.filters:
            df = self._prepare_band_df(df, band)

            # extract features
            features = self._single_band_extractor.extract(**df.to_dict("list"))

            # write to results including the band
            results.update(
                {f"{band}_{k}": v for k, v in features.as_frame().loc[0].items()}
            )

        return results

    # --------------------------------------------------------

    def _extract_multi_band(self, light_curve: pd.DataFrame):
        pairs = (
            self.band_pairs
            if self.band_pairs
            else [
                (self.filters[i], self.filters[j])
                for i in range(len(self.filters))
                for j in range(i + 1, len(self.filters))
            ]
        )

        results = {}
        for b1, b2 in pairs:
            df1 = self._prepare_band_df(light_curve, b1)
            df2 = self._prepare_band_df(light_curve, b2)

            if df1 is None or df2 is None:
                continue

            lc = {
                "time": df1.time,
                "magnitude": df1.magnitude,
                "error": df1.error,
                "time2": df2.time,
                "magnitude2": df2.magnitude,
                "error2": df2.error2,
            }

            # Synchronize the data from the two bands
            atime, amag, amag2, aerror, aerror2 = feets.preprocess.align(**lc)

            # For convenience, we store the preprocessed data in a dictionary.
            lc.update(
                {
                    "aligned_time": atime,
                    "aligned_magnitude": amag,
                    "aligned_magnitude2": amag2,
                    "aligned_error": aerror,
                    "aligned_error2": aerror2,
                }
            )

            features = self._multi_band_extractor.extract(**lc)

            for k, v in features.as_frame().loc[0].items():
                results[f"{b1}_{b2}{k}"] = v

        return results

    # --------------------------------------------------------

    def extract_feets(self, light_curve: pd.DataFrame) -> dict:
        results = {}
        results.update(self._extract_single_band(light_curve))
        results.update(self._extract_multi_band(light_curve))
        return results
