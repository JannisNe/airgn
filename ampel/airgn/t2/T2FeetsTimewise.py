from typing import Sequence, Literal

import pandas as pd
import numpy as np

from ampel.abstract.AbsTiedLightCurveT2Unit import AbsTiedLightCurveT2Unit
from ampel.struct.UnitResult import UnitResult
from ampel.types import UBson
from ampel.view.LightCurve import LightCurve
from ampel.view.T2DocView import T2DocView
from ampel.model.StateT2Dependency import StateT2Dependency

from ampel.airgn.t2.T2FeetsBase import T2FeetsBase
from timewise.process.stacking import FLUX_ZEROPOINTS

import feets
from feets import extractor_registry


MJD_COLNAMES = {
    "T2StackVisits": "mean_mjd",
    "T2MaggyToFluxDensity": "LC_MJD_W{band}",
}


# --------------------------------------------------------
# DEFINE NEW FEATURES


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


class RedderWhenBrighter(feets.Extractor):
    """Linear fit to color vs magnitude"""

    features = ["RedderWhenBrighter0", "RedderWhenBrighter1"]

    def extract(
        self, aligned_magnitude, aligned_magnitude2, aligned_error, aligned_error2
    ):
        color = aligned_magnitude - aligned_magnitude2
        color_e = np.sqrt(aligned_error**2 + aligned_error2**2)

        results = {}
        for i, x in enumerate([aligned_magnitude, aligned_magnitude2]):
            try:
                a, b = np.polyfit(-x, color, 1, w=1 / color_e)
            except np.linalg.LinAlgError:
                a = np.nan
            results[f"RedderWhenBrighter{i}"] = a

        return results


class NPoints(feets.Extractor):
    """Count number of detections"""

    features = ["NPoints"]

    def extract(self, magnitude):
        return {"NPoints": len(magnitude)}


class InverseEta(feets.Extractor):
    """Inverse of Eta"""

    features = ["InverseEta"]

    def extract(self, Eta):
        return {"InverseEta": 1 / Eta}


class InverseEtaColor(feets.Extractor):
    """Inverse of EtaColor"""

    features = ["InverseEtaColor"]

    def extract(self, Eta_color):
        return {"InverseEtaColor": 1 / Eta_color}


# --------------------------------------------------------
# REGISTER FEATURES


for ext in [PearsonsR, RedderWhenBrighter, NPoints, InverseEta, InverseEtaColor]:
    extractor_registry.register_extractor(ext)


# --------------------------------------------------------


class T2FeetsTimewise(AbsTiedLightCurveT2Unit, T2FeetsBase):
    t2_dependency: Sequence[
        StateT2Dependency[Literal["T2StackVisits", "T2MaggyToFluxDensity"]]
    ]

    # column names specific for timewise
    filters = ["W1", "W2"]
    time_col = "mean_mjd"
    value_col = "{band}mag"
    error_col = "e_{band}mag"
    row_per_filter = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(
        self, light_curve: LightCurve, t2_views: Sequence[T2DocView]
    ) -> UBson | UnitResult:
        records = [r.body[0] for r in t2_views][0]
        stacked_lightcurve = pd.DataFrame.from_records(records)

        for i in range(1, 3):
            f = stacked_lightcurve[f"w{i}meanfluxdensity"]  # in mJy
            fe = stacked_lightcurve[f"w{i}fluxdensityrms"]  # in mJy
            stacked_lightcurve[f"W{i}mag"] = -2.5 * np.log10(
                f * 1e-3 / FLUX_ZEROPOINTS[f"w{i}"]
            )
            stacked_lightcurve[f"e_W{i}mag"] = 2.5 / np.log(10) * fe / f

        return self.extract_feets(stacked_lightcurve)
