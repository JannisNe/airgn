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


MJD_COLNAMES = {
    "T2StackVisits": "mean_mjd",
    "T2MaggyToFluxDensity": "LC_MJD_W{band}",
}


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
            f = stacked_lightcurve[f"w{i}meanfluxdensity"]
            fe = stacked_lightcurve[f"w{i}fluxdensityrms"]
            stacked_lightcurve[f"W{i}mag"] = -2.5 * np.log10(
                f / FLUX_ZEROPOINTS[f"w{i}"]
            )
            stacked_lightcurve[f"e_W{i}mag"] = 2.5 / np.log(10) * fe / f

        return self.extract_feets(stacked_lightcurve)
