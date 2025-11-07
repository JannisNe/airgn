from typing import Sequence, Literal

import pandas as pd
from ampel.abstract.AbsTiedLightCurveT2Unit import AbsTiedLightCurveT2Unit
from ampel.struct.UnitResult import UnitResult
from ampel.types import UBson
from ampel.view.LightCurve import LightCurve
from ampel.view.T2DocView import T2DocView
from ampel.model.StateT2Dependency import StateT2Dependency


from timewise.process import keys


class T2CalculateChi2Stacked(AbsTiedLightCurveT2Unit):
    t2_dependency: Sequence[StateT2Dependency[Literal["T2StackVisits"]]]

    def process(
        self, light_curve: LightCurve, t2_views: Sequence[T2DocView]
    ) -> UBson | UnitResult:
        records = [r.body[0] for r in t2_views][0]
        stacked_lightcurve = pd.DataFrame.from_records(records)

        res = {}
        for i in range(1, 3):
            for key in [keys.FLUX_EXT, keys.FLUX_DENSITY_EXT]:
                nan_msak = (
                    stacked_lightcurve[f"w{i}{keys.MEAN}{key}"].notna()
                    | stacked_lightcurve[f"w{i}{key}{keys.RMS}"].notna()
                )
                stacked_lightcurve = stacked_lightcurve[nan_msak]
                res[f"chi2_w{i}_{key}"] = sum(
                    (
                        (
                            stacked_lightcurve[f"w{i}{keys.MEAN}{key}"]
                            - stacked_lightcurve[f"w{i}{keys.MEAN}{key}"].mean()
                        )
                        / stacked_lightcurve[f"w{i}{key}{keys.RMS}"]
                    )
                    ** 2
                )
                res[f"npoints_w{i}_{key}"] = sum(nan_msak)
                res[f"red_chi2_w{i}_{key}"] = (
                    res[f"chi2_w{i}_{key}"] / (res[f"npoints_w{i}_{key}"] - 1)
                    if res[f"npoints_w{i}_{key}"] > 0
                    else None
                )

        return res
