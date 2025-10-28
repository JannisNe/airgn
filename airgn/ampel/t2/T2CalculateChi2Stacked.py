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
            nan_msak = (
                stacked_lightcurve[f"w{i}{keys.MEAN}{keys.FLUX_DENSITY_EXT}"].notna()
                | stacked_lightcurve[f"w{i}{keys.FLUX_DENSITY_EXT}{keys.RMS}"].notna()
            )
            stacked_lightcurve = stacked_lightcurve[nan_msak]
            res[f"chi2_w{i}"] = sum(
                (
                    (
                        stacked_lightcurve[f"w{i}{keys.MEAN}{keys.FLUX_DENSITY_EXT}"]
                        - stacked_lightcurve[
                            f"w{i}{keys.MEAN}{keys.FLUX_DENSITY_EXT}"
                        ].median()
                    )
                    / stacked_lightcurve[f"w{i}{keys.FLUX_DENSITY_EXT}{keys.RMS}"]
                )
                ** 2
            )
            res[f"npoints_w{i}"] = sum(nan_msak)
            res[f"red_chi2_w{i}"] = (
                res[f"chi2_w{i}"] / res[f"npoints_w{i}"]
                if res[f"npoints_w{i}"] > 0
                else None
            )

        return res
