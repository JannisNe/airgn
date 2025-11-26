from typing import Sequence, Literal

import pandas as pd

from ampel.abstract.AbsTiedLightCurveT2Unit import AbsTiedLightCurveT2Unit
from ampel.struct.UnitResult import UnitResult
from ampel.types import UBson
from ampel.view.LightCurve import LightCurve
from ampel.view.T2DocView import T2DocView
from ampel.model.StateT2Dependency import StateT2Dependency

from timewise.process import keys


class T2CalculateMedians(AbsTiedLightCurveT2Unit):
    t2_dependency: Sequence[
        StateT2Dependency[Literal["T2StackVisits", "T2MaggyToFluxDensity"]]
    ]

    def process(
        self, light_curve: LightCurve, t2_views: Sequence[T2DocView]
    ) -> UBson | UnitResult:
        records = [r.body[0] for r in t2_views][0]
        data = pd.DataFrame.from_records(records)

        res = {}
        for i in range(1, 3):
            fd = data[f"w{i}{keys.MEAN}{keys.FLUX_DENSITY_EXT}"]
            fd = fd[fd.notna()]
            neowise_mask = data.loc[fd.notna(), "mean_mjd"] >= 56000
            res[f"median_w{i}_all"] = fd.median()
            res[f"median_w{i}_neowise"] = (
                fd[neowise_mask].median() if not fd[neowise_mask].empty else None
            )
            res[f"median_w{i}_allwise"] = (
                fd[~neowise_mask].median() if not fd[~neowise_mask].empty else None
            )
            res[f"allwise_neowise_ratio_w{i}"] = (
                res[f"median_w{i}_allwise"] / res[f"median_w{i}_neowise"]
                if (
                    (res[f"median_w{i}_neowise"] not in (0, None))
                    and (res[f"median_w{i}_allwise"] is not None)
                )
                else None
            )

        return res
