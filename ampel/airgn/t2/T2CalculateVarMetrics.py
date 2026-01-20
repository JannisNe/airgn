from typing import Sequence, Literal, TypedDict, Callable

import pandas as pd
import numpy as np
import numpy.typing as npt

from ampel.abstract.AbsTiedLightCurveT2Unit import AbsTiedLightCurveT2Unit
from ampel.struct.UnitResult import UnitResult
from ampel.types import UBson
from ampel.view.LightCurve import LightCurve
from ampel.view.T2DocView import T2DocView
from ampel.model.StateT2Dependency import StateT2Dependency


from timewise.process import keys


class MetricOptions(TypedDict):
    log: bool
    range: tuple[float, float]
    pretty_name: str


float_arr = npt.NDArray[np.floating]
MetricFunc = Callable[[float_arr, float_arr, float_arr], float | None]


class T2CalculateVarMetrics(AbsTiedLightCurveT2Unit):
    t2_dependency: Sequence[
        StateT2Dependency[Literal["T2StackVisits", "T2MaggyToFluxDensity"]]
    ]

    _metrics = {}
    _metric_options = {}

    @classmethod
    def register(
        cls, log: bool, range: tuple[float, float], pretty_name: str
    ) -> Callable:
        def decorator(func: MetricFunc) -> MetricFunc:
            cls._metrics[func.__name__] = func
            cls._metric_options[func.__name__] = MetricOptions(
                log=log, range=range, pretty_name=pretty_name
            )
            return func

        return decorator

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

                f = stacked_lightcurve[f"w{i}{keys.MEAN}{key}"]
                fe = stacked_lightcurve[f"w{i}{key}{keys.RMS}"]
                t = stacked_lightcurve["mean_mjd"]
                for metric_name, metric_func in self._metrics.items():
                    res[f"{metric_name}_w{i}_{key}"] = metric_func(f, fe, t)

        return res


@T2CalculateVarMetrics.register(
    log=True, range=(-2, 2), pretty_name=r"$\chi_\mathrm{red}^2$"
)
def red_chi2(f: float_arr, fe: float_arr, t: float_arr) -> float | None:
    if len(f) > 1:
        return sum(((f - np.mean(f)) / fe) ** 2) / (len(f) - 1)
    return None


@T2CalculateVarMetrics.register(
    log=False, range=(0, 30), pretty_name=r"$N_\mathrm{points}$"
)
def npoints(f: float_arr, fe: float_arr, t: float_arr) -> float:
    return len(f)
