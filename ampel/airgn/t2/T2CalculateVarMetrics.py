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

from airgn.legacy_survey.util import align_legacy_survey_photometry
from timewise.process import keys


class MetricMeta(TypedDict):
    log: bool
    range: tuple[float, float]
    pretty_name: str
    multiband: bool


float_arr = npt.NDArray[np.floating]

MetricSingleFunc = Callable[[float_arr, float_arr, float_arr], float | None]
MetricMultiFunc = Callable[
    [
        float_arr,
        float_arr,
        float_arr,
        float_arr,
        float_arr,
        float_arr,
    ],
    float | None,
]


MJD_COLNAMES = {
    "T2StackVisits": "mean_mjd",
    "T2MaggyToFluxDensity": "LC_MJD_W{band}",
}


class T2CalculateVarMetrics(AbsTiedLightCurveT2Unit):
    t2_dependency: Sequence[
        StateT2Dependency[Literal["T2StackVisits", "T2MaggyToFluxDensity"]]
    ]
    metric_names: list[str] = []

    _metrics = {}
    _metric_meta = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.metric_names:
            self.metric_names = list(self._metrics.keys())
        self._single_band_metric_names = [
            n for n in self.metric_names if not self._metric_meta[n]["multiband"]
        ]
        self._multi_band_metric_names = [
            n for n in self.metric_names if self._metric_meta[n]["multiband"]
        ]
        self._mjd_colname = {
            i: MJD_COLNAMES[self.t2_dependency[0].unit].format(band=i)
            for i in range(1, 3)
        }

    from typing import Callable, overload, Literal

    @classmethod
    @overload
    def register(
        cls,
        log: bool,
        range: tuple[float, float],
        pretty_name: str,
        multiband: Literal[False],
    ) -> Callable[[MetricSingleFunc], MetricSingleFunc]: ...

    @classmethod
    @overload
    def register(
        cls,
        log: bool,
        range: tuple[float, float],
        pretty_name: str,
        multiband: Literal[True],
    ) -> Callable[[MetricMultiFunc], MetricMultiFunc]: ...

    @classmethod
    def register(
        cls,
        log: bool,
        range: tuple[float, float],
        pretty_name: str,
        multiband: bool,
    ):
        def decorator(func):
            cls._metrics[func.__name__] = func
            cls._metric_meta[func.__name__] = MetricMeta(
                log=log, range=range, pretty_name=pretty_name, multiband=multiband
            )
            return func

        return decorator

    def process(
        self, light_curve: LightCurve, t2_views: Sequence[T2DocView]
    ) -> UBson | UnitResult:
        records = [r.body[0] for r in t2_views][0]
        stacked_lightcurve = pd.DataFrame.from_records(records)

        res = {}
        for key in [keys.FLUX_EXT, keys.FLUX_DENSITY_EXT]:
            # ---------------- calculate single band metrics ---------------- #

            for i in range(1, 3):
                nan_msak = (
                    stacked_lightcurve[f"w{i}{keys.MEAN}{key}"].notna()
                    | stacked_lightcurve[f"w{i}{key}{keys.RMS}"].notna()
                )
                sel_data = stacked_lightcurve[nan_msak]

                f = sel_data[f"w{i}{keys.MEAN}{key}"].values
                fe = sel_data[f"w{i}{key}{keys.RMS}"].values
                t = sel_data[self._mjd_colname[i]].values
                for metric_name in self._single_band_metric_names:
                    res[f"{metric_name}_w{i}_{key}"] = self._metrics[metric_name](
                        f, fe, t
                    )

            # ---------------- calculate multi band metrics ---------------- #
            nan_msak = (
                stacked_lightcurve[f"w1{keys.MEAN}{key}"].notna()
                | stacked_lightcurve[f"w1{key}{keys.RMS}"].notna()
                | stacked_lightcurve[f"w2{keys.MEAN}{key}"].notna()
                | stacked_lightcurve[f"w2{key}{keys.RMS}"].notna()
            )
            sel_data = stacked_lightcurve[nan_msak]
            if self.t2_dependency[0].unit == "T2MaggyToFluxDensity":
                sel_data = align_legacy_survey_photometry(sel_data)

            f1 = sel_data[f"w1{keys.MEAN}{key}"].values
            fe1 = sel_data[f"w1{key}{keys.RMS}"].values
            t1 = sel_data[self._mjd_colname[1]].values
            f2 = sel_data[f"w2{keys.MEAN}{key}"].values
            fe2 = sel_data[f"w2{key}{keys.RMS}"].values
            t2 = sel_data[self._mjd_colname[2]].values

            for metric_name in self._multi_band_metric_names:
                res[f"{metric_name}_{key}"] = self._metrics[metric_name](
                    f1, fe1, t1, f2, fe2, t2
                )

        return res


@T2CalculateVarMetrics.register(
    log=True, range=(-2, 2), pretty_name=r"$\chi_\mathrm{red}^2$", multiband=False
)
def red_chi2(f: float_arr, fe: float_arr, t: float_arr) -> float | None:
    if len(f) > 1:
        f_mean = np.average(f, weights=1 / fe**2)
        return sum(((f - f_mean) / fe) ** 2) / (len(f) - 1)
    return None


@T2CalculateVarMetrics.register(
    log=False, range=(0, 30), pretty_name=r"$N_\mathrm{points}$", multiband=False
)
def npoints(f: float_arr, fe: float_arr, t: float_arr) -> float:
    return len(f)


@T2CalculateVarMetrics.register(
    log=False, range=(-1, 1), pretty_name=r"IQR$_\mathrm{rel}$", multiband=False
)
def relative_inter_quartile_range(
    f: float_arr, fe: float_arr, t: float_arr
) -> float | None:
    if len(f) > 1:
        f_med = np.median(f)
        upper_half = f >= f_med
        return float((np.median(f[upper_half]) - np.median(f[~upper_half])) / f_med)
    return None


@T2CalculateVarMetrics.register(
    log=True, range=(-1, 2), pretty_name=r"$1/\eta$", multiband=False
)
def inverse_von_neumann_ratio(
    f: float_arr, fe: float_arr, t: float_arr
) -> float | None:
    if len(f) > 1:
        sort_mask = np.argsort(t)
        f = f[sort_mask]
        fe = fe[sort_mask]
        f_mean = np.average(f, weights=1 / fe**2)
        return sum((f - f_mean) ** 2) / sum((f[1:] - f[:-1]) ** 2)
    return None


@T2CalculateVarMetrics.register(
    log=False, range=(-1, 1), pretty_name=r"$\sigma^2_\mathrm{rms}$", multiband=False
)
def normalized_excess_variance(
    f: float_arr, fe: float_arr, t: float_arr
) -> float | None:
    if len(f) > 0:
        mu = np.average(f, weights=1 / fe**2)
        return float(sum((f - mu) ** 2 - fe**2) / (len(f) * mu**2))
    return None


@T2CalculateVarMetrics.register(
    log=False, range=(-1, 1), pretty_name="$r$", multiband=True
)
def pearsons_r(
    f1: float_arr,
    fe1: float_arr,
    t1: float_arr,
    f2: float_arr,
    fe2: float_arr,
    t2: float_arr,
):
    assert len(f1) == len(f2), "Both flux arrays must have same length!"
    N = len(f1)
    mean1 = np.average(f1, weights=1 / fe1**2)
    diff1 = f1 - mean1
    mean2 = np.average(f2, weights=1 / fe2**2)
    diff2 = f2 - mean2
    return (N - 1) * sum(diff1 * diff2) / (sum(diff1**2) * sum(diff2**2))


@T2CalculateVarMetrics.register(
    log=False, range=(-1, 1), pretty_name="$r_\mathrm{log}$", multiband=True
)
def pearsons_r_log(
    f1: float_arr,
    fe1: float_arr,
    t1: float_arr,
    f2: float_arr,
    fe2: float_arr,
    t2: float_arr,
):
    assert len(f1) == len(f2), "Both flux arrays must have same length!"
    both_detections = (f1 > 0) & (f2 > 0)
    N = sum(both_detections)
    m1 = 2.5 * np.log10(f1[both_detections])
    m2 = 2.5 * np.log10(f2[both_detections])
    mean1 = sum(m1) / N
    mean2 = sum(m2) / N
    diff1 = m1 - mean1
    diff2 = m2 - mean2
    return (N - 1) * sum(diff1 * diff2) / (sum(diff1**2) * sum(diff2**2))
