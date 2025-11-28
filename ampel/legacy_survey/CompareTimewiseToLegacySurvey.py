from collections.abc import Generator

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ampel.abstract.AbsPhotoT3Unit import AbsPhotoT3Unit
from ampel.struct.T3Store import T3Store
from ampel.view.TransientView import TransientView
from ampel.struct.UnitResult import UnitResult
from ampel.types import T3Send, UBson

from timewise.util.path import expand
from timewise.process import keys
from timewise.plot.lightcurve import plot_lightcurve


class CompareTimewiseToLegacySurvey(AbsPhotoT3Unit):
    """
    Plot lightcurves of transients using matplotlib
    """

    base_path: str
    threshold_to_plot: float | None = None
    max_plot: int = 100

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._base_path = expand(self.base_path)
        self._base_path.parent.mkdir(exist_ok=True, parents=True)

    def process(
        self, gen: Generator[TransientView, T3Send, None], t3s: None | T3Store = None
    ) -> UBson | UnitResult:
        plot_ctr = 0

        res = []
        for view in gen:
            t2ls = pd.DataFrame(
                view.get_t2_body("T2MaggyToFluxDensity", ret_type=tuple)
            )
            t2tw = pd.DataFrame(view.get_t2_body("T2StackVisits", ret_type=tuple))
            if t2ls is None or t2tw is None:
                continue

            median_ratios = [
                t2ls[f"w{i}{keys.MEAN}{keys.FLUX_DENSITY_EXT}"].median()
                / t2tw[f"w{i}{keys.MEAN}{keys.FLUX_DENSITY_EXT}"].median()
                for i in range(1, 3)
            ]

            if (max(abs(np.log10(median_ratios))) > self.threshold_to_plot) and (
                plot_ctr < self.max_plot
            ):
                self.plot_lightcurves(t2ls, t2tw, view)
                plot_ctr += 1

            res.append(median_ratios)

        res = np.array(res)

        fig, ax = plt.subplots()
        ax.hist(np.log10(res[:, 0]), bins=30, alpha=0.5, label="W1", color="lightcoral")
        ax.hist(np.log10(res[:, 1]), bins=30, alpha=0.5, label="W2", color="maroon")
        ax.set_xlabel("Log10(Median Legacy Survey Flux / Median Timewise Flux)")
        ax.set_ylabel("Count")
        ax.legend()
        fn = self._base_path / "compare_timewise_legacy_survey_median_ratio_hist.pdf"
        fig.suptitle("Median Flux Ratio: Legacy Survey vs Timewise")
        fig.savefig(fn)
        plt.close(fig)

        return None

    def plot_lightcurves(
        self, t2ls: pd.DataFrame, t2tw: pd.DataFrame, view: TransientView
    ) -> None:
        fig, ax = plt.subplots()
        plot_lightcurve(
            lum_key=keys.FLUX_DENSITY_EXT,
            stacked_lightcurve=t2tw,
            ax=ax,
            add_to_label=" Timewise",
            colors={"w1": "lightcoral", "w2": "maroon"},
        )
        plot_lightcurve(
            lum_key=keys.FLUX_DENSITY_EXT,
            raw_lightcurve=t2ls,
            ax=ax,
            add_to_label=" Legacy Survey",
            colors={"w1": "lightseeblue", "w2": "navy"},
        )
        fn = self._base_path / f"{view.id}_compare_timewise_legacy_survey.pdf"
        fig.suptitle(f"Transient {view.id}")
        fig.savefig(fn)
        plt.close(fig)
