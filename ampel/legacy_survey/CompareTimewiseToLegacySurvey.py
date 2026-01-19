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
        base_path = expand(self.base_path)
        self._plot_dir = base_path.parent
        self._plot_dir.mkdir(exist_ok=True, parents=True)
        self._name = base_path.name

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
            if t2ls.empty or t2tw.empty:
                continue

            median_ratios = []
            for i in range(1, 3):
                lsmed = t2ls[f"w{i}{keys.MEAN}{keys.FLUX_DENSITY_EXT}"].median()
                lsstd = t2ls[f"w{i}{keys.MEAN}{keys.FLUX_DENSITY_EXT}"].std()
                twmed = t2tw[f"w{i}{keys.MEAN}{keys.FLUX_DENSITY_EXT}"].median()
                twstd = t2tw[f"w{i}{keys.MEAN}{keys.FLUX_DENSITY_EXT}"].std()
                median_ratios.append(
                    (twmed - lsmed) / np.sqrt(twstd**2 + lsstd**2)
                    if (twstd > 0 and lsstd > 0)
                    else np.nan
                )

            if (
                (self.threshold_to_plot is not None)
                and (plot_ctr < self.max_plot)
                and (
                    10 ** max(abs(np.log10(np.abs(median_ratios))))
                    > self.threshold_to_plot
                )
            ):
                self.plot_lightcurves(t2ls, t2tw, view, median_ratios)
                plot_ctr += 1

            res.append(median_ratios)

        res = np.array(res)

        fig, ax = plt.subplots(figsize=(3 * 1.618, 3))
        ax.hist(np.log10(res[:, 0]), bins=30, alpha=0.5, label="W1", color="lightcoral")
        ax.hist(np.log10(res[:, 1]), bins=30, alpha=0.5, label="W2", color="maroon")
        ax.set_xlabel(
            r"$(\mu_\mathrm{tw} - \mu_\mathrm{LS}) / \sqrt{\sigma_\mathrm{tw}^2 + \sigma_\mathrm{LS}^2}$"
        )
        ax.set_ylabel("Count")
        ax.legend()
        fn = self._plot_dir / (
            self._name + "compare_timewise_legacy_survey_median_ratio_hist.pdf"
        )
        fig.suptitle("Median Flux Ratio: Legacy Survey vs Timewise")
        fig.tight_layout()
        fig.savefig(fn)
        plt.close(fig)

        return None

    def plot_lightcurves(
        self,
        t2ls: pd.DataFrame,
        t2tw: pd.DataFrame,
        view: TransientView,
        median_ratios: list[float],
    ) -> None:
        fig, ax = plt.subplots(figsize=(3 * 1.618, 4))
        plot_lightcurve(
            lum_key=keys.FLUX_DENSITY_EXT,
            stacked_lightcurve=t2tw,
            ax=ax,
            add_to_label=" Timewise",
            colors={"w1": "lightcoral", "w2": "maroon"},
        )

        lsc = {"w1": "lightsteelblue", "w2": "navy"}
        for b in ["w1", "w2"]:
            ax.errorbar(
                t2ls[f"LC_MJD_{b.upper()}"],
                t2ls[f"{b}{keys.MEAN}{keys.FLUX_DENSITY_EXT}"],
                yerr=t2ls[f"{b}{keys.FLUX_DENSITY_EXT}{keys.RMS}"],
                label=f"{b} Legacy Survey",
                ls="",
                marker="s",
                c=lsc[b],
                markersize=4,
                markeredgecolor="none",
                ecolor=lsc[b],
                capsize=2,
                zorder=3,
                barsabove=True,
                elinewidth=0.5,
            )

        l = f"W1 median ratio: {median_ratios[0]:.2f}, W2 median ratio: {median_ratios[1]:.2f}"
        ax.set_title(l)

        ax.set_ylabel("Flux Density (mJy)")
        ax.set_xlabel("MJD")
        ax.legend()
        d = self._plot_dir / self._name
        d.mkdir(exist_ok=True, parents=True)
        fn = d / f"{view.id}_compare_timewise_legacy_survey.pdf"
        fig.suptitle(f"Transient {view.id}")
        fig.tight_layout()
        fig.savefig(fn)
        plt.close(fig)
