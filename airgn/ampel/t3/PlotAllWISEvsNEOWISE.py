from collections.abc import Generator
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ampel.abstract.AbsPhotoT3Unit import AbsPhotoT3Unit
from ampel.struct.T3Store import T3Store
from ampel.types import T3Send
from ampel.view.TransientView import TransientView
from ampel.timewise.util.pdutil import datapoints_to_dataframe

from timewise.plot.lightcurve import plot_lightcurve
from timewise.process import keys


class PlotAllWISEvsNEOWISE(AbsPhotoT3Unit):
    """
    Plot lightcurves of transients using matplotlib
    """

    path: str
    thresh_to_plot: float = 5.0
    plot_dir: str = "./"

    def process(
        self, gen: Generator[TransientView, T3Send, None], t3s: None | T3Store = None
    ) -> None:
        plot_dir = Path(self.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)

        columns = [
            "ra",
            "dec",
            "mjd",
        ]
        for i in range(1, 3):
            for key in [keys.MAG_EXT, keys.FLUX_EXT]:
                columns.extend([f"w{i}{key}", f"w{i}{keys.ERROR_EXT}{key}"])

        res = []
        for view in gen:
            t2res = view.get_t2_body(unit="T2CalculateMedians")
            for b in ["w1", "w2"]:
                res.append(t2res)

            if any(
                t2res
                and (k := f"allwise_neowise_ratio_w{b}") in t2res
                and t2res[k] > self.thresh_to_plot
                for b in ["w1", "w2"]
            ):
                raw_lightcurve = datapoints_to_dataframe(
                    view.get_photopoints(), columns
                )[0]
                stacked_lc = pd.DataFrame(
                    view.get_t2_body(unit="T2StackVisits", ret_type=tuple),
                )
                fig, ax = plot_lightcurve(
                    lum_key=keys.FLUX_EXT,
                    stacked_lightcurve=stacked_lc,
                    raw_lightcurve=raw_lightcurve,
                )
                fig.suptitle(f"Transient {view.id}")
                fig.savefig(plot_dir / f"chi2_exceed_{view.id}.pdf")
                plt.close(fig)

        res = pd.DataFrame(res)

        fig, axs = plt.subplots(nrows=2, sharex="all", sharey="all")
        for i in range(1, 3):
            ax = axs[i - 1]
            ax.scatter(
                res[f"median_w{i}_allwise"],
                res[f"allwise_neowise_ratio_w{i}"],
                alpha=0.5,
                s=1,
            )
            ax.set_ylabel(f"w{i}")
            ax.axhline(1.0, color="k", ls="--")

        axs[-1].set_xscale("log")
        axs[-1].set_xlabel("Median AllWISE Flux Density")
        fig.supylabel("AllWISE / NEOWISE Flux Density Ratio")

        fig.savefig(self.path)
        plt.close()
