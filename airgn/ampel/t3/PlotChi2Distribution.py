from collections.abc import Generator
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from pymongo import MongoClient
import pandas as pd

from ampel.abstract.AbsPhotoT3Unit import AbsPhotoT3Unit
from ampel.struct.T3Store import T3Store
from ampel.types import T3Send
from ampel.view.TransientView import TransientView
from ampel.timewise.util.pdutil import datapoints_to_dataframe

from timewise.plot.lightcurve import plot_lightcurve
from timewise.process import keys


class PlotChi2Distribution(AbsPhotoT3Unit):
    """
    Plot lightcurves of transients using matplotlib
    """

    path: str
    input_mongodb_name: str

    chi2_thresh_to_plot: float = 5.0
    plot_dir: str = "./"

    def process(
        self, gen: Generator[TransientView, T3Send, None], t3s: None | T3Store = None
    ) -> None:
        input_collection = MongoClient()[self.input_mongodb_name]["input"]
        qso_ids = [
            doc["orig_id"]
            for doc in input_collection.find({"objtype": "b'QSO'"}, {"orig_id": 1})
        ]

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

        units = ["T2CalculateChi2Stacked", "T2CalculateChi2"]
        chi2_res = {u: {"w1": [], "w2": []} for u in units}
        chi2_qso_res = {u: {"w1": [], "w2": []} for u in units}
        for view in gen:
            for unit in units:
                t2res = view.get_t2_body(unit=unit)
                for b in ["w1", "w2"]:
                    if t2res and (k := f"red_chi2_{b}") in t2res:
                        if view.id in qso_ids:
                            chi2_qso_res[unit][b].append(t2res[k])
                        else:
                            chi2_res[unit][b].append(t2res[k])

                if any(
                    t2res
                    and (k := f"red_chi2_{b}") in t2res
                    and t2res[k] > self.chi2_thresh_to_plot
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
                    fig.suptitle(f"Transient {view.id} (unit: {unit})")
                    fig.savefig(plot_dir / f"chi2_exceed_{view.id}_{unit}.pdf")
                    plt.close(fig)

        fig, axs = plt.subplots(nrows=2, sharex="all", sharey="all")
        bins = list(np.linspace(0, 6, 100)) + [1e9]
        for i, (u, bres) in enumerate(chi2_res.items()):
            for ax, (b, chi2_dist) in zip(axs, bres.items()):
                ax.hist(
                    chi2_dist,
                    density=True,
                    bins=bins,
                    cumulative=False,
                    alpha=0.5,
                    label=u,
                    color=f"C{i}",
                )
                ax.hist(
                    chi2_qso_res[u][b],
                    density=True,
                    bins=bins,
                    cumulative=False,
                    alpha=0.5,
                    histtype="step",
                    linestyle="dashed",
                    color=f"C{i}",
                )

        for i, ax in enumerate(axs):
            n1 = 10
            n2 = 8
            x = np.linspace(0, 6, 1000)
            ax.plot(x, stats.chi2(n1 * n2 - 1, 0, 1 / (n1 * n2 - 1)).pdf(x), color="C1")
            ax.plot(x, stats.chi2(n1 - 1, 0, 1 / (n1 - 1)).pdf(x), color="C0")
            ax.plot(x, stats.f(n1 - 1, n2, 0).pdf(x), color="C0")
            ax.set_xlim(0, 6)
            ax.set_ylabel(f"w{i + 1}")
            ax.legend()
            ax.set_ylim(0, 1.2)

        axs[-1].set_xlabel("Reduced Chi-Squared")
        fig.supylabel("Probability Density")

        fig.savefig(self.path)
        plt.close()
