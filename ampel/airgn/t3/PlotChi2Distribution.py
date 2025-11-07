from collections.abc import Generator
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from pymongo import MongoClient
import pandas as pd

from ampel.abstract.AbsPhotoT3Unit import AbsPhotoT3Unit
from ampel.struct.T3Store import T3Store
from ampel.view.TransientView import TransientView
from ampel.timewise.util.pdutil import datapoints_to_dataframe
from ampel.struct.UnitResult import UnitResult
from ampel.types import T3Send, UBson

from timewise.plot.lightcurve import plot_lightcurve, BAND_PLOT_COLORS
from timewise.process import keys


class PlotChi2Distribution(AbsPhotoT3Unit):
    """
    Plot lightcurves of transients using matplotlib
    """

    path: str

    input_mongodb_name: str | None = None
    chi2_range_to_plot: tuple[float, float] | None = None
    plot_dir: str = "./"
    upper_lim: float = 6
    cumulative: bool = False
    n1: float = 10
    n2: float = 8

    def process(
        self, gen: Generator[TransientView, T3Send, None], t3s: None | T3Store = None
    ) -> UBson | UnitResult:
        if self.input_mongodb_name:
            input_collection = MongoClient()[self.input_mongodb_name]["input"]
            qso_ids = [
                doc["orig_id"]
                for doc in input_collection.find({"objtype": "b'QSO'"}, {"orig_id": 1})
            ]
        else:
            qso_ids = []

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
        unit_keys = [
            f"T2CalculateChi2Stacked_{keys.FLUX_EXT}",
            f"T2CalculateChi2Stacked_{keys.FLUX_DENSITY_EXT}",
            f"T2CalculateChi2_{keys.FLUX_EXT}",
        ]
        chi2_res = {u: {"w1": [], "w2": []} for u in unit_keys}
        chi2_qso_res = {u: {"w1": [], "w2": []} for u in unit_keys}
        for view in gen:
            for unit in units:
                t2res = view.get_t2_body(unit=unit)
                for b in ["w1", "w2"]:
                    keys_to_check = [keys.FLUX_EXT]
                    if unit == "T2CalculateChi2Stacked":
                        keys_to_check.append(keys.FLUX_DENSITY_EXT)
                    for lk in keys_to_check:
                        unit_key = f"{unit}_{lk}"
                        if t2res and (k := f"red_chi2_{b}_{lk}") in t2res:
                            if view.id in qso_ids:
                                chi2_qso_res[unit_key][b].append(t2res[k])
                            else:
                                chi2_res[unit_key][b].append(t2res[k])

                if any(
                    self.chi2_range_to_plot is not None
                    and t2res
                    and (k := f"red_chi2_{b}_{keys.FLUX_DENSITY_EXT}") in t2res
                    and self.chi2_range_to_plot[0]
                    <= t2res[k]
                    <= self.chi2_range_to_plot[1]
                    for b in ["w1", "w2"]
                ):
                    raw_lightcurve = datapoints_to_dataframe(
                        view.get_photopoints(), columns
                    )[0]
                    stacked_lc = pd.DataFrame(
                        view.get_t2_body(unit="T2StackVisits", ret_type=tuple),
                    )
                    chi2_text = "\n".join(
                        r"$\chi^2_{{red,{b}}} = {chi2:.2f}$".format(
                            b=b, chi2=t2res[f"red_chi2_{b}_{keys.FLUX_DENSITY_EXT}"]
                        )
                        for b in ["w1", "w2"]
                        if t2res and (f"red_chi2_{b}" in t2res)
                    )
                    fig, axs = plt.subplots(nrows=3, sharex="all")
                    plot_lightcurve(
                        lum_key=keys.FLUX_EXT,
                        stacked_lightcurve=stacked_lc,
                        raw_lightcurve=raw_lightcurve,
                        ax=axs[0],
                    )
                    plot_lightcurve(
                        lum_key=keys.FLUX_DENSITY_EXT,
                        stacked_lightcurve=stacked_lc,
                        ax=axs[1],
                    )
                    axs[2].annotate(
                        chi2_text,
                        xy=(0.05, 0.05),
                        xycoords="axes fraction",
                        fontsize=10,
                        verticalalignment="bottom",
                        horizontalalignment="left",
                    )
                    for b in ["w1", "w2"]:
                        axs[2].plot(
                            stacked_lc["mean_mjd"],
                            stacked_lc[
                                f"{b}{keys.FLUX_DENSITY_EXT}{keys.KSTEST_NORM_EXT}"
                            ],
                            "o-",
                            color=BAND_PLOT_COLORS[b],
                        )
                    axs[2].axhline(0.05, color="red", linestyle="dashed")
                    axs[2].set_ylim(1e-3, 1)
                    axs[2].set_ylabel("K-S Test p-value")
                    axs[2].set_xlabel("MJD")
                    axs[0].set_xlabel("")
                    axs[1].set_xlabel("")
                    for ax in axs:
                        ax.set_yscale("log")
                    fig.suptitle(f"Transient {view.id} (unit: {unit})")
                    fig.savefig(plot_dir / f"chi2_exceed_{view.id}_{unit}.pdf")
                    plt.close(fig)

        fig, axs = plt.subplots(nrows=2, sharex="all", sharey="all")
        bins = list(np.linspace(0, self.upper_lim, 100)) + [1e9]
        x = np.linspace(0, self.upper_lim, 1000)

        # method name of PDFs
        method_name = "pdf" if not self.cumulative else "cdf"

        # set up result structure
        res = {}

        for i, (u, bres) in enumerate(chi2_res.items()):
            for ax, (b, chi2_dist) in zip(axs, bres.items()):
                ax.hist(
                    chi2_dist,
                    density=True,
                    bins=bins,
                    cumulative=self.cumulative,
                    alpha=0.5,
                    label=u,
                    color=f"C{i}",
                )
                chi2_dist = np.array(chi2_dist)
                m = ~np.isnan(chi2_dist) & np.isfinite(chi2_dist)
                ffit = stats.f.fit(chi2_dist[m], self.n1, self.n2, floc=0)
                key_base = f"{u}_{b}"
                for p, p_name in zip(ffit, ["d1", "d2", "loc", "scale"]):
                    res[f"{key_base}_{p_name}"] = p

                ax.plot(
                    x,
                    stats.f(*ffit).__getattribute__(method_name)(x),
                    color=f"C{i}",
                    linestyle="dotted",
                )
                ax.hist(
                    chi2_qso_res[u][b],
                    density=True,
                    bins=bins,
                    cumulative=self.cumulative,
                    alpha=0.5,
                    histtype="step",
                    linestyle="dashed",
                    color=f"C{i}",
                )

        for i, ax in enumerate(axs):
            ax.plot(
                x,
                stats.chi2(
                    self.n1 * self.n2 - 1, 0, 1 / (self.n1 * self.n2 - 1)
                ).__getattribute__(method_name)(x),
                color="C1",
            )
            ax.plot(
                x,
                stats.chi2(self.n1 - 1, 0, 1 / (self.n1 - 1)).__getattribute__(
                    method_name
                )(x),
                color="C0",
            )
            ax.plot(
                x,
                stats.f(self.n1 - 1, self.n2 - 1).__getattribute__(method_name)(x),
                color="C2",
            )
            ax.set_xlim(0, self.upper_lim)
            ax.set_ylabel(f"w{i + 1}")
            ax.legend()
            ax.set_ylim(0, 1.2)

        axs[-1].set_xlabel("Reduced Chi-Squared")
        fig.supylabel("Probability Density")

        fig.savefig(self.path)
        plt.close()

        return res
