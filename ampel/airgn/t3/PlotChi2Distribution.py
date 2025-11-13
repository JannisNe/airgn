from collections.abc import Generator
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from ampel.content.DataPoint import DataPoint
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
from timewise.process.stacking import FLUX_ZEROPOINTS


class PlotChi2Distribution(AbsPhotoT3Unit):
    """
    Plot lightcurves of transients using matplotlib
    """

    base_path: str
    chi2_stacked_std_names: list[str]

    input_mongodb_name: str | None = None
    chi2_range_to_plot: tuple[float, float] | None = None
    plot_dir: str = "./"
    upper_lim: float = 6
    cumulative: bool = False
    n1: float = 10
    n2: float = 8

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._base_path = Path(self.base_path)
        self._base_path.parent.mkdir(exist_ok=True, parents=True)
        self._plot_dir = Path(self.plot_dir)
        self._plot_dir.mkdir(exist_ok=True, parents=True)

        # method name of PDFs
        self._method_name = "pdf" if not self.cumulative else "cdf"

        columns = [
            "ra",
            "dec",
            "mjd",
        ]
        for i in range(1, 3):
            for key in [keys.MAG_EXT, keys.FLUX_EXT]:
                columns.extend([f"w{i}{key}", f"w{i}{keys.ERROR_EXT}{key}"])
        self._columns = columns

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

        # -----------------------------------------------------------
        # COLLECT UNIT RESULTS
        # -----------------------------------------------------------

        units = ["T2CalculateChi2Stacked", "T2CalculateChi2"]

        t2_reses = []
        for view in gen:
            row = {}
            for iunit, unit in enumerate(units):
                for t2 in view.get_t2_views(unit=unit):
                    t2res = t2.body[-1]
                    if unit == "T2CalculateChi2Stacked":
                        std_name = (
                            t2.config["t2_dependency"][0]["config"]["std_name"]
                            or "sdom-1"
                        )
                    else:
                        std_name = ""
                    for b in ["w1", "w2"]:
                        keys_to_check = (
                            [keys.FLUX_EXT]
                            if unit == "T2CalculateChi2"
                            else [keys.FLUX_DENSITY_EXT]
                        )
                        for lk in keys_to_check:
                            unit_key = f"{unit}_{lk}"
                            if "Stacked" in unit:
                                unit_key += f"_{std_name}"
                            if t2res and (k := f"red_chi2_{b}_{lk}") in t2res:
                                row[f"{b}_{unit_key}"] = t2res[k]

                    if any(
                        self.chi2_range_to_plot is not None
                        and t2res
                        and (k := f"red_chi2_{b}_{keys.FLUX_DENSITY_EXT}") in t2res
                        and (dps := view.get_photopoints())
                        and self.chi2_range_to_plot[0]
                        <= t2res[k]
                        <= self.chi2_range_to_plot[1]
                        for b in ["w1", "w2"]
                    ):
                        self.plot_single(
                            view=view,
                            t2res=t2res,
                            descriptor=f"{unit}{std_name}",
                            dps=dps,
                        )

                medians_body = view.get_t2_body("T2CalculateMedians")
                for b in ["w1", "w2"]:
                    row[f"{b}_T2CalculateMedians"] = medians_body[f"median_{b}_all"]

            t2_reses.append(row)

        df = pd.DataFrame(t2_reses)

        # -----------------------------------------------------------
        # PLOT FULL CHI2 DISTRIBUTION
        # -----------------------------------------------------------

        fig, axs = plt.subplots(nrows=2, sharex="all", sharey="all")
        bins = list(np.linspace(0, self.upper_lim, 100)) + [1e9]
        x = np.linspace(0, self.upper_lim, 1000)

        # set up result structure
        res = {}
        cols = [c for c in df.columns if "T2CalculateChi2" in c]
        chi2_units = {c.replace("w1_", "").replace("w2_", "") for c in cols}
        colors = {c: f"C{i}" for i, c in enumerate(chi2_units)}

        for i, u in enumerate(cols):
            b = u[:2]
            ax = axs[int(b[1]) - 1]
            chi2_dist = df[u]
            ax.hist(
                chi2_dist,
                density=True,
                bins=bins,
                cumulative=self.cumulative,
                alpha=0.5,
                label=u,
                color=colors[u[3:]],
            )
            chi2_dist = np.array(chi2_dist)
            m = ~np.isnan(chi2_dist) & np.isfinite(chi2_dist)
            ffit = stats.f.fit(chi2_dist[m], self.n1, self.n2, floc=0)
            key_base = f"{u}_{b}"
            for p, p_name in zip(ffit, ["d1", "d2", "loc", "scale"]):
                res[f"{key_base}_{p_name}"] = p

            ax.plot(
                x,
                stats.f(*ffit).__getattribute__(self._method_name)(x),
                color=colors[u[3:]],
                linestyle="dotted",
            )

        for i, ax in enumerate(axs):
            self.add_expected_dist(ax, x)
            ax.set_xlim(0, self.upper_lim)
            ax.set_ylabel(f"w{i + 1}")
            ax.set_ylim(0, 1.2)

        axs[0].legend(ncols=2, bbox_to_anchor=(0.5, 1.05), loc="lower center")
        axs[-1].set_xlabel("Reduced Chi-Squared")
        fig.supylabel("Probability Density")
        fig.tight_layout()
        fig.savefig(self._base_path.parent / (self._base_path.name + "_full_chi2.pdf"))
        plt.close()

        # -----------------------------------------------------------
        # PLOT CHI2 IN MAG BINS
        # -----------------------------------------------------------

        # convert flux densities to mag
        bins = np.arange(5, 18)
        for b in ["w1", "w2"]:
            df[f"{b}_median_mag"] = -2.5 * np.log10(
                df[f"{b}_T2CalculateMedians"] * 1e-3 / FLUX_ZEROPOINTS[b]
            )
            df[f"{b}_mag_bin"] = np.digitize(df[f"{b}_median_mag"], bins=bins)
        full_bins = np.unique(
            np.concatenate([df[f"{b}_mag_bin"] for b in ["w1", "w2"]])
        )

        for c in chi2_units:
            if "Stacked" not in c:
                continue

            w, h = plt.rcParams["figure.figsize"]
            factor = 0.7
            fig, axs = plt.subplots(
                figsize=(w * 2 * factor, h * len(full_bins) * factor),
                ncols=2,
                nrows=len(full_bins),
                sharex="all",
                sharey="all",
            )

            for bin_number in df[f"{b}_mag_bin"]:
                for ib in range(1, 3):
                    b = f"w{ib}"
                    vals = df[f"{b}_{c}"]
                    ax = axs[int(np.where(full_bins == bin_number)[0]), ib - 1]
                    m = (
                        (df[f"{b}_mag_bin"] == bin_number)
                        & np.isfinite(vals)
                        & vals.notna()
                    )
                    if any(m):
                        ax.hist(
                            vals[m],
                            density=True,
                            cumulative=self.cumulative,
                            alpha=0.5,
                            color=colors[c],
                        )
                    ax.annotate(
                        rf"{bins[bin_number - 1]} < {b} < {bins[bin_number]}",
                        (0.05, 0.95),
                        ha="left",
                        va="top",
                    )
                    ax.set_xlim(0, self.upper_lim)

            xlim = axs[0, 0].get_xlim()
            x = np.linspace(*xlim, 100)
            for ax in axs.flatten():
                self.add_expected_dist(ax, x)

            axs[0][0].legend(ncols=3, bbox_to_anchor=[1, 1.05], loc="lower center")
            fig.supxlabel("Reduced Chi-Squared")
            fig.supylabel("Probability Density")
            fig.tight_layout()
            fig.savefig(
                self._base_path.parent / (self._base_path.name + f"_{c}_chi2_bins.pdf")
            )
            plt.close()

        return res

    # -----------------------------------------------------------
    # PLOT HELPER FUNCTIONS
    # -----------------------------------------------------------

    def plot_single(
        self,
        view: TransientView,
        t2res: UBson,
        descriptor: str,
        dps: Sequence[DataPoint],
    ):
        raw_lightcurve = datapoints_to_dataframe(dps, self._columns)[0]
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
                stacked_lc[f"{b}{keys.FLUX_DENSITY_EXT}{keys.KSTEST_NORM_EXT}"],
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
        fig.suptitle(f"Transient {view.id} (unit: {descriptor})")
        fig.savefig(self._plot_dir / f"chi2_exceed_{view.id}_{descriptor}.pdf")
        plt.close(fig)

    def add_expected_dist(self, ax: plt.Axes, x: Sequence[float]):
        ax.plot(
            x,
            stats.chi2(
                self.n1 * self.n2 - 1, 0, 1 / (self.n1 * self.n2 - 1)
            ).__getattribute__(self._method_name)(x),
            color="k",
            ls="--",
            label=rf"$\chi^2_{{{self.n1 * self.n2 - 1:.0f}}}$",
        )
        ax.plot(
            x,
            stats.chi2(self.n1 - 1, 0, 1 / (self.n1 - 1)).__getattribute__(
                self._method_name
            )(x),
            color="k",
            ls=":",
            label=rf"$\chi^2_{{{self.n1 - 1:.0f}}}$",
        )
        ax.plot(
            x,
            stats.f(self.n1 - 1, self.n2 - 1).__getattribute__(self._method_name)(x),
            color="k",
            ls="-.",
            label=f"F({self.n1 - 1:.0f},{self.n2 - 1:.0f})",
        )
