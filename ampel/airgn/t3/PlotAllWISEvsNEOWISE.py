from collections.abc import Generator

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from ampel.abstract.AbsPhotoT3Unit import AbsPhotoT3Unit
from ampel.struct.T3Store import T3Store
from ampel.types import T3Send
from ampel.view.TransientView import TransientView
from ampel.timewise.util.pdutil import datapoints_to_dataframe

from timewise.plot.lightcurve import plot_lightcurve
from timewise.process import keys
from timewise.util.path import expand


def fd2w1mag(flux_density: float) -> float:
    """Convert flux density in microJy to magnitude"""
    return -2.5 * np.log10(flux_density / 309540.0)


def w1mag2fd(mag: float) -> float:
    """Convert magnitude to flux density in microJy"""
    return 309540.0 * 10 ** (-mag / 2.5)


def fd2w2mag(flux_density: float) -> float:
    """Convert flux density in microJy to magnitude"""
    return -2.5 * np.log10(flux_density / 171787.0)


def w2mag2fd(mag: float) -> float:
    """Convert magnitude to flux density in microJy"""
    return 171787.0 * 10 ** (-mag / 2.5)


class PlotAllWISEvsNEOWISE(AbsPhotoT3Unit):
    """
    Plot lightcurves of transients using matplotlib
    """

    path: str
    thresh_to_plot: float | None = None
    plot_dir: str = "./"

    def process(
        self, gen: Generator[TransientView, T3Send, None], t3s: None | T3Store = None
    ) -> None:
        plot_dir = expand(self.plot_dir)
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
            for t2 in view.get_t2_views(unit="T2CalculateMedians"):
                if t2 is None or t2.body is None:
                    continue

                t2res = dict(t2.body[-1])
                if t2res is None:
                    continue

                t2dep = t2.config["t2_dependency"][0]["unit"]
                t2res["t2dep"] = t2dep
                res.append(t2res)

                if any(
                    self.thresh_to_plot is not None
                    and t2res
                    and (k := f"allwise_neowise_ratio_w{b}") in t2res
                    and t2res[k] > self.thresh_to_plot
                    for b in ["w1", "w2"]
                ):
                    raw_lightcurve = datapoints_to_dataframe(
                        view.get_photopoints(), columns
                    )[0]
                    stacked_lc = pd.DataFrame(
                        view.get_t2_body(unit=t2dep, ret_type=tuple),
                    )
                    fig, ax = plot_lightcurve(
                        lum_key=keys.FLUX_EXT,
                        stacked_lightcurve=stacked_lc,
                        raw_lightcurve=raw_lightcurve,
                    )
                    fig.suptitle(f"Transient {view.id}")
                    fig.savefig(plot_dir / f"chi2_exceed_{view.id}_{t2dep}.pdf")
                    plt.close(fig)

        res = pd.DataFrame(res)

        fig, axs = plt.subplots(nrows=2, sharex="all", sharey="all")
        for j, t2dep in enumerate(res["t2dep"].unique()):
            m = res["t2dep"] == t2dep
            for i in range(1, 3):
                ax = axs[i - 1]
                x = res.loc[m, f"median_w{i}_allwise"]
                y = res.loc[m, f"allwise_neowise_ratio_w{i}"]
                ax.scatter(
                    x,
                    y,
                    alpha=0.1,
                    s=1,
                    c=f"C{j}",
                )

                # Define 50 equal-width bins
                bins = pd.cut(np.log10(x), bins=50)
                bm = 10**bins.cat.categories.mid
                quantiles = (
                    y.groupby(bins, observed=False)
                    .quantile([0.05, 0.5, 0.95])
                    .unstack()
                )
                ax.plot(
                    bm,
                    quantiles[0.5],
                    ls="-",
                    color=f"C{j}",
                    label=t2dep,
                )
                ax.plot(bm, quantiles[0.05], ls=":", color=f"C{j}")
                ax.plot(bm, quantiles[0.95], ls=":", color=f"C{j}")

        for i in range(1, 3):
            ax = axs[i - 1]
            ax.set_ylabel(f"w{i}")
            ax.axhline(1.0, color="k", ls="--")
            ylim = ax.get_ylim()
            ax.set_ylim(max(0.2, ylim[0]), min(5.0, ylim[1]))

        axs[-1].set_xscale("log")
        axs[-1].set_yscale("log")
        axs[0].legend(frameon=False, loc="upper left")
        mag_ticks = [6, 8, 10, 12, 14, 16]
        secax1 = axs[0].secondary_xaxis("top", functions=(fd2w1mag, w1mag2fd))
        secax1.set_xlabel("Apparent Magnitude")
        secax1.set_xticks(mag_ticks, labels=mag_ticks)
        secax2 = axs[1].secondary_xaxis("top", functions=(fd2w2mag, w2mag2fd))
        secax2.set_xticks(mag_ticks, labels=mag_ticks)
        axs[-1].set_xlabel("Median AllWISE Flux Density")
        fig.supylabel("AllWISE / NEOWISE Flux Density Ratio")

        fig.savefig(expand(self.path))
        plt.close()
