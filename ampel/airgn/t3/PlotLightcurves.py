from collections.abc import Generator

import pandas as pd
from ampel.abstract.AbsPhotoT3Unit import AbsPhotoT3Unit
from ampel.struct.T3Store import T3Store
from ampel.struct.UnitResult import UnitResult
from ampel.timewise.util.pdutil import datapoints_to_dataframe
from ampel.types import T3Send, UBson
from ampel.view.TransientView import TransientView
from matplotlib import pyplot as plt
from timewise.plot import plot_lightcurve
from timewise.plot.lightcurve import BAND_PLOT_COLORS
from timewise.process import keys
from timewise.util.path import expand


class PlotLightcurves(AbsPhotoT3Unit):
    """
    Plot lightcurves of transients using matplotlib
    """

    base_dir: str

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        columns = [
            "ra",
            "dec",
            "mjd",
        ]
        for i in range(1, 3):
            for key in [keys.MAG_EXT, keys.FLUX_EXT]:
                columns.extend([f"w{i}{key}", f"w{i}{keys.ERROR_EXT}{key}"])
        self._columns = columns

        self._base_dir = expand(self.base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def process(
        self, gen: Generator[TransientView, T3Send, None], t3s: None | T3Store = None
    ) -> UBson | UnitResult:
        for view in gen:
            dps = view.get_photopoints()
            assert dps is not None

            raw_lightcurve = datapoints_to_dataframe(dps, self._columns)[0]
            stock = view.stock
            assert stock is not None
            stock_id = stock["stock"]

            if tw_view := view.get_t2_body(unit="T2StackVisits", ret_type=tuple):
                stacked_lc = pd.DataFrame(tw_view)
                fig, ax = plot_lightcurve(
                    lum_key=keys.FLUX_EXT,
                    stacked_lightcurve=stacked_lc,
                    raw_lightcurve=raw_lightcurve,
                )
                fig.tight_layout()
                fig.savefig(f"{self._base_dir}/{stock_id}_tw.pdf")
                plt.close(fig)

            if ls_view := view.get_t2_body(unit="T2MaggyToFluxDensity", ret_type=tuple):
                fig, ax = plt.subplots()
                for b in ["w1", "w2"]:
                    ax.errorbar(
                        ls_view[f"LC_MJD_{b.upper()}"],
                        ls_view[f"{b}{keys.MEAN}{keys.FLUX_DENSITY_EXT}"],
                        yerr=ls_view[f"{b}{keys.FLUX_DENSITY_EXT}{keys.RMS}"],
                        label=f"{b} Legacy Survey",
                        ls="",
                        marker="s",
                        c=BAND_PLOT_COLORS[b],
                        markersize=4,
                        markeredgecolor="none",
                        ecolor=BAND_PLOT_COLORS[b],
                        capsize=2,
                        zorder=3,
                        barsabove=True,
                        elinewidth=0.5,
                    )

                ax.set_ylabel("Flux Density (mJy)")
                ax.set_xlabel("MJD")
                ax.legend()
                fig.savefig(f"{self._base_dir}/{stock_id}_ls.pdf")
                plt.close(fig)
