from collections.abc import Generator
from typing import Literal, Self

import pandas as pd
from ampel.abstract.AbsPhotoT3Unit import AbsPhotoT3Unit
from ampel.struct.T3Store import T3Store
from ampel.struct.UnitResult import UnitResult
from ampel.timewise.util.pdutil import datapoints_to_dataframe
from ampel.types import T3Send, UBson
from ampel.view.TransientView import TransientView
from matplotlib import pyplot as plt
from pydantic import model_validator
from timewise.plot import plot_lightcurve
from timewise.process import keys
from timewise.util.path import expand


ylabels = {
    keys.MAG_EXT: "Mag",
    keys.FLUX_EXT: "Flux",
    keys.FLUX_DENSITY_EXT: "Flux Density [mJy]",
}


class PlotLightcurves(AbsPhotoT3Unit):
    """
    Plot lightcurves of transients using matplotlib
    """

    base_dir: str
    filename_extra_keys: list[str] | None = None

    timewise_raw: bool = False
    timewise_key: Literal["flux", "mpro", "fluxdensity"] = keys.FLUX_DENSITY_EXT

    w1_color: str = "dodgerblue"
    w1_marker: str = "o"
    w2_color: str = "crimson"
    w2_marker: str = "s"

    mplstyle: str | None = None

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
        self._timewise_raw_columns = columns

        self._base_dir = expand(self.base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

        self._colors = {"w1": self.w1_color, "w2": self.w2_color}
        self._markers = {"w1": self.w1_marker, "w2": self.w2_marker}

        if self.mplstyle:
            plt.style.use(self.mplstyle)

    @model_validator(mode="after")
    def check_timewise_key(self) -> Self:
        if self.timewise_raw and (self.timewise_key == keys.FLUX_DENSITY_EXT):
            raise ValueError(
                f"Raw lightcurve can not be shown for {keys.FLUX_DENSITY_EXT}!"
            )
        return self

    def process(
        self, gen: Generator[TransientView, T3Send, None], t3s: None | T3Store = None
    ) -> UBson | UnitResult:
        for view in gen:
            dps = view.get_photopoints()
            assert dps is not None

            stock = view.stock
            assert stock is not None
            stock_id = stock["stock"]

            errorbar_kw = dict(
                ms=5,
                ls="",
                capsize=1,
                capthick=0.5,
                barsabove=True,
                ecolor="k",
                elinewidth=0.5,
            )

            filename_extension = ""
            if self.filename_extra_keys is not None:
                assert view.extra is not None, "Extra keys missing!"
                for k in self.filename_extra_keys:
                    filename_extension += f"_{k}{view.extra[k]}"

            if tw_view := view.get_t2_body(unit="T2StackVisits", ret_type=tuple):
                stacked_lc = pd.DataFrame(tw_view)
                raw_lightcurve = (
                    datapoints_to_dataframe(dps, self._timewise_raw_columns)[0]
                    if self.timewise_raw
                    else None
                )

                fig, ax = plot_lightcurve(
                    lum_key=self.timewise_key,
                    stacked_lightcurve=stacked_lc,
                    raw_lightcurve=raw_lightcurve,
                    colors=self._colors,
                )
                ax.set_ylabel(ylabels[self.timewise_key])
                fig.tight_layout()
                fig.savefig(f"{self._base_dir}/{stock_id}_tw{filename_extension}.pdf")
                plt.close(fig)

            if ls_view := view.get_t2_body(unit="T2MaggyToFluxDensity", ret_type=tuple):
                fig, ax = plt.subplots()
                ls_lc = pd.DataFrame(ls_view)
                for b in ["w1", "w2"]:
                    ax.errorbar(
                        ls_lc[f"LC_MJD_{b.upper()}"],
                        ls_lc[f"{b}{keys.MEAN}{keys.FLUX_DENSITY_EXT}"],
                        yerr=ls_lc[f"{b}{keys.FLUX_DENSITY_EXT}{keys.RMS}"],
                        label=f"{b}",
                        marker=self._markers[b],
                        c=self._colors[b],
                        markeredgecolor="none",
                        zorder=3,
                        **errorbar_kw,
                    )

                ax.set_ylabel(ylabels[keys.FLUX_DENSITY_EXT])
                ax.set_xlabel("MJD")
                ax.legend()
                fig.tight_layout()
                fig.savefig(f"{self._base_dir}/{stock_id}_ls{filename_extension}.pdf")
                plt.close(fig)
