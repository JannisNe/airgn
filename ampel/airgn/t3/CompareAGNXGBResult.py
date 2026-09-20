from typing import Generator

import numpy as np
import pandas as pd
from ampel.abstract.AbsPhotoT3Unit import AbsPhotoT3Unit
from ampel.abstract.AbsT3Unit import T
from ampel.struct.T3Store import T3Store
from ampel.struct.UnitResult import UnitResult
from ampel.types import T3Send, UBson
from matplotlib import pyplot as plt
from timewise.util.path import expand


class CompareAGNXGBResult(AbsPhotoT3Unit):
    prepended_keys: tuple[str, str]
    labels: tuple[str, str]
    filename_base: str
    mplstyle: str | None = None

    def process(
        self, gen: Generator[T, T3Send, None], t3s: T3Store
    ) -> UBson | UnitResult:
        data = pd.DataFrame.from_dict(
            {view.stock["stock"]: view.extra if view.extra else {} for view in gen},
            orient="index",
        ).dropna(how="any", axis="index")

        ppk = self.prepended_keys
        for k in ["agn", "wise_agn"]:
            assert all(data[f"{ppk[0]}_{k}"] == data[f"{ppk[0]}_{k}"])

        non_agn_mask = ~(data[f"{ppk[0]}_agn"].astype(bool))
        wise_agn_mask = data[f"{ppk[0]}_wise_agn"].astype(bool)
        non_wise_agn_mask = ~non_agn_mask & ~wise_agn_mask
        masks = [non_agn_mask, wise_agn_mask, non_wise_agn_mask]
        labels = ["non AGN", "WISE AGN", "non-WISE AGN"]

        if self.mplstyle is not None:
            plt.style.use(self.mplstyle)

        fs = plt.rcParams["figure.figsize"]
        fig, axs = plt.subplots(
            ncols=3, figsize=(fs[0] * 2, fs[1]), sharex=True, sharey=True
        )
        xx = np.linspace(0, 1, 10)
        for m, ax, title in zip(masks, axs, labels):
            ax.scatter(
                data.loc[m, f"{ppk[0]}_probability"],
                data.loc[m, f"{ppk[1]}_probability"],
                zorder=10,
                s=1,
                alpha=0.1,
            )
            ax.plot(xx, xx, ls=":", zorder=1, alpha=0.3)
            ax.set_title(title)

        fig.supxlabel(self.labels[0])
        fig.supylabel(self.labels[1])

        fn_base = expand(self.filename_base)
        fn_base.parent.mkdir(parents=True, exist_ok=True)
        fn = str(fn_base) + "_scatter.png"
        fig.tight_layout()
        fig.savefig(fn)
        plt.close()

        fig, axs = plt.subplots(
            ncols=3, figsize=(fs[0] * 2, fs[1]), sharex=True, sharey=True
        )
        for m, ax, title in zip(masks, axs, labels):
            ratio = (
                data.loc[m, f"{ppk[0]}_probability"]
                / data.loc[m, f"{ppk[1]}_probability"]
            )
            ax.hist(np.log10(ratio), ec="white", alpha=0.8, bins=20, density=True)
            ax.set_title(title)
        fig.supylabel("density")
        fig.supxlabel(f"{self.labels[0]} / {self.labels[1]}")
        fn = str(fn_base) + "_hist.pdf"
        fig.tight_layout()
        fig.savefig(fn)
        plt.close()
