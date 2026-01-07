from collections.abc import Generator
from itertools import pairwise

from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from ampel.abstract.AbsPhotoT3Unit import AbsPhotoT3Unit
from ampel.struct.T3Store import T3Store
from ampel.types import T3Send
from ampel.view.TransientView import TransientView

from timewise.util.path import expand
from airgn.desi.agn_value_added_catalog import get_agn_bitmask


def get_agn_desc(agn_bitmask, agn_mask) -> list[str]:
    mask = str(bin(int(agn_bitmask))).replace("0b", "")[::-1]
    return [am[0] for im, am in zip(mask, agn_mask["AGN_MASKBITS"]) if im]


class Chi2VsAGN(AbsPhotoT3Unit):
    """
    Plot lightcurves of transients using matplotlib
    """

    path: str
    input_mongo_db_name: str
    ylim: tuple[float, float]
    n_points_bins: tuple[int, ...] = (0, 10, 20, 30)
    mongo_uri: str = "mongodb://localhost:27017"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._client = MongoClient(self.mongo_uri)
        self._col = self._client[self.input_mongo_db_name]["input"]
        self._agn_bitmask = get_agn_bitmask()
        self._path = expand(self.path)
        self._path.mkdir(parents=True, exist_ok=True)

    def process(
        self, gen: Generator[TransientView, T3Send, None], t3s: None | T3Store = None
    ) -> None:
        res = {}
        for view in gen:
            input_res = None
            for t2 in view.get_t2_views("T2CalculateChi2Stacked"):
                input_res = self._col.find_one({"orig_id": t2.stock})
                break
            if not input_res:
                continue
            ires = dict(view.get_latest_t2_body("T2CalculateChi2Stacked"))
            ires.update(input_res)
            mask = str(bin(int(input_res["AGN_MASKBITS"]))).replace("0b", "")[::-1]
            ires["decoded_agn_mask"] = mask
            res[view.stock["stock"]] = ires

        res = pd.DataFrame.from_dict(res, orient="index")
        chi2_cols = [f"red_chi2_w{i + 1}_fluxdensity" for i in range(2)]
        npoints_cols = [f"npoints_w{i + 1}_fluxdensity" for i in range(2)]

        fig, ax = plt.subplots()
        ax.hist(res[npoints_cols].min(axis=1), ec="white", alpha=0.8)
        ax.set_xlabel("N")
        ax.set_ylabel("count")
        fn = self._path / "n_points_hist.png"
        self.logger.info(f"saving {fn}")
        fig.tight_layout()
        fig.savefig(fn)
        plt.close()

        for s, e in pairwise(self.n_points_bins):
            bin_mask = (res[npoints_cols] >= s).all(axis=1) & (
                res[npoints_cols] < e
            ).any(axis=1)
            res_bin = res[bin_mask]

            both_chi2 = res_bin[chi2_cols].notna().all(axis=1)
            n = [((res_bin["decoded_agn_mask"] == "0") & both_chi2).sum()]
            for ix in self._agn_bitmask["AGN_MASKBITS"]:
                ixb = res_bin["decoded_agn_mask"].str[ix[1]]
                ixm = both_chi2 & ixb.notna() & ixb.astype(float).astype(bool)
                n.append(ixm.sum())

            labels = ["no AGN"] + [ix[0] for ix in self._agn_bitmask["AGN_MASKBITS"]]

            fig, axs = plt.subplots(nrows=3, sharex="all")
            axs[0].bar(np.arange(-1, len(labels) - 1), n, alpha=0.5, ec="none")
            axs[0].set_yscale("log")

            for i, ax in enumerate(axs[1:]):
                m = res_bin[f"red_chi2_w{i + 1}_fluxdensity"].notna()
                x = []
                y = []
                for ix in self._agn_bitmask["AGN_MASKBITS"]:
                    ixb = res_bin["decoded_agn_mask"].str[ix[1]]
                    ixm = m & ixb.notna() & ixb.astype(float).astype(bool)
                    if any(ixm):
                        y.append(
                            np.log10(
                                res_bin.loc[
                                    ixm,
                                    f"red_chi2_w{i + 1}_fluxdensity",
                                ].values.tolist()
                            )
                        )
                        x.append(
                            ix[1] if ix[1] < 10 else ix[1] - 2
                        )  # bits 8 and 9 are skipped
                not_agn_mask = m & (res_bin["decoded_agn_mask"] == "0")
                x.append(-1)
                y.append(
                    np.log10(
                        res_bin.loc[
                            not_agn_mask, f"red_chi2_w{i + 1}_fluxdensity"
                        ].values.tolist()
                    )
                )
                ax.violinplot(
                    dataset=y, positions=x, showextrema=False, showmedians=True
                )
                ax.set_ylabel(f"W{i + 1}")

            fig.supylabel(r"$\chi^2_\mathrm{red}$")
            axs[-1].set_xticks(np.arange(-1, len(labels) - 1))
            axs[-1].set_xticklabels(labels, rotation=60, ha="right")

            fn = self._path / f"bin_{s}_{e}.png"
            self.logger.info(f"saving {fn}")
            fig.tight_layout()
            fig.savefig(fn)
            plt.close()
