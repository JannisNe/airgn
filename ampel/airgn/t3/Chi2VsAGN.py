from collections.abc import Generator

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
    mongo_uri: str = "mongodb://localhost:27017"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._client = MongoClient(self.mongo_uri)
        self._col = self._client[self.input_mongo_db_name]["input"]
        self._agn_bitmask = get_agn_bitmask()
        self._path = expand(self.path).with_suffix(".pdf")
        self._path.parent.mkdir(parents=True, exist_ok=True)

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

        fig, axs = plt.subplots(nrows=2, sharex="all")

        for i, ax in enumerate(axs):
            x = [ix[1] for ix in self._agn_bitmask["AGN_MASKBITS"]]
            y = [
                res.loc[
                    res["decoded_agn_mask"].str[ix[1]].astype(bool),
                    f"red_chi2_w{i + 1}_fluxdensity",
                ]
                for ix in self._agn_bitmask["AGN_MASKBITS"]
            ]
            ax.violinplot(
                y,
                x,
            )
            ax.set_ylabel(f"W{i + 1}")

        fig.supylabel(r"$\chi^2_\mathrm{red}$")
        axs[-1].tick_params(axis="x", labelrotation=60)
        labels = [ix[0] for ix in self._agn_bitmask["AGN_MASKBITS"]]
        axs[-1].set_xticks(np.arange(1, len(labels) + 1), labels=labels)

        self.logger.info(f"saving {self._path}")
        fig.savefig(self._path)
        plt.close()
