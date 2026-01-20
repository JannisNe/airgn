from collections.abc import Generator
from itertools import pairwise
from typing import TypedDict

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
from ampel.airgn.t2.T2CalculateVarMetrics import T2CalculateVarMetrics, MetricOptions


# AB offset to Vega for WISE bands from Jarrett et al. (2011)
# https://dx.doi.org/10.1088/0004-637X/735/2/112
WISE_AB_OFFSET = {
    "W1": 2.699,
    "W2": 3.339,
}


def get_agn_desc(agn_bitmask, agn_mask) -> list[str]:
    mask = str(bin(int(agn_bitmask))).replace("0b", "")[::-1]
    return [am[0] for im, am in zip(mask, agn_mask["AGN_MASKBITS"]) if im]


class VarMetricsVsAGN(AbsPhotoT3Unit):
    """
    Plot lightcurves of transients using matplotlib
    """

    path: str
    input_mongo_db_name: str
    n_points_bins: tuple[int, ...] = (0, 10, 20, 30)
    mongo_uri: str = "mongodb://localhost:27017"
    iter_max: int | None = None
    file_format: str = "pdf"
    metric_names: list[str] = list(T2CalculateVarMetrics._metrics.keys())
    metric_options: dict[str, MetricOptions] = {}

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._client = MongoClient(self.mongo_uri)
        self._col = self._client[self.input_mongo_db_name]["input"]
        self._agn_bitmask = get_agn_bitmask()
        self._path = expand(self.path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._metric_options = {
            mn: self.metric_options.get(mn, T2CalculateVarMetrics._metric_options[mn])
            for mn in self.metric_names
        }

    def iter_npoints_binned(
        self, df: pd.DataFrame
    ) -> Generator[tuple[pd.DataFrame, float, float], None, None]:
        npoints_cols = [f"npoints_w{i + 1}_fluxdensity" for i in range(2)]
        bins = list(pairwise(self.n_points_bins)) + [
            (min(self.n_points_bins), max(self.n_points_bins))
        ]
        for s, e in bins:
            bin_mask = (df[npoints_cols] >= s).all(axis=1) & (df[npoints_cols] < e).any(
                axis=1
            )
            yield df[bin_mask], s, e

    def process(
        self, gen: Generator[TransientView, T3Send, None], t3s: None | T3Store = None
    ) -> None:
        # ---------------------- aggregate results ---------------------- #

        res = {}
        n_iter = 0
        for view in gen:
            input_res = None
            for t2 in view.get_t2_views("T2CalculateVarMetrics"):
                input_res = self._col.find_one({"orig_id": t2.stock})
                break
            if not input_res:
                continue

            latest_body = view.get_latest_t2_body("T2CalculateVarMetrics")
            if not latest_body:
                continue
            ires = dict(latest_body)
            ires.update(input_res)
            mask = str(bin(int(input_res["AGN_MASKBITS"]))).replace("0b", "")[::-1]
            ires["decoded_agn_mask"] = mask
            res[view.stock["stock"]] = ires

            n_iter += 1
            if self.iter_max and n_iter >= self.iter_max:
                self.logger.info("iteration limit reached, stopping loop")
                break

        res = pd.DataFrame.from_dict(res, orient="index")

        # ---------------------- histograms ---------------------- #

        for metric_name in self.metric_names:
            log = self._metric_options[metric_name]["log"]
            lim = self._metric_options[metric_name]["range"]
            pn = self._metric_options[metric_name]["pretty_name"]
            pl = r"$\log_{10}($" + pn + "$)$" if log else pn
            pdir = self._path / metric_name
            pdir.mkdir(parents=True, exist_ok=True)

            cols = [f"{metric_name}_w{i}_fluxdensity" for i in range(1, 3)]
            fig, ax = plt.subplots()
            vals = res[cols].min(axis=1)
            if log:
                m = vals > 0
                vals = np.log10(vals[m])
            ax.hist(vals, ec="white", alpha=0.8)
            ax.set_xlabel(pl)
            ax.set_ylabel("counts")
            fn = self._path / f"{metric_name}_hist.{self.file_format}"
            self.logger.info(f"saving {fn}")
            fig.tight_layout()
            fig.savefig(fn)
            plt.close()

            # ---------------------- violinplots ---------------------- #

            for res_bin, s, e in self.iter_npoints_binned(res):
                both_bands_mask = res_bin[cols].notna().all(axis=1)
                n = [((res_bin["decoded_agn_mask"] == "0") & both_bands_mask).sum()]
                for ix in self._agn_bitmask["AGN_MASKBITS"]:
                    ixb = res_bin["decoded_agn_mask"].str[ix[1]]
                    ixm = both_bands_mask & ixb.notna() & ixb.astype(float).astype(bool)
                    n.append(ixm.sum())

                labels = ["no AGN"] + [
                    ix[0] for ix in self._agn_bitmask["AGN_MASKBITS"]
                ]

                fig, axs = plt.subplots(nrows=3, sharex="all")
                axs[0].bar(np.arange(-1, len(labels) - 1), n, alpha=0.5, ec="none")
                axs[0].set_yscale("log")
                axs[0].set_ylabel("counts")

                for i, ax in enumerate(axs[1:]):
                    m = res_bin[f"{metric_name}_w{i + 1}_fluxdensity"].notna()
                    x = []
                    y = []
                    for ix in self._agn_bitmask["AGN_MASKBITS"]:
                        ixb = res_bin["decoded_agn_mask"].str[ix[1]]
                        ixm = m & ixb.notna() & ixb.astype(float).astype(bool)
                        if any(ixm):
                            vals = res_bin.loc[
                                ixm,
                                f"{metric_name}_w{i + 1}_fluxdensity",
                            ].values.tolist()
                            y.append(np.log10(vals) if log else vals)
                            x.append(
                                ix[1] if ix[1] < 10 else ix[1] - 2
                            )  # bits 8 and 9 are skipped
                    not_agn_mask = m & (res_bin["decoded_agn_mask"] == "0")
                    x.append(-1)
                    not_agn_vals = np.log10(
                        res_bin.loc[
                            not_agn_mask, f"{metric_name}_w{i + 1}_fluxdensity"
                        ].values.tolist()
                    )
                    y.append(np.log10(not_agn_vals) if log else not_agn_vals)
                    ax.violinplot(
                        dataset=y, positions=x, showextrema=False, showmedians=True
                    )
                    ax.set_ylabel(f"W{i + 1}")
                    ax.set_ylim(*lim)

                fig.supylabel(pl)
                axs[-1].set_xticks(np.arange(-1, len(labels) - 1))
                axs[-1].set_xticklabels(labels, rotation=60, ha="right")

                fn = pdir / f"bin_{s}_{e}.{self.file_format}"
                self.logger.info(f"saving {fn}")
                fig.tight_layout()
                fig.savefig(fn)
                plt.close()

            # ---------------------- metric vs color ---------------------- #

            metric_threshs = np.linspace(*lim, 100)
            res["agn"] = ~(res["decoded_agn_mask"] == "0")
            wise_agn_bit = res["decoded_agn_mask"].str[15]
            wise_agn_mask = wise_agn_bit.notna() & wise_agn_bit.astype(float).astype(
                bool
            )
            res["wise_agn"] = wise_agn_mask
            res["non_wise_agn"] = res["agn"] & ~wise_agn_mask

            for res_bin, s, e in self.iter_npoints_binned(res):
                for ix in self._agn_bitmask["AGN_MASKBITS"]:
                    ixb = res_bin["decoded_agn_mask"].str[ix[1]]
                    type_mask = ixb.notna() & ixb.astype(float).astype(bool)
                    type_res_bin = res_bin[type_mask]

                    n_agn = type_res_bin["agn"].sum()
                    n_wise_agn = type_res_bin["wise_agn"].sum()
                    n_non_wise_agn = type_res_bin["non_wise_agn"].sum()

                    completeness = []
                    purity = []
                    percentage_wise_agn = []
                    percentage_non_wise_agn = []

                    for thresh in metric_threshs:
                        n_selected_agn = (
                            (type_res_bin.loc[type_res_bin["agn"], cols] > thresh)
                            .all(axis=1)
                            .sum()
                        )
                        n_selected_non_agn = (
                            (res_bin.loc[~res_bin["agn"], cols] > thresh)
                            .all(axis=1)
                            .sum()
                        )
                        completeness.append(n_selected_agn / n_agn)
                        purity.append(
                            n_selected_agn / (n_selected_agn + n_selected_non_agn)
                        )

                        n_selected_wise_agn = (
                            (type_res_bin.loc[type_res_bin["wise_agn"], cols] > thresh)
                            .all(axis=1)
                            .sum()
                        )
                        percentage_wise_agn.append(n_selected_wise_agn / n_wise_agn)
                        n_selected_non_wise_agn = (
                            (
                                type_res_bin.loc[type_res_bin["non_wise_agn"], cols]
                                > thresh
                            )
                            .all(axis=1)
                            .sum()
                        )
                        percentage_non_wise_agn.append(
                            n_selected_non_wise_agn / n_non_wise_agn
                        )

                    fig, ax = plt.subplots()
                    ax.plot(
                        metric_threshs,
                        completeness,
                        label="completeness",
                        lw=2,
                        color="C0",
                    )
                    ax.plot(
                        metric_threshs,
                        purity,
                        label="purity",
                        lw=2,
                        ls="--",
                        color="C0",
                    )
                    ax.plot(
                        metric_threshs,
                        percentage_wise_agn,
                        label="WISE AGN",
                        lw=2,
                        ls=":",
                        color="C1",
                    )
                    ax.plot(
                        metric_threshs,
                        percentage_non_wise_agn,
                        label="Non WISE AGN",
                        lw=2,
                        ls="-.",
                        color="C1",
                    )
                    pl_thresh = (
                        r"$\log_{10}($" + pn + r"$\,_\mathrm{thresh})$"
                        if log
                        else pl + r"$_\mathrm{thresh}$"
                    )
                    ax.set_xlabel(pl_thresh)
                    ax.set_ylabel(rf"percentage with {pl} > {pl_thresh}")
                    ax.legend()
                    fn = pdir / f"bin_{s}_{e}_{ix[0]}.{self.file_format}"
                    self.logger.info(f"saving {fn}")
                    fig.tight_layout()
                    fig.savefig(fn)
                    plt.close()
