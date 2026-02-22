from collections.abc import Generator
from typing import Optional

from matplotlib.colors import Normalize
from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import umap

from ampel.abstract.AbsPhotoT3Unit import AbsPhotoT3Unit
from ampel.struct.T3Store import T3Store
from ampel.types import T3Send
from ampel.view.TransientView import TransientView

from timewise.util.path import expand
from airgn.desi.agn_value_added_catalog import get_agn_bitmask
from ampel.util.NPointsIterator import NPointsIterator


# AB offset to Vega for WISE bands from Jarrett et al. (2011)
# https://dx.doi.org/10.1088/0004-637X/735/2/112
WISE_AB_OFFSET = {
    "W1": 2.699,
    "W2": 3.339,
}


def get_agn_desc(agn_bitmask, agn_mask) -> list[str]:
    mask = str(bin(int(agn_bitmask))).replace("0b", "")[::-1]
    return [am[0] for im, am in zip(mask, agn_mask["AGN_MASKBITS"]) if im]


metric_params = {
    "RedderWhenBrighter": dict(
        log=False, range=(-10, 10), pretty_name="RWB", multiband=True
    ),
    "StetsonL": dict(log=False, range=(-10, 100), pretty_name=r"$L$", multiband=True),
    "PearsonsR": dict(log=False, range=(-1, 1), pretty_name="$r$", multiband=True),
    "ExcessVariance": dict(
        log=False,
        range=(-0.1, 0.1),
        pretty_name=r"$\sigma^2_\mathrm{rms}$",
        multiband=False,
    ),
    "Eta": dict(log=True, range=(-1, 2), pretty_name=r"$\eta$", multiband=False),
    "ReducedChi2": dict(
        log=True, range=(-2, 2), pretty_name=r"$\chi_\mathrm{red}^2$", multiband=False
    ),
    "Autocor_length": dict(
        log=True, range=(-2, 2), pretty_name=r"$\tau$", multiband=False
    ),
    "MeanVariance": dict(
        log=True,
        range=(-2, 2),
        pretty_name=r"$\sigma_\mathrm{\mu} / \mu$",
        multiband=False,
    ),
}


class FeetsOfAGN(AbsPhotoT3Unit, NPointsIterator):
    """
    Plot lightcurves of transients using matplotlib
    """

    path: str
    input_mongo_db_name: str
    exclude_features: Optional[list[str]] = None
    n_point_cols: list[str] = [f"W{i + 1}_NPoints" for i in range(2)]
    mongo_uri: str = "mongodb://localhost:27017"
    iter_max: int | None = None
    file_format: str = "pdf"
    corner: bool = True
    umap: bool = True

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
        # ---------------------- aggregate results ---------------------- #

        res = {}
        n_iter = 0
        for view in gen:
            input_res = None
            body = None
            for t2 in view.get_t2_views("T2FeetsTimewise", code=0):
                body = dict(t2.get_payload())
                input_res = self._col.find_one({"orig_id": t2.stock})
                break

            if not input_res:
                continue

            if not body:
                continue

            # do not include objects that are outside the specified range of npoint bins
            if not all([self.is_in_range(body[c]) for c in self.n_point_cols]):
                continue

            body.update(input_res)
            mask = str(bin(int(input_res["AGN_MASKBITS"]))).replace("0b", "")[::-1]
            body["decoded_agn_mask"] = mask
            res[view.stock["stock"]] = body

            n_iter += 1
            if self.iter_max and n_iter >= self.iter_max:
                self.logger.info("iteration limit reached, stopping loop")
                break

        res = pd.DataFrame.from_dict(res, orient="index")
        metric_names = res.columns

        # ---------------------- select agn and non-agn ---------------------- #

        res["agn"] = ~(res["decoded_agn_mask"] == "0")
        wise_agn_bit = res["decoded_agn_mask"].str[15]
        wise_agn_mask = wise_agn_bit.notna() & wise_agn_bit.astype(float).astype(bool)
        res["wise_agn"] = wise_agn_mask
        res["non_wise_agn"] = res["agn"] & ~wise_agn_mask

        # ---------------------- umap and corner plot ---------------------- #

        if self.corner or self.umap:
            for res_bin, s, e in self.iter_npoints_binned(res):
                corner_df = pd.DataFrame(index=res_bin.index)
                corner_df["agn"] = res_bin["agn"]
                for m in metric_names:
                    # exclude npoints because it's not a real variability metric
                    # and has not enough variance so the KDE will collapse
                    if (
                        (m in self.n_point_cols)
                        or m.startswith("containment")
                        or any([m.startswith(mn) for mn in self.exclude_features])
                    ):
                        continue
                    meta = metric_params[m.split("_")[-1]]
                    pn = meta["pretty_name"]
                    lim = meta["range"]
                    if meta["log"]:
                        pl = r"$\log_{10}($" + pn + "$)$"
                        vals = np.log10(res_bin[m])
                    else:
                        pl = pn
                        vals = res_bin[m]
                    if meta["multiband"]:
                        pl += " " + " - ".join(m.split("_")[:-1])
                    else:
                        pl += " " + m.split("_")[0]

                    vals_mask = (vals > lim[0]) & (vals < lim[1])
                    pd.options.mode.chained_assignment = None
                    vals.loc[~vals_mask] = np.nan
                    pd.options.mode.chained_assignment = "warn"
                    corner_df[pl] = vals

                # ---------------------- UMAP ---------------------- #

                bindir = self._path / f"bin_{s}_{e}"
                bindir.mkdir(parents=True, exist_ok=True)

                if self.umap:
                    reducer = umap.UMAP(random_state=42)
                    nan_mask = corner_df.isna().any(axis=1)
                    reducer.fit(
                        corner_df.loc[~nan_mask, [c for c in corner_df if c != "agn"]]
                    )
                    embedding = reducer.embedding_
                    agn_mask = corner_df.loc[~nan_mask, "agn"]

                    # make bins in feature space
                    bins = [np.linspace(np.min(e), np.max(e), 40) for e in embedding.T]

                    # calculate 2d histograms for AGN and non-AGN
                    agn_h, _, _ = np.histogram2d(*embedding[agn_mask].T, bins=bins)
                    non_agn_h, _, _ = np.histogram2d(*embedding[~agn_mask].T, bins=bins)

                    # check their distributions with relative histograms
                    agn_h_rel = agn_h / np.nansum(agn_h)
                    non_agn_h_rel = non_agn_h / np.nansum(non_agn_h)

                    # define a relative efficiency
                    rh = agn_h_rel / non_agn_h_rel
                    rh[non_agn_h_rel == 0] = np.nanmax(rh)

                    # purity histogram
                    p = agn_h / (agn_h + non_agn_h)

                    # come up with a 90% contour by weighting the AGN distribution by the purity
                    weighted_agn_dist = agn_h_rel * p
                    pdist = weighted_agn_dist / np.nansum(weighted_agn_dist)

                    flat = pdist.ravel()
                    order = np.argsort(flat)[::-1]
                    flat_sorted = flat[order]

                    # Map from density → cumulative probability
                    containment = np.empty_like(flat_sorted)
                    containment[order] = np.nancumsum(flat_sorted)

                    containment = containment.reshape(pdist.shape)

                    # map containment back to original points
                    xbin = np.digitize(embedding[:, 0], bins[0]) - 1
                    ybin = np.digitize(embedding[:, 1], bins[1]) - 1

                    nx, ny = containment.shape
                    xbin = np.clip(xbin, 0, nx - 1)
                    ybin = np.clip(ybin, 0, ny - 1)
                    cpn = metric_params["containment"]["pretty_name"]
                    corner_df[cpn] = np.nan
                    corner_df.loc[~nan_mask, cpn] = 1 - containment[xbin, ybin]

                    # also map back to original results data to make histograms later
                    res.loc[corner_df.index, "containment"] = corner_df[cpn]

                    X, Y = np.meshgrid(*bins)

                    itr = [
                        (
                            agn_h_rel,
                            "Reds",
                            None,
                            None,
                            "agn_distribution",
                            "percentage",
                        ),
                        (
                            non_agn_h_rel,
                            "Blues",
                            None,
                            None,
                            "non_agn_distribution",
                            "percentage",
                        ),
                        (p, "viridis", None, None, "purity", "purity"),
                        (
                            np.log10(rh),
                            "RdBu_r",
                            -1,
                            1,
                            "relative_efficiency",
                            r"$\log_{10}(relative efficiency)$",
                        ),
                        (
                            weighted_agn_dist,
                            "Reds",
                            None,
                            None,
                            "weighted_agn_distribution",
                            "percentage",
                        ),
                        (containment, "viridis", 0, 1, "contours", "containment level"),
                    ]

                    for i, (h, cmap, vmin, vmax, fext, l) in enumerate(itr):
                        fig, ax = plt.subplots()
                        norm = Normalize(vmin=vmin, vmax=vmax)
                        ax.pcolormesh(X, Y, h, cmap=cmap, norm=norm)
                        fig.colorbar(
                            ax=ax,
                            location="right",
                            mappable=plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                            label=l,
                            extend="both",
                            pad=0.1,
                        )
                        fn = bindir / f"umap_2d{fext}.{self.file_format}"
                        fig.savefig(fn)
                        plt.close()

                # ---------------------- corner ---------------------- #

                if self.corner:
                    fig = sns.pairplot(
                        corner_df,
                        kind="kde",
                        corner=True,
                        hue="agn",
                        plot_kws={
                            "levels": [0.5, 0.68, 0.9, 0.99],
                            "linestyles": ["-", "--", "-.", ":"],
                        },
                    )
                    fn = bindir / f"corner.{self.file_format}"
                    fig.savefig(fn)
                    plt.close()

                    # make the same plot in magnitude bins
                    mask = ~res_bin.agn & res_bin["w1_mean"].notna()
                    w1_means = res_bin.loc[mask, "w1_mean"]
                    w1_bins = np.quantile(w1_means, [0, 0.33, 0.66, 1])
                    w1_bins[-1] *= 1.1
                    w1_bins_labels = np.array(
                        [
                            f"{sb:.1f} < W1 < {eb:.1f}"
                            for sb, eb in zip(w1_bins[:-1], w1_bins[1:])
                        ]
                    )
                    w1_bin = np.digitize(w1_means, bins=w1_bins)
                    corner_non_agn = corner_df[mask].drop(columns=["agn"])
                    corner_non_agn["W1 mean"] = w1_bins_labels[w1_bin - 1]

                    fig = sns.pairplot(
                        corner_non_agn,
                        kind="kde",
                        corner=True,
                        hue="W1 mean",
                        plot_kws={
                            "levels": [0.5, 0.68, 0.9],
                            "linestyles": ["-", "--", ":"],
                        },
                    )
                    fn = bindir / f"corner_non_agn.{self.file_format}"
                    fig.savefig(fn)
                    plt.close()

        # ---------------------- histograms ---------------------- #

        for metric_name in metric_names:
            meta = metric_params[metric_name]
            pn = meta["pretty_name"]
            log = meta["log"]
            mb = meta["multiband"]
            pl = r"$\log_{10}($" + pn + "$)$" if log else pn

            cols = (
                [f"w{i}_{metric_name}" for i in range(1, 3)]
                if not mb
                else [f"w1_w2_{metric_name}"]
            )

            fig, ax = plt.subplots()
            vals = res[cols].min(axis=1)
            if log:
                m = vals > 0
                vals = np.log10(vals[m])
            ax.hist(vals, ec="white", alpha=0.8)
            ax.set_xlabel(pl)
            ax.set_ylabel("counts")
            ax.set_yscale("log")
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

                fig, axs = plt.subplots(nrows=len(cols) + 1, sharex="all")
                axs[0].bar(np.arange(-1, len(labels) - 1), n, alpha=0.5, ec="none")
                axs[0].set_yscale("log")
                axs[0].set_ylabel("counts")

                for i, (ax, col) in enumerate(zip(axs[1:], cols)):
                    m = res_bin[col].notna()
                    x = []
                    y = []
                    for ix in self._agn_bitmask["AGN_MASKBITS"]:
                        ixb = res_bin["decoded_agn_mask"].str[ix[1]]
                        ixm = m & ixb.notna() & ixb.astype(float).astype(bool)
                        if any(ixm):
                            vals = res_bin.loc[
                                ixm,
                                col,
                            ].values.tolist()
                            y.append(np.log10(vals) if log else vals)
                            x.append(
                                ix[1] if ix[1] < 10 else ix[1] - 2
                            )  # bits 8 and 9 are skipped
                    not_agn_mask = m & (res_bin["decoded_agn_mask"] == "0")
                    x.append(-1)
                    not_agn_vals = res_bin.loc[not_agn_mask, col].values.tolist()
                    y.append(np.log10(not_agn_vals) if log else not_agn_vals)
                    ax.violinplot(
                        dataset=y, positions=x, showextrema=False, showmedians=True
                    )
                    if not mb:
                        ax.set_ylabel(f"W{i + 1}")
                    ax.set_ylim(*meta["range"])

                if mb:
                    axs[-1].set_ylabel(pl)
                else:
                    fig.supylabel(pl)
                axs[-1].set_xticks(np.arange(-1, len(labels) - 1))
                axs[-1].set_xticklabels(labels, rotation=60, ha="right")

                fn = (
                    self._path
                    / f"bin_{s}_{e}"
                    / f"{metric_name}_bin_{s}_{e}.{self.file_format}"
                )
                fn.parent.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"saving {fn}")
                fig.tight_layout()
                fig.savefig(fn)
                plt.close()

            # ---------------------- metric cuts ---------------------- #

            metric_threshs = np.linspace(*meta["range"], 100)

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
                        if n_agn > 0:
                            completeness.append(n_selected_agn / n_agn)
                        else:
                            completeness.append(np.nan)
                        if (total := n_selected_agn + n_selected_non_agn) > 0:
                            purity.append(n_selected_agn / total)
                        else:
                            purity.append(np.nan)

                        n_selected_wise_agn = (
                            (type_res_bin.loc[type_res_bin["wise_agn"], cols] > thresh)
                            .all(axis=1)
                            .sum()
                        )
                        if n_wise_agn > 0:
                            percentage_wise_agn.append(n_selected_wise_agn / n_wise_agn)
                        else:
                            percentage_wise_agn.append(np.nan)
                        n_selected_non_wise_agn = (
                            (
                                type_res_bin.loc[type_res_bin["non_wise_agn"], cols]
                                > thresh
                            )
                            .all(axis=1)
                            .sum()
                        )
                        if n_non_wise_agn > 0:
                            percentage_non_wise_agn.append(
                                n_selected_non_wise_agn / n_non_wise_agn
                            )
                        else:
                            percentage_non_wise_agn.append(np.nan)

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
                    fn = (
                        self._path
                        / f"bin_{s}_{e}"
                        / f"{metric_name}_bin_{s}_{e}_{ix[0]}.{self.file_format}"
                    )
                    self.logger.info(f"saving {fn}")
                    fig.tight_layout()
                    fig.savefig(fn)
                    plt.close()
