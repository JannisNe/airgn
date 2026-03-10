import json
import re
from collections.abc import Generator
from collections import defaultdict
from typing import Optional, Any, Literal
from pathlib import Path

from matplotlib.colors import Normalize
from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import umap
from scipy.stats import kstest

from ampel.abstract.AbsPhotoT3Unit import AbsPhotoT3Unit
from ampel.struct.T3Store import T3Store
from ampel.types import T3Send
from ampel.view.TransientView import TransientView

from timewise.util.path import expand
from airgn.desi.agn_value_added_catalog import get_agn_bitmask
from ampel.airgn.t3.NPointsVarMetricsAggregator import NPointsVarMetricsAggregator
from airgn.rejection_sampling import repeated_matching


# AB offset to Vega for WISE bands from Jarrett et al. (2011)
# https://dx.doi.org/10.1088/0004-637X/735/2/112
WISE_AB_OFFSET = {
    "W1": 2.699,
    "W2": 3.339,
}


def get_agn_desc(agn_bitmask, agn_mask) -> list[str]:
    mask = str(bin(int(agn_bitmask))).replace("0b", "")[::-1]
    return [am[0] for im, am in zip(mask, agn_mask["AGN_MASKBITS"]) if im]


METRIC_PARAMS = pd.read_csv(Path(__file__).parent / "metric_params.csv", index_col=0)


class FeetsOfAGN(AbsPhotoT3Unit, NPointsVarMetricsAggregator):
    """
    Plot lightcurves of transients using matplotlib
    """

    path: str
    exclude_features_fit: Optional[list[str]] = None
    exclude_features_corner: Optional[list[str]] = None
    n_point_cols: list[str] = [f"W{i + 1}_NPoints" for i in range(2)]
    mongo_uri: str = "mongodb://localhost:27017"
    file_format: str = "pdf"
    corner: bool = True
    umap: bool = True
    umap_parameters: dict[str, Any] = {}
    resample: Literal["agn", "non-agn", "none"] = "agn"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._client = MongoClient(self.mongo_uri)
        self._col = self._client[self.input_mongo_db_name]["input"]
        self._agn_bitmask = get_agn_bitmask()
        self._path = expand(self.path)
        self._path.mkdir(parents=True, exist_ok=True)

        if self.exclude_features_fit and not self.exclude_features_corner:
            self.exclude_features_corner = self.exclude_features_fit

    def _get_metric_name(self, raw_name) -> str | None:
        mns = {s for s in METRIC_PARAMS.index if s in raw_name}
        if len(mns) == 0:
            return None
        match_length = [len(re.search(imn, raw_name).group()) for imn in mns]
        return list(mns)[np.argmax(match_length)]

    def _bin_key(self, s, e):
        return f"bin_{s}_{e}"

    def process(
        self, gen: Generator[TransientView, T3Send, None], t3s: None | T3Store = None
    ) -> dict[str, dict[str, Any]]:
        # ---------------------- aggregate results ---------------------- #

        res = self.aggregate_results(gen)
        metric_names = res.columns

        # ---------------------- select agn and non-agn ---------------------- #

        res["agn"] = ~(res["decoded_agn_mask"] == "0")
        wise_agn_bit = res["decoded_agn_mask"].str[15]
        wise_agn_mask = wise_agn_bit.notna() & wise_agn_bit.astype(float).astype(bool)
        res["wise_agn"] = wise_agn_mask
        res["non_wise_agn"] = res["agn"] & ~wise_agn_mask

        # ---------------------- re-sample non-agn to match agn ---------------------- #
        res["sampled"] = True
        if self.resample != "none":
            resample_mask = res.agn if self.resample == "agn" else ~res.agn
            proposal = res.loc[resample_mask, "W1_Mean"]
            target = res.loc[~resample_mask, "W1_Mean"]
            # to be able to resample the non-AGN to the AGN distribution, the AGN distribution has to be
            # within the bounds of the non-AGN distribution
            target_outside_proposal = (target < proposal.min()) | (
                target > proposal.max()
            )
            sampled_proposal_index = repeated_matching(
                proposal,
                target[~target_outside_proposal],
                min_samples=10,
                plot_path=self._path / "W1_Mean_sampling.pdf",
            )

            # make sure the sampling produced two compatible distributions
            pval = kstest(
                target[~target_outside_proposal],
                proposal.loc[proposal.index.difference(sampled_proposal_index)],
            ).pvalue
            assert pval > 0.05
            res.loc[sampled_proposal_index, "sampled"] = False
            res.loc[
                target_outside_proposal.index[target_outside_proposal], "sampled"
            ] = False

        # ---------------------- umap and corner plot ---------------------- #

        t3res = defaultdict(dict)

        if self.corner or self.umap:
            for res_bin, s, e in self.iter_npoints_binned(res):
                corner_df = pd.DataFrame(index=res_bin.index[res_bin.sampled])
                corner_df["agn"] = res_bin.loc[corner_df.index, "agn"]
                for m in metric_names:
                    mn = self._get_metric_name(m)

                    if (
                        # Skip if no matching metric was found for the column name
                        not mn
                        # exclude npoints because it's not a real variability metric
                        # and has not enough variance so the KDE will collapse
                        or (m in self.n_point_cols)
                        # containment will be calculated l8er :-)
                        or m.endswith("Containment")
                        # skip features if requested
                        or any([m.endswith(mn) for mn in self.exclude_features_corner])
                    ):
                        continue

                    meta = METRIC_PARAMS.loc[mn]
                    pn = meta["pretty_name"]
                    lim = (meta["lower"], meta["upper"])
                    if meta["log"]:
                        pl = r"$\log_{10}($" + pn + "$)$"
                        vals = np.log10(res_bin.loc[res_bin.sampled, m])
                    else:
                        pl = pn
                        vals = res_bin.loc[res_bin.sampled, m]
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

                bindir = self._path / self._bin_key(s, e)
                bindir.mkdir(parents=True, exist_ok=True)

                if self.umap:
                    reducer = umap.UMAP(random_state=42, **self.umap_parameters)
                    corner_df_nans = corner_df.isna()
                    nan_mask = corner_df_nans.any(axis=1)
                    not_fit_columns = list(self.exclude_features_fit) + ["agn"]
                    fit_df = corner_df.loc[~nan_mask].drop(
                        columns=corner_df.columns.intersection(not_fit_columns)
                    )

                    reducer.fit(fit_df)
                    embedding = reducer.embedding_
                    agn_mask = corner_df.loc[~nan_mask, "agn"]

                    # check percentages of objects that are valid
                    umap_res = {
                        "total": nan_mask.sum() / len(nan_mask),
                        "agn": (nan_mask & corner_df.agn).sum() / corner_df.agn.sum(),
                        "non_agn": (nan_mask & ~corner_df.agn).sum()
                        / (~corner_df.agn).sum(),
                    }
                    self.logger.info(
                        f"UMAP bin ({s}, {e}): {json.dumps(umap_res, indent=4)}"
                    )
                    t3res[self._bin_key(s, e)]["umap"] = umap_res

                    # plot features that are nan per class
                    fig, ax = plt.subplots()
                    for m, label in zip(
                        [~corner_df.agn, corner_df.agn], ["non-AGN", "AGN"]
                    ):
                        pdf = (
                            corner_df_nans.loc[m].drop(columns=["agn"]).sum() / m.sum()
                        )
                        ax.bar(
                            pdf.index, pdf.values, label=label, ec="white", alpha=0.5
                        )
                    xticklabels = pdf.index
                    ax.set_ylabel("Percentage")
                    ax.set_xticks(np.arange(0, len(xticklabels)))
                    ax.set_xticklabels(xticklabels, rotation=60, ha="right")
                    ax.legend()
                    fig.tight_layout()
                    fn = bindir / f"nan_percentages.{self.file_format}"
                    self.logger.debug(f"Saving {fn}")
                    fig.savefig(fn)
                    plt.close()

                    # make bins in feature space
                    bins = [np.linspace(np.min(e), np.max(e), 40) for e in embedding.T]

                    # calculate 2d histograms for AGN and non-AGN
                    agn_h, _ = np.histogramdd(embedding[agn_mask], bins=bins)
                    non_agn_h, _ = np.histogramdd(embedding[~agn_mask], bins=bins)

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
                    bin_maps = [
                        np.clip(
                            np.digitize(embedding[:, i], bins[i]) - 1,
                            0,
                            containment.shape[i] - 1,
                        )
                        for i in range(embedding.shape[1])
                    ]

                    cpn = METRIC_PARAMS.loc["Containment", "pretty_name"]
                    corner_df[cpn] = np.nan
                    corner_df.loc[~nan_mask, cpn] = 1 - containment[*bin_maps]

                    # also map back to original results data to make histograms later
                    res.loc[corner_df.index, "Containment"] = corner_df[cpn]

                    meshbins = np.meshgrid(*bins)

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

                    for i, (h, cmap, vmin, vmax, fext, label) in enumerate(itr):
                        fig, ax = plt.subplots()

                        if (ndim := embedding.shape[1]) == 2:
                            norm = Normalize(vmin=vmin, vmax=vmax)
                            ax.pcolormesh(*meshbins, h, cmap=cmap, norm=norm)
                            fig.colorbar(
                                ax=ax,
                                location="right",
                                mappable=plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                                label=label,
                                extend="both",
                                pad=0.1,
                            )
                            fn = bindir / f"umap_2d{fext}.{self.file_format}"
                        elif ndim == 1:
                            hb = meshbins[0]
                            ax.bar(hb[:-1], h, width=np.diff(hb), alpha=0.8, ec="white")
                            fn = bindir / f"umap_1d{fext}.{self.file_format}"
                        else:
                            raise NotImplementedError()

                        fig.savefig(fn)
                        plt.close()

                # ---------------------- corner ---------------------- #

                if self.corner:
                    plot_df = corner_df.drop(
                        columns=corner_df.columns.intersection(
                            self.exclude_features_corner
                        )
                    )
                    fig = sns.pairplot(
                        plot_df,
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
                    plot_df["W1_Mean"] = res_bin.loc[plot_df.index, "W1_Mean"]
                    mask = ~plot_df.agn & plot_df["W1_Mean"].notna()
                    w1_means = plot_df.loc[mask, "W1_Mean"]
                    w1_bins = np.quantile(w1_means, [0, 0.33, 0.66, 1])
                    w1_bins[-1] *= 1.1
                    w1_bins_labels = np.array(
                        [
                            f"{sb:.1f} < W1 < {eb:.1f}"
                            for sb, eb in zip(w1_bins[:-1], w1_bins[1:])
                        ]
                    )
                    w1_bin = np.digitize(w1_means, bins=w1_bins)
                    corner_non_agn = plot_df[mask].drop(columns=["agn"])
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

        for metric_name, meta in METRIC_PARAMS.iterrows():
            pn = meta["pretty_name"]
            log = meta["log"]
            mb = meta["multiband"]
            pl = r"$\log_{10}($" + pn + "$)$" if log else pn

            cols = (
                (
                    [f"W{i}_{metric_name}" for i in range(1, 3)]
                    if not mb
                    else [f"W1_W2_{metric_name}"]
                )
                if not metric_name == "Containment"
                else ["Containment"]
            )

            fig, ax = plt.subplots()
            vals = res[cols].min(axis=1)
            if log:
                m = vals > 0
                vals = np.log10(vals[m])
            else:
                m = np.array([True] * len(vals))
            bins = np.linspace(vals.min(), vals.max(), 20)
            ax.hist(
                vals[res.agn & m & res.sampled],
                ec="white",
                alpha=0.5,
                color="C1",
                label=f"{(res.agn & m & res.sampled).sum() / res.agn.sum() * 100:.1f}% of AGN",
                bins=bins,
            )
            ax.hist(
                vals[~res.agn & m & res.sampled],
                ec="white",
                alpha=0.5,
                color="C0",
                label=f"{(~res.agn & m & res.sampled).sum() / (~res.agn).sum() * 100:.1f}% of non-AGN",
                bins=bins,
            )
            if any(~res.sampled & ~res.agn):
                ax.hist(
                    vals[m & ~res.sampled],
                    alpha=0.8,
                    color="C0",
                    label=f"{(m & ~res.sampled & ~res.agn).sum() / (~res.agn).sum() * 100:.1f}% of non-AGN ignored",
                    bins=bins,
                    histtype="step",
                    ls=":",
                )
            if any(~res.sampled & res.agn):
                ax.hist(
                    vals[m & ~res.sampled],
                    alpha=0.8,
                    color="C1",
                    label=f"{(m & ~res.sampled & res.agn).sum() / res.agn.sum() * 100:.1f}% of AGN ignored",
                    bins=bins,
                    histtype="step",
                    ls=":",
                )
            ax.legend()
            ax.set_xlabel(pl)
            ax.set_ylabel("counts")
            ax.set_yscale("log")
            fn = self._path / f"{metric_name}_hist.{self.file_format}"
            self.logger.debug(f"saving {fn}")
            fig.tight_layout()
            fig.savefig(fn)
            plt.close()

            # ---------------------- violinplots ---------------------- #

            for res_bin, s, e in self.iter_npoints_binned(res):
                res_bin = res_bin[res_bin.sampled]
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
                            vals = res_bin.loc[ixm, col].values
                            y.append(np.log10(vals[vals > 0]) if log else vals)
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
                    ax.set_ylim(meta["lower"], meta["upper"])

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
                self.logger.debug(f"saving {fn}")
                fig.tight_layout()
                fig.savefig(fn)
                plt.close()

            # ---------------------- metric cuts ---------------------- #

            metric_threshs = np.linspace(meta["lower"], meta["upper"], 100)

            for res_bin, s, e in self.iter_npoints_binned(res):
                res_bin = res_bin[res_bin.sampled]
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
                        if log:
                            vals_mask = (type_res_bin[cols] > 0).all(axis=1)
                            vals = np.log10(type_res_bin.loc[vals_mask, cols])
                        else:
                            vals_mask = type_res_bin[cols].notna().all(axis=1)
                            vals = type_res_bin.loc[vals_mask, cols]

                        # number of agn passing cut
                        n_selected_agn = (
                            (vals[type_res_bin.loc[vals_mask, "agn"]] > thresh)
                            .all(axis=1)
                            .sum()
                        )

                        # number of non agn passing cut
                        if log:
                            res_bin_mask = (res_bin[cols] > 0).all(axis=1)
                            res_bin_non_agn_vals = np.log10(
                                res_bin.loc[~res_bin["agn"] & res_bin_mask, cols]
                            )
                        else:
                            res_bin_mask = res_bin[cols].notna().all(axis=1)
                            res_bin_non_agn_vals = res_bin.loc[
                                ~res_bin["agn"] & res_bin_mask, cols
                            ]

                        n_selected_non_agn = (
                            (res_bin_non_agn_vals > thresh).all(axis=1).sum()
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
                            (vals[type_res_bin.loc[vals_mask, "wise_agn"]] > thresh)
                            .all(axis=1)
                            .sum()
                        )
                        if n_wise_agn > 0:
                            percentage_wise_agn.append(n_selected_wise_agn / n_wise_agn)
                        else:
                            percentage_wise_agn.append(np.nan)
                        n_selected_non_wise_agn = (
                            (vals[type_res_bin.loc[vals_mask, "non_wise_agn"]] > thresh)
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
                    self.logger.debug(f"saving {fn}")
                    fig.tight_layout()
                    fig.savefig(fn)
                    plt.close()

        return t3res
