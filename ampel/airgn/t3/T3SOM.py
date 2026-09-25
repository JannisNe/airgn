from typing import Generator, Literal

import numpy as np
import pandas as pd
from astropy.time import Time
from matplotlib import pyplot as plt
import python_som
from ampel.abstract.AbsPhotoT3Unit import AbsPhotoT3Unit
from ampel.abstract.AbsT3Unit import T
from ampel.struct.T3Store import T3Store
from ampel.struct.UnitResult import UnitResult
from ampel.types import T3Send, UBson
from scipy.stats import kstest
from sklearn.model_selection import StratifiedKFold, cross_validate
from timewise.util.path import expand
from tqdm import tqdm

from airgn.rejection_sampling import repeated_matching


class T3SOM(AbsPhotoT3Unit):
    # SOM parameters
    som_size: tuple[int, int] = 10, 10

    # input data processing
    t2_lc_unit: Literal["T2StackVisits", "T2MaggyToFluxDensity"]
    drop_wise_agn: bool = False
    resample: Literal["agn", "non-agn", "none"] = "agn"

    # output
    plot_dir: str
    mplstyle: str | None = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._random_state = 42
        self._plot_path = expand(self.plot_dir)
        self._plot_path.mkdir(parents=True, exist_ok=True)

    def process(
        self, gen: Generator[T, T3Send, None], t3s: T3Store
    ) -> UBson | UnitResult:
        res = {}
        raw_lcs = {}
        n_steps = 0
        for view in gen:
            for t2 in view.get_t2_views(self.t2_lc_unit, code=0):
                lc = t2.get_payload()
                break

            if not view.extra:
                continue

            body = dict(view.extra)
            mask = str(bin(int(view.extra["AGN_MASKBITS"]))).replace("0b", "")[::-1]
            body["decoded_agn_mask"] = mask
            res[view.stock["stock"]] = body
            raw_lcs[view.stock["stock"]] = lc

            if (i_n_steps := len(lc)) > n_steps:
                n_steps = i_n_steps

        res = pd.DataFrame.from_dict(res, orient="index")
        res["agn"] = ~(res["decoded_agn_mask"] == "0")
        wise_agn_bit = res["decoded_agn_mask"].str[15]
        wise_agn_mask = wise_agn_bit.notna() & wise_agn_bit.astype(float).astype(bool)
        res["wise_agn"] = wise_agn_mask
        res["non_wise_agn"] = res["agn"] & ~wise_agn_mask

        if self.drop_wise_agn:
            res = res[~res.wise_agn]

        if self.mplstyle is not None:
            plt.style.use(self.mplstyle)

        # ---------------------- re-sample non-agn to match agn ---------------------- #

        res["sampled"] = True
        if self.resample != "none":
            resample_mask = res.agn if self.resample == "agn" else ~res.agn
            proposal = res.loc[resample_mask, "FLUX_W1"]
            target = res.loc[~resample_mask, "FLUX_W1"]
            # to be able to resample the non-AGN to the AGN distribution, the AGN distribution has to be
            # within the bounds of the non-AGN distribution
            target_outside_proposal = (target < proposal.min()) | (
                target > proposal.max()
            )
            sampled_proposal_index = repeated_matching(
                proposal,
                target[~target_outside_proposal],
                min_samples=int(0.01 * len(proposal)),
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

        # ------------------------------ collect features ------------------------------ #
        # The features in this case are just the w1 and w2 flux densities normed by the
        # respective median, stacked horizontally per source.

        index = res[res.sampled].index
        features = np.full((len(index), n_steps * 2), np.nan)

        wise_end = Time("2011-02-01").mjd
        neowise_start = Time("2013-12-29").mjd
        gap_length = neowise_start - wise_end

        for row, i in enumerate(tqdm(index, desc="Formatting lightcurves")):
            lc = raw_lcs[i]

            if self.t2_lc_unit == "T2MaggyToFluxDensity":
                mjd1 = np.fromiter((x["LC_MJD_W1"] for x in lc), dtype=float)
                mjd2 = np.fromiter((x["LC_MJD_W2"] for x in lc), dtype=float)

                assert np.all(np.abs(mjd1 - mjd2) <= 10)
                mean_mjd = (mjd1 + mjd2) / 2
            else:
                mean_mjd = np.fromiter((x["mean_mjd"] for x in lc), dtype=float)

            offset = (gap_length - 180) * (mean_mjd > wise_end).astype(float)
            epoch = np.rint((mean_mjd - mean_mjd.min() - offset) / 180).astype(int)

            if np.unique(epoch).size != epoch.size:
                raise RuntimeError(f"Found ambiguous epochs!\n{epoch}")

            w1 = np.fromiter((x["w1meanfluxdensity"] for x in lc), dtype=float)
            w2 = np.fromiter((x["w2meanfluxdensity"] for x in lc), dtype=float)

            features[row, epoch] = w1 / np.median(w1)
            features[row, epoch + n_steps] = w2 / np.median(w2)

        features = pd.DataFrame(features, index=index, columns=range(n_steps * 2))

        target = res.loc[res.sampled, "agn"].astype(int)

        # ------------------------------ train the map ------------------------------ #
        n_splits = 10
        kf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self._random_state
        )

        soms = []
        test_indices = []
        for train_index, test_index in kf.split(features, target):
            som = python_som.SOM(
                x=self.som_size[0],
                y=self.som_size[1],
                input_len=n_steps * 2,
                learning_rate=0.5,
                neighborhood_radius=1.0,
                neighborhood_function="gaussian",
                cyclic_x=True,
                cyclic_y=True,
                random_seed=self._random_state,
            )
            som.fit(features.iloc[train_index])
            soms.append(som)
            test_indices.append(test_index)

        # ---------------------- plot individual models ---------------------- #

        individual_models_path = self._plot_path / "individual_models"
        individual_models_path.mkdir(parents=True, exist_ok=True)

        xx = np.linspace(0, 1, 100)
        recalls = []
        precisions = []

        for i, (isom, test_indices) in enumerate(zip(soms, test_indices)):
            data_test = features.iloc[test_indices]
            win_map = np.array(
                np.unravel_index(isom.predict(data_test), isom.get_shape())
            ).T

            fig, axs = plt.subplots(*self.som_size, figsize=(7, 7))
            for position in np.unique(win_map, axis=0):
                mask = (win_map[:, 0] == position[0]) & (win_map[:, 1] == position[1])
                if not any(mask):
                    continue
                ax = (
                    axs[self.som_size[0] - 1 - position[0], position[1]]
                    if self.som_size[1] > 1
                    else axs[position[0]]
                )
                ax.plot(np.nanmean(data_test[mask], axis=0), c="k")
                ax.fill_between(
                    np.arange(n_steps),
                    *np.nanquantile(data_test[mask], [0.05, 0.95], axis=0),
                    color="gray",
                    alpha=0.5,
                )
                ax.xaxis.set_ticklabels([])
                ax.yaxis.set_ticklabels([])
            fig.savefig(individual_models_path / f"{i}_som_timeseries.pdf")
            plt.close()

            maps = []
            target_test = target.iloc[test_indices]
            for mask in [~target_test, target_test]:
                counts = np.unique(
                    som.predict(data_test[mask]),
                    return_counts=True,
                    axis=0,
                )
                i_map = np.zeros(self.som_size)
                for p, c in zip(
                    np.array(np.unravel_index(counts[0], som.get_shape())).T,
                    counts[1],
                    strict=False,
                ):
                    i_map[p[0], p[1]] = c
                maps.append(i_map)

            purity_map = maps[1] / (maps[0] + maps[1])
            recall_map = maps[1] / maps[1].sum()

            fig, axs = plt.subplots(ncols=4, figsize=(20, 5))
            for cmap, pmap, ax in zip(
                ["Reds", "Blues", "copper", "Reds"],
                [maps[1], maps[0], purity_map, recall_map],
                axs,
                strict=False,
            ):
                mesh = ax.pcolormesh(
                    pmap, cmap=cmap
                )  # plotting the distance map as background
                fig.colorbar(mesh, ax=ax)
            fig.tight_layout()
            fig.savefig(individual_models_path / f"{i}_som_maps.pdf")
            plt.close()

            flat_sig_map = maps[1].flatten()
            flat_bkg_map = maps[0].flatten()
            probs = purity_map.flatten()

            recall = []
            precision = []

            for i in xx:
                m = probs >= i
                precision.append(
                    flat_sig_map[m].sum()
                    / (flat_sig_map[m].sum() + flat_bkg_map[m].sum())
                )
                recall.append(flat_sig_map[m].sum() / flat_sig_map.sum())

            fig, ax = plt.subplots()
            ax.plot(xx, precision, label="Precision")
            ax.plot(xx, recall, label="Recall")
            ax.set_xlabel("Precision")
            ax.set_ylabel("Score")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.legend()
            fig.tight_layout()
            fig.savefig(individual_models_path / f"{i}_scores.pdf")
            plt.close()

        # ---------------------- plot totals ---------------------- #

        fig, ax = plt.subplots()
        for i, arr, label in enumerate(
            zip([precisions, recalls], ["Precision", "Recall"])
        ):
            c = f"C{i}"
            ax.plot(xx, np.median(arr, axis=0), label=label, color=c)
            ax.fill_between(
                xx, *np.quantile(arr, [0.05, 0.95], axis=0), alpha=0.5, color=c
            )
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()
        fig.tight_layout()
        fig.savefig(self._plot_path / "scores.pdf")
        plt.close()

        # ---------------------- plot totals separate for WISE AGN ---------------------- #
