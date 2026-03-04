from typing import Generator, Literal
import os
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, KFold, cross_validate
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from timewise.util.path import expand

from ampel.abstract.AbsPhotoT3Unit import AbsPhotoT3Unit
from ampel.abstract.AbsT3Unit import T
from ampel.struct.T3Store import T3Store
from ampel.struct.UnitResult import UnitResult
from ampel.types import T3Send, UBson

from ampel.airgn.t3.NPointsVarMetricsAggregator import NPointsVarMetricsAggregator
from ampel.airgn.t3.FeetsOfAGN import METRIC_PARAMS

from airgn.rejection_sampling import repeated_matching


class AGNVarXGB(AbsPhotoT3Unit, NPointsVarMetricsAggregator):
    plot_dir: str

    resample: Literal["agn", "non-agn", "none"] = "agn"
    exclude_features: list[str] | None = None

    n_cpu: int = os.cpu_count()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._rng = np.random.default_rng(42)
        self._plot_path = expand(self.plot_dir)
        self._plot_path.mkdir(parents=True, exist_ok=True)

    def process(self, gen: Generator[T, T3Send, None], t3s: T3Store) -> UBson:
        res = self.aggregate_results(gen)

        res["agn"] = ~(res["decoded_agn_mask"] == "0")

        # ---------------------- re-sample non-agn to match agn ---------------------- #

        res["sampled"] = True
        if self.resample != "none":
            resample_mask = res.agn if self.resample == "agn" else ~res.agn
            proposal = res.loc[resample_mask, "W1_Mean"]
            target = res.loc[~resample_mask, "W1_Mean"]
            # to be able to resample the non-AGN to the AGN ditribution, the AGN distribution has to be
            # within the bounds of the non-AGN distribution
            target_outside_proposal = (target < proposal.min()) | (
                target > proposal.max()
            )
            sampled_proposal_index = repeated_matching(
                proposal, target[~target_outside_proposal], min_samples=10
            )
            res.loc[sampled_proposal_index, "sampled"] = False
            res.loc[
                target_outside_proposal.index[target_outside_proposal], "sampled"
            ] = False

        metrics = []
        for m, info in METRIC_PARAMS.items():
            if m in self.exclude_features:
                continue
            if info.multiband:
                metrics.append(f"W1_W2_{m}")
            else:
                metrics.extend([f"W{i}_m" for i in range(1, 3)])

        target = res.loc[res.sampled, "agn"].astype(int).values
        data = res.loc[res.sampled, metrics].values

        n_splits = 6
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self._rng)
        xgb_model = xgb.XGBClassifier(n_jobs=self.n_cpu)
        scores = ["accuracy", "precision", "recall", "f1"]
        res = cross_validate(
            xgb_model,
            data,
            target,
            scoring=scores,
            cv=kf,
            n_jobs=1,
            return_estimator=True,
            return_indices=True,
        )

        # plot the models
        for i in range(n_splits):
            est = res["estimator"][i]
            xgb.plot_importance(est)
            fig = plt.gcf()
            fn = self._plot_path / f"{i}_importance.pdf"
            fig.savefig(fn, bbox_inches="tight")
            plt.close()

            indices = res["indices"]["test"][i]
            conf_disp = ConfusionMatrixDisplay.from_estimator(
                est, data[indices], target[indices]
            )
            conf_disp.plot()
            fig = plt.gcf()
            fn = self._plot_path / f"{i}_confudion_matrix.pdf"
            fig.savefig(fn, bbox_inches="tight")
            plt.close()

        # drop estimators and return non-binary results
        res.pop("estimator")
        return res
