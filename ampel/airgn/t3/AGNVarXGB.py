from typing import Generator, Literal
import os

from sklearn.metrics import (
    mean_squared_error,
    ConfusionMatrixDisplay,
    recall_score,
    precision_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from timewise.util.path import expand

from ampel.abstract.AbsPhotoT3Unit import AbsPhotoT3Unit
from ampel.abstract.AbsT3Unit import T
from ampel.struct.T3Store import T3Store
from ampel.types import T3Send, UBson

from ampel.airgn.t3.NPointsVarMetricsAggregator import NPointsVarMetricsAggregator
from ampel.airgn.t3.FeetsOfAGN import METRIC_PARAMS, get_metric_info

from airgn.rejection_sampling import repeated_matching


class AGNVarXGB(AbsPhotoT3Unit, NPointsVarMetricsAggregator):
    plot_dir: str
    n_estimators: int

    learning_rate: float = 1
    smote: bool = False
    drop_wise_agn: bool = False

    resample: Literal["agn", "non-agn", "none"] = "agn"
    exclude_features: list[str] | None = None

    n_cpu: int = os.cpu_count() - 1

    n_point_cols: list[str] = [f"W{i + 1}_NPoints" for i in range(2)]
    mongo_uri: str = "mongodb://localhost:27017"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._random_state = 42
        self._plot_path = expand(self.plot_dir)
        self._plot_path.mkdir(parents=True, exist_ok=True)

    def process(self, gen: Generator[T, T3Send, None], t3s: T3Store) -> UBson:
        res = self.aggregate_results(gen)

        res["agn"] = ~(res["decoded_agn_mask"] == "0")
        wise_agn_bit = res["decoded_agn_mask"].str[15]
        wise_agn_mask = wise_agn_bit.notna() & wise_agn_bit.astype(float).astype(bool)
        res["wise_agn"] = wise_agn_mask
        res["non_wise_agn"] = res["agn"] & ~wise_agn_mask

        if self.drop_wise_agn:
            res = res[~res.wise_agn]

        # SMOTE can not handle nans so drop
        if self.smote:
            res = res[~res.isna().any(axis=1)]

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
                proposal, target[~target_outside_proposal], min_samples=10
            )
            res.loc[sampled_proposal_index, "sampled"] = False
            res.loc[
                target_outside_proposal.index[target_outside_proposal], "sampled"
            ] = False

        # ---------------------- collect metrics and labels ---------------------- #

        metrics = []
        for m, info in METRIC_PARAMS.iterrows():
            if (m in self.exclude_features) or ("Containment" in m):
                continue
            if info["multiband"]:
                metrics.append(f"W1_W2_{m}")
            else:
                metrics.extend([f"W{i}_{m}" for i in range(1, 3)])

        target = res.loc[res.sampled, "agn"].astype(int)
        data = res.loc[res.sampled, metrics]
        ratio = 1 if self.smote else (len(target) - sum(target)) / sum(target)

        # ---------------------- train the models ---------------------- #

        n_splits = 10
        kf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self._random_state
        )
        xgb_model = xgb.XGBClassifier(
            n_jobs=self.n_cpu,
            n_estimators=self.n_estimators,
            scale_pos_weight=ratio,
            learning_rate=self.learning_rate,
        )
        if self.smote:
            pipeline_list = [SMOTE(random_state=self._random_state), xgb_model]
        else:
            pipeline_list = [xgb_model]

        pipeline = make_pipeline(*pipeline_list)
        scores = ["precision", "recall", "f1"]
        xgb_res = cross_validate(
            pipeline,
            data,
            target,
            scoring=scores,
            cv=kf,
            n_jobs=1,
            return_estimator=True,
            return_indices=True,
        )

        # ---------------------- plot individual models ---------------------- #

        individual_models_path = self._plot_path / "individual_models"
        individual_models_path.mkdir(parents=True, exist_ok=True)
        for i in range(n_splits):
            # plot importance
            est = xgb_res["estimator"][i].named_steps["xgbclassifier"]
            xgb.plot_importance(est)
            fig = plt.gcf()
            fn = individual_models_path / f"{i}_importance.pdf"
            fig.savefig(fn, bbox_inches="tight")
            plt.close("all")

            # plot confusion matrix
            test_indices = xgb_res["indices"]["test"][i]
            data_test = data.iloc[test_indices]
            target_test = target.iloc[test_indices]
            conf_disp = ConfusionMatrixDisplay.from_estimator(
                est, data_test, target_test
            )
            conf_disp.plot()
            fig = plt.gcf()
            fn = individual_models_path / f"{i}_confusion_matrix.pdf"
            fig.savefig(fn, bbox_inches="tight")
            plt.close()

            # plot error in train and test samples
            train_errors = []
            val_errors = []
            train_indices = xgb_res["indices"]["train"][i]
            target_train = target.iloc[train_indices]
            data_train = data.iloc[train_indices]

            for j in range(self.n_estimators):
                y_train = est.predict_proba(data_train, iteration_range=(1, j + 1))[
                    :, 1
                ]
                train_errors.append(mean_squared_error(target_train, y_train))
                y_test = est.predict_proba(data_test, iteration_range=(1, j + 1))[:, 1]
                val_errors.append(mean_squared_error(target_test, y_test))

            fig, ax = plt.subplots()
            x = np.arange(len(train_errors))
            ax.plot(x, train_errors, label="training sample")
            ax.plot(x, val_errors, label="test sample")
            ax.set_xlabel("Boosting iteration")
            ax.set_ylabel("MSE")
            fn = individual_models_path / f"{i}_mse.pdf"
            fig.savefig(fn, bbox_inches="tight")
            plt.close()

        # ---------------------- plot average importance ---------------------- #

        importances = pd.concat(
            [
                pd.Series(est.named_steps["xgbclassifier"].get_booster().get_score())
                for est in xgb_res["estimator"]
            ],
            axis=1,
        )
        importances_meds = importances.quantile(0.5, axis=1)
        importances_lower = importances_meds - importances.quantile(0.05, axis=1)
        importances_upper = importances.quantile(0.95, axis=1) - importances_meds
        xlabels = [get_metric_info(m)[2] for m in importances.index]

        fig, ax = plt.subplots()
        ax.errorbar(
            np.arange(len(importances_meds)),
            importances_meds,
            yerr=[importances_lower, importances_upper],
            fmt="s",
        )
        ax.set_xticks(np.arange(len(importances_meds)))
        ax.set_xticklabels(xlabels, rotation=90, ha="right")
        ax.set_ylabel("Importance")
        ax.grid(ls=":", color="grey", alpha=0.3)
        fn = self._plot_path / "importances.pdf"
        fig.savefig(fn, bbox_inches="tight")
        plt.close()

        # ---------------------- plot average error in train and test samples ---------------------- #
        train_errors = []
        test_errors = []

        for i in range(n_splits):
            train_indices = xgb_res["indices"]["train"][i]
            target_train = target.iloc[train_indices]
            data_train = data.iloc[train_indices]

            test_indices = xgb_res["indices"]["test"][i]
            target_test = target.iloc[test_indices]
            data_test = data.iloc[test_indices]

            est = xgb_res["estimator"][i]

            i_train_errors = []
            i_test_errors = []

            for j in range(self.n_estimators):
                y_train = est.predict_proba(data_train, iteration_range=(1, j + 1))[
                    :, 1
                ]
                i_train_errors.append(mean_squared_error(target_train, y_train))
                y_test = est.predict_proba(data_test, iteration_range=(1, j + 1))[:, 1]
                i_test_errors.append(mean_squared_error(target_test, y_test))

            train_errors.append(i_train_errors)
            test_errors.append(i_test_errors)

        train_errors = np.array(train_errors)
        test_errors = np.array(test_errors)

        fig, ax = plt.subplots()
        x = np.arange(train_errors.shape[1])
        for si, (s, label) in enumerate(
            zip([train_errors, test_errors], ["train", "test"])
        ):
            c = f"C{si}"
            ax.plot(x, np.median(s, axis=0), label=label, color=c)
            ax.fill_between(
                x,
                *np.quantile(s, [0.05, 0.95], axis=0),
                alpha=0.2,
                color=c,
                ec="none",
            )
        ax.set_xlabel("Boosting iteration")
        ax.set_ylabel("MSE")
        fn = self._plot_path / "mse.pdf"
        fig.savefig(fn, bbox_inches="tight")
        plt.close()

        # ---------------------- plot average recall and accuracy depending on prob ---------------------- #

        x = np.linspace(0.01, 0.95, 100)
        precisions = []
        recalls = []
        for i in range(n_splits):
            test_indices = xgb_res["indices"]["test"][i]
            target_test = target.iloc[test_indices]
            data_test = data.iloc[test_indices]

            est = xgb_res["estimator"][i]
            i_probs = est.predict_proba(data_test)[:, 1]
            precisions.append([precision_score(target_test, i_probs > ix) for ix in x])
            recalls.append([recall_score(target_test, i_probs > ix) for ix in x])

        fig, ax = plt.subplots()
        for i, (s, label) in enumerate(
            zip([precisions, recalls], ["precision", "recall"])
        ):
            color = f"C{i}"
            ax.plot(x, np.median(s, axis=0), color=color, label=label)
            ax.fill_between(
                x,
                *np.quantile(s, [0.05, 0.95], axis=0),
                alpha=0.2,
                color=color,
                ec="none",
            )
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.legend()
        fn = self._plot_path / "scores.pdf"
        fig.savefig(fn, bbox_inches="tight")
        plt.close()

        # ---------------------- drop estimators and return non-binary results ---------------------- #
        xgb_res.pop("estimator")
        xgb_res.pop("indices")
        for k, v in xgb_res.items():
            xgb_res[k] = v.tolist()
        return xgb_res
