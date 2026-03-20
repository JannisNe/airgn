#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:                airgn/ampel/legacy_survey/LegacySurveyWISEFilter.py
# License:             BSD-3-Clause
# Author:              Jannis Necker <jannis.necker@gmail.com>
# Date:                16.09.2025
# Last Modified Date:  16.09.2025
# Last Modified By:    Jannis Necker <jannis.necker@gmail.com>
import numpy as np

from ampel.abstract.AbsAlertFilter import AbsAlertFilter
from ampel.protocol.AmpelAlertProtocol import AmpelAlertProtocol

from timewise.util.visits import get_visit_map
from timewise.process import keys


class LegacySurveyWISEFilter(AbsAlertFilter):
    min_exp_per_visit: int = 8
    min_detections: int = 10

    def process(self, alert: AmpelAlertProtocol) -> None | bool | int:
        # enough single exposures per visit
        min_exp_per_visit = np.min(
            [[dp[f"LC_NOBS_W{i}"] for dp in alert.datapoints for i in range(1, 3)]]
        )
        if not min_exp_per_visit >= self.min_exp_per_visit:
            self.logger.debug(None, extra={"min_exp_per_visit": min_exp_per_visit})
            return None

        # enough detections
        min_detections = np.min(
            [
                [
                    dp[f"LC_FLUX_W{i}"] * np.sqrt(dp[f"LC_FLUX_IVAR_W{i}"])
                    for dp in alert.datapoints
                ]
                for i in range(1, 3)
            ]
        )
        if not min_detections >= self.min_detections:
            self.logger.debug(None, extra={"n_detections": len(min_detections)})
            return None

        return True
