#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:                airgn/ampel/legacy_survey/LegacySurveyDatapointShaper.py
# License:             BSD-3-Clause
# Author:              Jannis Necker <jannis.necker@gmail.com>
# Date:                26.11.2025
# Last Modified Date:  26.11.2025
# Last Modified By:    Jannis Necker <jannis.necker@gmail.com>

from collections.abc import Iterable
from typing import Any


from ampel.base.AmpelUnit import AmpelUnit
from ampel.content.DataPoint import DataPoint
from ampel.types import StockId

from ampel.timewise.ingest.tags import tags


class LegacySurveyDatapointShaper(AmpelUnit):
    """
    This class 'shapes' datapoints in a format suitable
    to be saved into the ampel database
    """

    # JD2017 is used to define upper limits primary IDs
    JD2017: float = 2457754.5
    #: Byte width of datapoint ids
    digest_size: int = 8

    # Mandatory implementation
    def process(self, arg: Iterable[dict[str, Any]], stock: StockId) -> list[DataPoint]:
        ret_list: list[DataPoint] = []
        popitem = dict.pop

        for photo_dict in arg:
            # Photopoint
            assert photo_dict.get("candid"), "photometry points does not have 'candid'!"
            ret_list.append(
                {  # type: ignore[typeddict-item]
                    "id": photo_dict["candid"],
                    "stock": stock,
                    "tag": tags[photo_dict["table_name"]],
                    "body": photo_dict,
                }
            )

            popitem(photo_dict, "candid", None)

        return ret_list
