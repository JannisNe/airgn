#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:                airgn/ampel/legacy_survey/LegacySurveyBrickLoader.py
# License:             BSD-3-Clause
# Author:              Jannis Necker <jannis.necker@gmail.com>
# Date:                05.11.2025
# Last Modified Date:  05.11.2025
# Last Modified By:    Jannis Necker <jannis.necker@gmail.com>
from typing import Dict, Generator
from pathlib import Path

from astropy.table import Table
import pandas as pd
from ampel.abstract.AbsAlertLoader import AbsAlertLoader

from airgn.legacy_survey.download import (
    get_filenames,
    get_local_path,
    download_file_by_index,
)

LEGACY_SURVEY_WISE_COLUMNS = [
    "LC_FLUX_W1",
    "LC_FLUX_W2",
    "LC_FLUX_IVAR_W1",
    "LC_FLUX_IVAR_W2",
    "LC_NOBS_W1",
    "LC_NOBS_W2",
    "LC_MJD_W1",
    "LC_MJD_W2",
    "LC_FRACFLUX_W1",
    "LC_FRACFLUX_W2",
    "LC_RCHISQ_W1",
    "LC_RCHISQ_W2",
    "LC_EPOCH_INDEX_W1",
    "LC_EPOCH_INDEX_W2",
]


class LegacySurveyBrickLoader(AbsAlertLoader[Dict]):
    """
    Load alerts from one of more files.
    """

    # path to timewise download config file
    file_indices: list[int] | None = None

    # download files if they do not exist locally
    download_if_missing: bool = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # get all filenames from legacy survey server
        filenames = get_filenames()
        if self.file_indices is not None:
            filenames = [filenames[i] for i in self.file_indices]

        # download files if missing
        if self.download_if_missing:
            indices = (
                range(len(filenames))
                if self.file_indices is None
                else self.file_indices
            )
            download_file_by_index(indices)

        # generate local filenames
        local_filenames = [[get_local_path(fn) for fn in pair] for pair in filenames]

        # set up processing generator
        self._gen = self.iter_lightcurves(local_filenames)

    def iter_lightcurves(
        self, filenames: list[list[Path]]
    ) -> Generator[pd.DataFrame, None, None]:
        # loop over pairs of summary and lightcurve files
        for summary_fn, lightcurve_fn in filenames:
            summary_table = (
                Table.read(summary_fn, format="fits")
                .to_pandas()
                .set_index(["RELEASE", "BRICKID", "OBJID"])
            )

            for row in Table.read(lightcurve_fn, format="fits"):
                lc = {
                    col: row[col]
                    for col in LEGACY_SURVEY_WISE_COLUMNS
                    if col in row.colnames
                }
                cntr = row["RELEASE"], row["BRICKID"], row["OBJID"]
                ra = summary_table.loc[cntr, "RA"]
                dec = summary_table.loc[cntr, "DEC"]
                lc = pd.DataFrame(lc).assign(
                    release=cntr[0], brickid=cntr[1], objid=cntr[2], ra=ra, dec=dec
                )
                yield lc[lc["LC_MJD_W2"] > 0]  # unused entries have zeros in MJD

    def __iter__(self):
        return self

    def __next__(self) -> pd.DataFrame:  # type: ignore
        return next(self._gen)
