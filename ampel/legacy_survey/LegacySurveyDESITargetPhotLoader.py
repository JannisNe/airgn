#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:                airgn/ampel/legacy_survey/LegacySurveyBrickLoader.py
# License:             BSD-3-Clause
# Author:              Jannis Necker <jannis.necker@gmail.com>
# Date:                05.11.2025
# Last Modified Date:  05.11.2025
# Last Modified By:    Jannis Necker <jannis.necker@gmail.com>
from typing import Dict, Generator

from astropy.table import Table
import pandas as pd
from ampel.abstract.AbsAlertLoader import AbsAlertLoader

from timewise.util.path import expand


LEGACY_SURVEY_WISE_COLUMNS = [
    "LC_FLUX_W1",
    "LC_FLUX_W2",
    "LC_FLUX_IVAR_W1",
    "LC_FLUX_IVAR_W2",
    "LC_NOBS_W1",
    "LC_NOBS_W2",
    "LC_MJD_W1",
    "LC_MJD_W2",
]


class LegacySurveyDESITargetPhotLoader(AbsAlertLoader[Dict]):
    """
    Load alerts from one of more files.
    """

    # path to timewise download config file
    filenames: list[str]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._filenames = [expand(fn) for fn in self.filenames]

        # set up processing generator
        self._gen = self.iter_lightcurves()

    def iter_lightcurves(
        self, row_indices: list[int] | None = None
    ) -> Generator[pd.DataFrame, None, None]:
        # loop over pairs of summary and lightcurve files
        for filename in self._filenames:
            for row in Table.read(filename, format="fits", character_as_bytes=False):
                cntr = row["RELEASE"], row["BRICKID"], row["BRICK_OBJID"]

                # if filtering by row indices, skip if not in the list
                if (
                    row_indices is not None
                    and int("".join([str(ic) for ic in cntr])) not in row_indices
                ):
                    continue

                lc = {
                    col: row[col]
                    .byteswap()
                    .view(row[col].dtype.newbyteorder("="))  # convert to native endian
                    for col in LEGACY_SURVEY_WISE_COLUMNS
                    if col in row.colnames
                }
                lc = pd.DataFrame(lc).assign(
                    release=cntr[0],
                    brickid=cntr[1],
                    objid=cntr[2],
                    ra=row["RA"],
                    dec=row["DEC"],
                )
                yield lc[lc["LC_MJD_W2"] > 0]  # unused entries have zeros in MJD

    def __iter__(self):
        return self

    def __next__(self) -> pd.DataFrame:  # type: ignore
        return next(self._gen)
