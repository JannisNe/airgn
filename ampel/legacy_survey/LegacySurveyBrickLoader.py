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
import warnings
from itertools import chain

from astropy.utils.exceptions import AstropyWarning
from astropy.table import Table
import pandas as pd
from ampel.abstract.AbsAlertLoader import AbsAlertLoader

from timewise.config import TimewiseConfig
from timewise.util.path import expand

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

    dr: int = 10
    sv: int = 1

    # path to timewise download config file
    file_indices: list[int] | None = None

    # download files if they do not exist locally
    download_if_missing: bool = False

    # iterate only over a subset of the data given by the chunks
    # defined by timewise. For this, the timewise download config file
    # is needed.
    timewise_config_file: Path | None = None
    timewise_chunks: list[int] | None = None

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # get all filenames from legacy survey server
        filenames = get_filenames(dr=self.dr, sv=self.sv)
        if self.file_indices is not None:
            filenames = [filenames[i] for i in self.file_indices]

        # download files if missing
        if self.download_if_missing:
            indices = (
                range(len(filenames))
                if self.file_indices is None
                else self.file_indices
            )
            download_file_by_index(indices, dr=self.dr, sv=self.sv)

        # generate local filenames
        local_filenames = [
            [get_local_path(fn, dr=self.dr) for fn in pair] for pair in filenames
        ]

        # filter by timewise chunks if requested
        if self.timewise_config_file is not None and self.timewise_chunks is not None:
            config = TimewiseConfig.from_yaml(expand(self.timewise_config_file))
            dl = config.download.build_downloader()
            row_indices = list(
                chain(
                    *[
                        dl.chunker.get_chunk(chunk_id=c).data.orig_id.tolist()
                        for c in self.timewise_chunks
                    ]
                )
            )
        else:
            row_indices = None

        # set up processing generator
        self._gen = self.iter_lightcurves(local_filenames, row_indices)

    def iter_lightcurves(
        self, filenames: list[list[Path]], row_indices: list[int] | None = None
    ) -> Generator[pd.DataFrame, None, None]:
        # loop over pairs of summary and lightcurve files
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AstropyWarning)
            for summary_fn, lightcurve_fn in filenames:
                summary_table = (
                    Table.read(summary_fn, format="fits")[
                        ["RELEASE", "BRICKID", "OBJID", "RA", "DEC"]
                    ]
                    .to_pandas()
                    .set_index(["RELEASE", "BRICKID", "OBJID"])
                )

                for row in Table.read(
                    lightcurve_fn, format="fits", character_as_bytes=False
                ):
                    cntr = row["RELEASE"], row["BRICKID"], row["OBJID"]

                    # if filtering by row indices, skip if not in the list
                    if (
                        row_indices is not None
                        and int("".join([str(ic) for ic in cntr])) not in row_indices
                    ):
                        continue

                    lc = {
                        col: row[col]
                        .byteswap()
                        .view(
                            row[col].dtype.newbyteorder("=")
                        )  # convert to native endian
                        for col in LEGACY_SURVEY_WISE_COLUMNS
                        if col in row.colnames
                    }
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
