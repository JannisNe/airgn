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

import numpy as np
from astropy.utils.exceptions import AstropyWarning
from astropy.io import fits
from astropy.table import Table
import pandas as pd
from ampel.abstract.AbsAlertLoader import AbsAlertLoader

from timewise.config import TimewiseConfig
from timewise.util.path import expand

from airgn.legacy_survey.download import (
    get_filenames,
    get_local_path,
    download_file_by_index,
    parse_sweep_filename,
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

CACHE_FILE_EXT = ".map_cache.csv"


class LegacySurveyBrickDESITargetIDLoader(AbsAlertLoader[Dict]):
    """
    Load alerts from one of more files.
    """

    dr: int = 10
    sv: int = 1

    # Path to CSV or FITS file with columns TARGETID, RELEASE, BRICKID, OBJID, RA, DEC
    # If the columns have different names can also specify column_map
    # with e.g. {TARGETID: my_column_name}
    target_map_file: str
    column_mapping: Dict[str, str] = {}
    index_columns: list[str] = ["RELEASE", "BRICKID", "OBJID"]

    cache_dir: str

    # download files if they do not exist locally
    download_if_missing: bool = False

    # number of sweep files to go through
    iter_nfiles_max: int | None = None

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # get all filenames from legacy survey server
        self.filenames = get_filenames(dr=self.dr, sv=self.sv)

        # check if cache exists
        self._cache_dir = expand(self.cache_dir)
        self.cache_files = [
            self._cache_dir / (fns[1] + CACHE_FILE_EXT) for fns in self.filenames
        ]
        cache_missing = any([~p.exists() for p in self.cache_files])

        # make sweep file map per brick file, saving the info for all targets
        # in these files per file
        if cache_missing:
            default_columns = [
                "TARGETID",
                "RA",
                "DEC",
            ] + self.index_columns
            usecols = [self.column_mapping.get(c, c) for c in default_columns]
            target_map_file = expand(self.target_map_file)
            if target_map_file.suffix == ".fits":
                target_map = Table.read(
                    target_map_file, format="fits", character_as_bytes=False
                ).to_pandas()[usecols]
            else:
                target_map = pd.read_csv(
                    expand(self.target_map_file),
                    usecols=usecols,
                )
            target_map.rename(
                columns={self.column_mapping.get(c, c): c for c in default_columns},
                inplace=True,
            )

            for fns, cfn in zip(self.filenames, self.cache_files):
                ra_range, dec_range = parse_sweep_filename(fns[1])
                cfn.parent.mkdir(parents=True, exist_ok=True)
                target_map.loc[
                    (target_map["RA"] > ra_range[0])
                    & (target_map["RA"] < ra_range[1])
                    & (target_map["DEC"] > dec_range[0])
                    & (target_map["DEC"] < dec_range[1])
                ].to_csv(cfn, index=False)

        # set up processing generator
        self._gen = self.iter_lightcurves()

    def iter_lightcurves(self) -> Generator[pd.DataFrame, None, None]:
        # count processed bricks files
        iter_nfiles = 0

        # use str when loading to prevent precision errors for large ints
        index_dtype = {c: str for c in self.index_columns}

        # loop over cahce files per bricks file
        for i, cache_file in enumerate(self.cache_files):
            # load cache, index by LS index columns
            cache = (
                pd.read_csv(
                    cache_file,
                    index_col=0,
                    dtype={"TARGETID": str, **index_dtype},
                )
                .set_index(self.index_columns)
                .sort_index()
            )

            # parse correspodning lightcurve filename
            lightcurve_fn = (
                str(cache_file)
                .replace(str(self._cache_dir), "")
                .replace(CACHE_FILE_EXT, "")
                .strip("/")
            )

            # if there are no objects in this bricks file, skip
            if len(cache) == 0:
                self.logger.info(
                    f"Skipping {lightcurve_fn} because no associated objects"
                )
                continue

            # check if file is downloaded
            local_fn = get_local_path(lightcurve_fn, dr=self.dr)
            if not local_fn.exists():
                self.logger.info(f"{local_fn} not found")
                if self.download_if_missing:
                    download_file_by_index(
                        i, dr=self.dr, sv=self.sv, only_lightcurves=True
                    )
                else:
                    self.logger.info("Skipping")
                    continue

            # load cahce index into array, convert to integer
            np_index_dtype = [(c, ">i4") for c in self.index_columns]
            cache_index = np.array(cache.index.tolist(), dtype=np_index_dtype)

            # load bricks file, use memmap to lazy load
            with fits.open(local_fn, memmap=True) as hdul:
                data = hdul[1].data
                index_data = [data[c] for c in self.index_columns]
                fits_array = np.rec.fromarrays(index_data, dtype=np_index_dtype)

                # select objects specified by cache
                mask = np.isin(fits_array, cache_index)
                table = Table(data[mask])
                del hdul[1].data

            # make sure all objects were selected
            # ! Some objects have zeros in release, brickid and objid, i.e. no LS match?
            assert len(np.unique(cache_index[cache_index["RELEASE"] != 0])) == len(
                table
            )

            for row in table:
                cntr = tuple(row[c] for c in self.index_columns)

                # get parent sample info
                cache_info = cache.loc[cntr]

                lc = {
                    col: row[col]
                    .byteswap()
                    .view(row[col].dtype.newbyteorder("="))  # convert to native endian
                    for col in LEGACY_SURVEY_WISE_COLUMNS
                    if col in row.colnames
                }

                lc = pd.DataFrame(lc).assign(
                    targetid=cache_info.loc["TARGETID"],
                    ra=cache_info.loc["RA"],
                    dec=cache_info.loc["DEC"],
                )
                yield lc[lc["LC_MJD_W2"] > 0]  # unused entries have zeros in MJD

            iter_nfiles += 1

            if self.iter_nfiles_max is not None and iter_nfiles >= self.iter_nfiles_max:
                self.logger.info(f"Reached {self.iter_nfiles_max} iterations, stopping")
                break

    def __iter__(self):
        return self

    def __next__(self) -> pd.DataFrame:  # type: ignore
        return next(self._gen)
