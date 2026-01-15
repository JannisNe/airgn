import logging
import warnings

import numpy as np
from astropy.table import Table
from astropy.utils.exceptions import AstropyWarning
from airgn.legacy_survey.download import download_file_by_index, get_data_dir


logger = logging.getLogger(__name__)


def make_csv_file():
    dr = 10
    sv = 1
    sweep0_summary_file = download_file_by_index(0, dr, sv)[0]
    logger.info(f"Reading {sweep0_summary_file}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        table = Table.read(sweep0_summary_file, format="fits")
    mask = (
        (table["TYPE"] != "PSF")
        & (table["PARALLAX"] == 0)
        & (table["PMRA"] == 0)
        & (table["PMDEC"] == 0)
    )
    logger.info(f"Removing {(~mask).sum()} likely stars from sweep0")
    table = table[mask]
    # DCHISQ is ,multidimensional
    table.remove_columns("DCHISQ")
    table.rename_columns(["RA", "DEC"], ["ra", "dec"])
    orig_id = np.char.add(
        np.char.add(table["RELEASE"].astype(str), table["BRICKID"].astype(str)),
        table["OBJID"].astype(str),
    ).astype(int)
    table["orig_id"] = orig_id
    csv_filename = get_data_dir(dr) / "sweep0" / f"DR{dr}_{sv}_sweep0.csv"
    logger.info(f"Writing {csv_filename}")
    csv_filename.parent.mkdir(parents=True, exist_ok=True)
    table.write(csv_filename, format="csv", overwrite=True)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    make_csv_file()
