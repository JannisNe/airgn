import logging

import pandas as pd
import requests
import yaml
from astropy.io import fits
from astropy.table import Table
from timewise.util.path import expand
from tqdm import tqdm

logger = logging.getLogger(__name__)

URL = "https://data.desi.lbl.gov/public/dr1/vac/dr1/lsdr9-photometry/iron/v1.1/observed-targets/targetphot-iron.fits"
BASE_DIR = expand("$AIRGNDATA/desi_targetphot")
LOCAL_FILE_PATH = BASE_DIR / "LSDR9_phot_iron_v1.1.fits"
EXTRACTED_FILE_PATH = LOCAL_FILE_PATH.parent / (LOCAL_FILE_PATH.stem + "extracted.fits")
EXTRACTED_FILE_COLUMNS = [
    "RELEASE",
    "BRICKID",
    "BRICK_OBJID",
    "TARGETID",
    "RA",
    "DEC",
    "LC_FLUX_W1",
    "LC_FLUX_W2",
    "LC_FLUX_IVAR_W1",
    "LC_FLUX_IVAR_W2",
    "LC_NOBS_W1",
    "LC_NOBS_W2",
    "LC_MJD_W1",
    "LC_MJD_W2",
    "WISEMASK_W1",
    "WISEMASK_W2",
    "PARALLAX",
    "PMDEC",
    "PMRA",
    "FLUX_W1",
    "FLUX_W2",
    "FLUX_IVAR_W1",
    "FLUX_IVAR_W2",
]


def download():
    logger.info("downloading LS DR9 target photometry")
    LOCAL_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(URL, stream=True) as response:
        response.raise_for_status()
        with open(LOCAL_FILE_PATH, "wb") as f:
            for chunk in tqdm(
                response.iter_content(chunk_size=int(2**20)),
                desc="Downloading",
                unit="MB",
                unit_scale=True,
            ):
                f.write(chunk)


def make_extracted_file(columns: list[str]):
    logger.info("extracting LS DR9 target photometry")
    Table.read(LOCAL_FILE_PATH)[columns].write(EXTRACTED_FILE_PATH)


def make(columns: list[str] = EXTRACTED_FILE_COLUMNS):
    if not LOCAL_FILE_PATH.exists():
        download()
    else:
        logger.info(f"{LOCAL_FILE_PATH} already exists")
    if not EXTRACTED_FILE_PATH.exists():
        make_extracted_file(columns)
    else:
        logger.info(f"{EXTRACTED_FILE_PATH} already exists")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    make()
