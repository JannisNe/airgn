import logging

import requests
from astropy.table import Table
from astropy.io import fits
from timewise.util.path import expand
from tqdm import tqdm
from datetime import datetime
from mmap import MADV_SEQUENTIAL
import numpy as np


logger = logging.getLogger(__name__)

URL = "https://data.desi.lbl.gov/public/dr1/vac/dr1/lsdr9-photometry/iron/v1.1/observed-targets/targetphot-iron.fits"
BASE_DIR = expand("$AIRGNDATA/desi_targetphot")
LOCAL_FILE_PATH = BASE_DIR / "LSDR9_phot_iron_v1.1.fits"
EXTRACTED_FILE_PATH = LOCAL_FILE_PATH.parent / (
    LOCAL_FILE_PATH.stem + "_extracted.fits"
)
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


def log_size(path):
    logger.debug(f"{str(path)}: {path.stat().st_size * 1e-9} GB")
    with fits.open(path, memmap=True) as hdul:
        hdul.info()


def make_extracted_file(columns: list[str]):
    logger.info("extracting LS DR9 target photometry")

    logger.debug("making Primary HDU")
    chunk_size = 100_000
    tmp_path = EXTRACTED_FILE_PATH.parent / (EXTRACTED_FILE_PATH.name + ".tmp")

    # Make some header entries for important information
    primary_header_cards = [
        ("DATE", datetime.now().strftime("%Y-%m-%d"), "Creation date"),
        ("CREATOR", "Jannis Necker", "Who created this file"),
    ]

    # Build the Primary HDU object and put it in an HDU list
    primary_hdu = fits.PrimaryHDU(header=fits.Header(primary_header_cards))
    hdu_list = fits.HDUList([primary_hdu])

    # Write the HDU list to file
    logger.debug(f"writing {tmp_path}")
    hdu_list.writeto(tmp_path, overwrite=True)
    log_size(tmp_path)

    logger.debug(f"reading {LOCAL_FILE_PATH}")
    with fits.open(LOCAL_FILE_PATH, memmap=True) as hdul:
        cols = hdul[1].columns
        nrows = hdul[1].header["NAXIS2"]

        # Build output column definitions (no data yet)
        selected_columns = []
        for c in columns:
            in_c = cols[c]
            out_c = fits.Column(
                name=in_c.name,
                format=in_c.format,
                unit=in_c.unit,
                dim=in_c.dim,
                null=in_c.null,
            )
            selected_columns.append(out_c)
        selected_coldefs = fits.ColDefs(selected_columns)

        # Create empty output file with correct structure
        logger.debug("creating output table structure")
        table_hdu = fits.BinTableHDU.from_columns(selected_coldefs, nrows=0)
        table_hdu.header["NAXIS2"] = nrows
        tablesize_in_bytes = (
            (table_hdu.header["NAXIS1"] * nrows + 2880 - 1) // 2880
        ) * 2880

        with open(tmp_path, "ab") as ff:
            ff.write(bytearray(table_hdu.header.tostring(), encoding="utf-8"))

        filelen = tmp_path.stat().st_size

        with open(tmp_path, "r+b") as ff:
            ff.seek(filelen + tablesize_in_bytes - 1)
            ff.write(b"\0")
        log_size(tmp_path)

        logger.debug("streaming rows in chunks")
        data = hdul[1].data
        with fits.open(tmp_path, mode="update", memmap=True) as out_hdul:
            ext_table_data = out_hdul[1].data
            mm = fits.util._get_array_mmap(ext_table_data)
            mm.madvise(MADV_SEQUENTIAL)

            for start in tqdm(range(0, nrows, chunk_size), desc="writing chunks"):
                stop = min(start + chunk_size, nrows)
                logger.debug(f"processing rows {start}:{stop}")
                for col in selected_coldefs.names:
                    ext_table_data[start:stop][col] = data[start:stop][col]

            logger.debug("done")

    logger.debug(f"writing {EXTRACTED_FILE_PATH}")
    tmp_path.replace(EXTRACTED_FILE_PATH)
    logger.info("done")


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
    logging.basicConfig(level=logging.DEBUG)

    make()

    logger.info("validating extraction")
    logger.debug(f"reading {LOCAL_FILE_PATH}")
    with fits.open(LOCAL_FILE_PATH, memmap=True) as hdul_in:
        data_in = hdul_in[1].data
        logger.debug(f"reading {EXTRACTED_FILE_PATH}")
        with fits.open(EXTRACTED_FILE_PATH, memmap=True) as hdul_out:
            data_out = hdul_out[1].data

            for c in EXTRACTED_FILE_COLUMNS:
                logger.debug(f"comparing {c}")
                assert np.all(data_out[c][:100] == data_in[c][:100]), (
                    f"column {c} does not match at the beginning!"
                )
                assert np.all(data_out[c][100:] == data_in[c][100:]), (
                    f"column {c} does not match at the end!"
                )
