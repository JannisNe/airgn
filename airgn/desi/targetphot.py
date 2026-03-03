import logging

import requests
from astropy.io import fits
from timewise.util.path import expand
from tqdm import tqdm

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


def make_extracted_file(columns: list[str]):
    logger.info("extracting LS DR9 target photometry")
    logger.debug(f"reading {LOCAL_FILE_PATH}")
    chunk_size = 100_000
    tmp_path = EXTRACTED_FILE_PATH.parent / (EXTRACTED_FILE_PATH.name + ".tmp")
    with fits.open(LOCAL_FILE_PATH, memmap=True) as hdul:
        cols = hdul[1].columns
        nrows = hdul[1].header["NAXIS2"]
        data = hdul[1].data

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
        fits.BinTableHDU.from_columns(selected_coldefs, nrows=0).writeto(
            tmp_path, overwrite=True
        )

        logger.debug("streaming rows in chunks")

        with fits.open(tmp_path, mode="append") as out_hdul:
            for start in tqdm(range(0, 200_000, chunk_size), desc="writing chunks"):
                stop = min(start + chunk_size, nrows)
                logger.debug(f"processing rows {start}:{stop}")

                chunk_arrays = [
                    data[col][start:stop]  # only this slice goes into memory
                    for col in selected_coldefs.names
                ]

                chunk_hdu = fits.BinTableHDU.from_columns(
                    [
                        fits.Column(
                            name=c.name,
                            format=c.format,
                            unit=c.unit,
                            dim=c.dim,
                            null=c.null,
                            array=chunk_arrays[i],
                        )
                        for i, c in enumerate(selected_coldefs.columns)
                    ]
                )

                logger.debug(f"appening to {tmp_path}")

                out_hdul.append(chunk_hdu)

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
    with fits.open(LOCAL_FILE_PATH, memmap=True) as hdul:
        origdata = hdul[1].data[:100].copy()
        del hdul[1].data

    logger.debug(f"reading {LOCAL_FILE_PATH}")
    with fits.open(EXTRACTED_FILE_PATH, memmap=True) as hdul:
        extdata = hdul[1].data[:100].copy()
        del hdul[1].data

    for c in EXTRACTED_FILE_COLUMNS:
        logger.debug(f"comparing {c}")
        assert origdata[c] == extdata[c], f"column {c} does not match!"
