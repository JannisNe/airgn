import requests
import logging
from tqdm import tqdm
from astropy.table import Table
from pathlib import Path


NEDLVS_URL = "https://ned.ipac.caltech.edu/NED::LVS/fits/Current/"
NEDLVS_CSV_PATH = Path("/Users/jannisnecker/airgn_data/nedlvs.csv")

logger = logging.getLogger(__name__)


def download():
    logger.info("downloading NED-LVS")
    fits_path = NEDLVS_CSV_PATH.with_suffix(".fits")

    logger.debug(f"downloading fits from {NEDLVS_URL} to {fits_path}")
    response = requests.get(NEDLVS_URL, stream=True)
    response.raise_for_status()
    with open(fits_path, "wb") as f:
        for chunk in tqdm(
            response.iter_content(chunk_size=8192),
            desc="Downloading",
            unit="KB",
            unit_scale=True,
        ):
            f.write(chunk)

    logger.info("Converting to CSV")
    df = Table.read(fits_path).to_pandas()
    df["orig_id"] = df.index
    logger.info(f"writing to {NEDLVS_CSV_PATH}")
    df.to_csv(NEDLVS_CSV_PATH)
    fits_path.unlink()


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    download()
