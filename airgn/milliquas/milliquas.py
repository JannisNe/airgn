import logging
import requests
from tqdm import tqdm
import gzip
import shutil
from astropy.table import Table

from timewise.util.path import expand


logger = logging.getLogger(__name__)

# https://cdsarc.cds.unistra.fr/viz-bin/cat/VII/294

MILLIQUAS_CSV_PATH = expand("$AIRGNDATA/milliquas.csv")
REFERENCE = ""
README_URL = "https://cdsarc.cds.unistra.fr/ftp/VII/294/ReadMe"
TABLE_URL = "https://cdsarc.cds.unistra.fr/ftp/VII/294/catalog.dat.gz"


def download():
    logger.info("downloading Milliquas catalog")
    MILLIQUAS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(TABLE_URL, stream=True)
    response.raise_for_status()
    gz_path = MILLIQUAS_CSV_PATH.parent / "catalog.dat.gz"
    unzipped_path = MILLIQUAS_CSV_PATH.parent / "catalog.dat"
    readme_path = MILLIQUAS_CSV_PATH.parent / "ReadMe"

    logger.debug(f"downloading ReadMe from {README_URL} to {readme_path}")
    readme_response = requests.get(README_URL)
    readme_response.raise_for_status()
    with open(readme_path, "w") as f:
        f.write(readme_response.text)

    logger.debug(f"downloading catalog from {TABLE_URL} to {gz_path}")
    with open(gz_path, "wb") as f:
        for chunk in tqdm(
            response.iter_content(chunk_size=8192),
            desc="Downloading",
            unit="KB",
            unit_scale=True,
        ):
            f.write(chunk)

    logger.debug(f"unzipping {gz_path} to {unzipped_path}")
    with gzip.open(gz_path, "rb") as f_in:
        with open(unzipped_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    gz_path.unlink()

    logger.debug(f"reading catalog from {unzipped_path}")
    t = Table.read(str(unzipped_path), format="ascii.cds", readme=str(readme_path))
    df = t.to_pandas()
    df["orig_id"] = df.index
    df.rename(columns={"RAdeg": "ra", "DEdeg": "dec"}, inplace=True)
    df.to_csv(MILLIQUAS_CSV_PATH)

    unzipped_path.unlink()
    readme_path.unlink()


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    download()
