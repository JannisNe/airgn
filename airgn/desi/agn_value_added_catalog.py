import logging
import yaml
import requests
from tqdm import tqdm
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from timewise.util.path import expand


logger = logging.getLogger(__name__)


URL = "https://data.desi.lbl.gov/public/dr1/vac/dr1/agnqso/v1.0/agnqso_desi.fits"
BASE_DIR = expand("$AIRGNDATA/desi_value_added_catalog")
CSV_FILE_PATH = BASE_DIR / "agnqso_desi.csv"

AGN_MASKBIT_URL = "https://data.desi.lbl.gov/public/dr1/vac/dr1/agnqso/v1.0/tutorial/agnmask.yaml"
AGN_MASKBITS_PATH = BASE_DIR / "agnqso_desi_mask.yaml"


def download():
    logger.info("downloading DESI value added catalog")
    fits_file_path = CSV_FILE_PATH.parent / (CSV_FILE_PATH.stem + ".fits")
    fits_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Download the DESI value added catalog
    with requests.get(URL, stream=True) as response:
        response.raise_for_status()
        with open(fits_file_path, "wb") as f:
            for chunk in tqdm(response.iter_content(chunk_size=int(2**20)), desc="Downloading", unit="MB", unit_scale=True):
                f.write(chunk)

    with fits.open(fits_file_path, memmap=True) as hdul:
        logger.info(f"reading DESI value added catalog from {fits_file_path}")
        # Read the data from the first HDU
        dfs = []
        for ihdul in range(1, 3):
            df = Table(hdul[ihdul].data).to_pandas().set_index("TARGETID", inplace=True)
            dfs.append(df)
            del hdul[ihdul].data

    df = pd.concat(dfs, axis=1)
    df.to_csv(CSV_FILE_PATH, index=False)
    logger.info(f"wrote to {CSV_FILE_PATH}")


def get_agn_bitmask() -> dict:
    if not AGN_MASKBITS_PATH.is_file():
        logger.info(f"downloading DESI AGN mask bitmask from {AGN_MASKBITS_PATH}")
        response = requests.get(AGN_MASKBIT_URL)
        response.raise_for_status()
        with open(AGN_MASKBITS_PATH, 'wb') as f:
            f.write(response.content)
        logger.info(f"wrote to {AGN_MASKBITS_PATH}")

    with open(AGN_MASKBITS_PATH, 'r') as f:
        agn_maskbits_info_list = yaml.safe_load(f)["OPT_UV_TYPE"]
    agn_maskbits_info_dict = {int(l[1]): l[0] for l in agn_maskbits_info_list}

    return agn_maskbits_info_dict


if __name__ == '__main__':
    logging.basicConfig(level="INFO")
    download()
