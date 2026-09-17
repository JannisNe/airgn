import logging
from pathlib import Path
from hashlib import md5

import numpy as np
import pandas as pd
import requests
import yaml
from astropy.io import fits
from matplotlib import pyplot as plt

from timewise.util.path import expand
from tqdm import tqdm

from airgn.legacy_survey.download import (
    get_downloaded_ls_brick_lightcurves,
    parse_sweep_filename,
)

logger = logging.getLogger(__name__)


URL = "https://data.desi.lbl.gov/public/dr1/vac/dr1/agnqso/v1.0/agnqso_desi.fits"
BASE_DIR = expand("$AIRGNDATA/desi_value_added_catalog")
CSV_FILE_PATH = BASE_DIR / "agnqso_desi.csv"

AGN_MASKBIT_URL = (
    "https://data.desi.lbl.gov/public/dr1/vac/dr1/agnqso/v1.0/tutorial/agnmask.yaml"
)
AGN_MASKBITS_PATH = BASE_DIR / "agnqso_desi_mask.yaml"

OBJECTS_IN_DOWNLOADED_LS_BRICKS_PATH = (
    BASE_DIR / "agnqso_desi_ls_bricks_dr{dr}_{sv}_{hash}.csv"
)

# AB offset to Vega for WISE bands from Jarrett et al. (2011)
# https://dx.doi.org/10.1088/0004-637X/735/2/112
WISE_AB_OFFSET = {
    "W1": 2.699,
    "W2": 3.339,
}


def download():
    logger.info("downloading DESI value added catalog")
    fits_file_path = CSV_FILE_PATH.parent / (CSV_FILE_PATH.stem + ".fits")
    fits_file_path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "TARGETID",
        "TARGET_RA",
        "TARGET_DEC",
        "AGN_MASKBITS",
        "OPT_UV_TYPE",
        "Z",
        "FLUX_W1",
        "FLUX_W2",
        "LS_ID",
    ]

    # Download the DESI value added catalog
    if not fits_file_path.exists():
        with requests.get(URL, stream=True) as response:
            response.raise_for_status()
            with open(fits_file_path, "wb") as f:
                for chunk in tqdm(
                    response.iter_content(chunk_size=int(2**20)),
                    desc="Downloading",
                    unit="MB",
                    unit_scale=True,
                ):
                    f.write(chunk)

    with fits.open(fits_file_path, memmap=True) as hdul:
        logger.info(f"reading DESI value added catalog from {fits_file_path}")
        # Read the data from the first HDU
        logger.debug(f"using columns {columns}")
        dtype = {"TARGETID": str, "TARGET_RA": float, "TARGET_DEC": float}
        dfs = []
        for ihdul in range(1, 3):
            df = pd.DataFrame()
            for c in columns:
                if c in hdul[ihdul].data.names:
                    df[c] = hdul[ihdul].data.field(c).astype(dtype.get(c, float))
            df.set_index("TARGETID", inplace=True)
            dfs.append(df)
            del hdul[ihdul].data

    df = pd.concat(dfs, axis=1)
    df["orig_id"] = df.index
    df["ra"] = df["TARGET_RA"]
    df["dec"] = df["TARGET_DEC"]
    df.to_csv(CSV_FILE_PATH, index=False)
    logger.info(f"wrote to {CSV_FILE_PATH}")


def get_agn_bitmask() -> dict:
    if not AGN_MASKBITS_PATH.is_file():
        logger.info(f"downloading DESI AGN mask bitmask from {AGN_MASKBITS_PATH}")
        response = requests.get(AGN_MASKBIT_URL)
        response.raise_for_status()
        with open(AGN_MASKBITS_PATH, "wb") as f:
            f.write(response.content)
        logger.info(f"wrote to {AGN_MASKBITS_PATH}")

    with open(AGN_MASKBITS_PATH, "r") as f:
        agn_maskbits_info_list = yaml.safe_load(f)

    return agn_maskbits_info_list


def get_selected_ls_brick_objects_path(dr: int, sv: int) -> Path:
    h = md5()
    h.update(f"{dr}.{sv}".encode())
    for fn in get_downloaded_ls_brick_lightcurves(dr, sv):
        h.update(fn.encode())
    return Path(
        str(OBJECTS_IN_DOWNLOADED_LS_BRICKS_PATH).format(
            dr=dr, sv=sv, hash=h.hexdigest()
        )
    )


def select_objects_in_downloaded_legacy_survey_bricks(dr: int, sv: int):
    fn = get_selected_ls_brick_objects_path(dr, sv)
    if fn.exists():
        logger.info(f"found {fn}")
        return
    logger.info(f"{fn} not found, making it now")
    downloaded_fns = get_downloaded_ls_brick_lightcurves(dr, sv)
    logger.info(f"found {len(downloaded_fns)} downloaded bricks")

    logger.info(f"reading {CSV_FILE_PATH}")
    sample = pd.read_csv(CSV_FILE_PATH)
    mask = np.zeros(len(sample), dtype=bool)
    for downloaded_fn in downloaded_fns:
        ra_range, dec_range = parse_sweep_filename(downloaded_fn)
        imask = (
            (sample["ra"] > ra_range[0])
            & (sample["ra"] < ra_range[1])
            & (sample["dec"] > dec_range[0])
            & (sample["dec"] < dec_range[1])
        )
        logger.info(f"Found {imask.sum()} objects in {downloaded_fn}")
        mask = mask | imask
    logger.info(f"Found {mask.sum()} objects in {len(downloaded_fns)} bricks")

    sample[mask].to_csv(fn, index=False)
    logger.info(f"wrote to {fn}")


def agn_bitmask_to_wise_agn(bitmask: str) -> bool:
    wise_mask_bit = 15
    if len(bitmask) < (wise_mask_bit + 1):
        return False
    return bool(int(bitmask[wise_mask_bit]))


def make_histograms():
    logger.info(f"loading {CSV_FILE_PATH}")
    data = pd.read_csv(CSV_FILE_PATH)
    data["decoded_agn_mask"] = (
        data["AGN_MASKBITS"]
        .astype(int)
        .apply(bin)
        .astype(str)
        .str.replace("0b", "")
        .apply(lambda x: x[::-1])
    )
    agn_mask = ~(data["decoded_agn_mask"] == "0")
    wise_agn_mask = data["decoded_agn_mask"].apply(agn_bitmask_to_wise_agn)
    non_wise_agn_mask = agn_mask & ~wise_agn_mask

    for i in range(1, 3):
        data[f"W{i}mag"] = (
            22.5 - 2.5 * np.log10(data[f"FLUX_W{i}"]) - WISE_AB_OFFSET[f"W{i}"]
        )

    labels = ["non AGN", "WISE AGN", "non-WISE AGN"]
    masks = [~agn_mask, wise_agn_mask, non_wise_agn_mask]
    colors = ["C0", "C1", "C2"]
    ls = [":", "--", "-"]

    keys = ["Z", "W1mag", "W2mag"]
    xlabels = ["$z$", r"$m_\mathrm{W1}$", r"$m_\mathrm{W2}$"]
    key_masks = [
        np.ones(len(data), dtype=bool),
        ~np.isinf(data["W1mag"]) & ~data["W1mag"].isna(),
        ~np.isinf(data["W2mag"]) & ~data["W2mag"].isna(),
    ]
    xlim = [(0, 4), (10, 25), (10, 25)]

    plt.style.use("airgn.paper")
    fs = plt.rcParams["figure.figsize"]
    fig, axs = plt.subplots(ncols=len(keys), figsize=(fs[0] * 2, fs[1]))

    for k, xl, km, lim, ax in zip(keys, xlabels, key_masks, xlim, axs):
        for mask, c, ils, label in zip(masks, colors, ls, labels):
            if any(~km):
                perc = (mask & ~km).sum() / mask.sum() * 100
                logger.info(f"{k}: {perc:.2f}% of {label} missing")
            ax.hist(
                data.loc[mask & km, k],
                color=c,
                ls=ils,
                label=label if k == keys[0] else "",
                density=True,
                alpha=1,
                histtype="step",
                bins=20,
            )
        ax.set_xlabel(xl)
        ax.set_xlim(lim)

    axs[0].set_ylabel("density")
    fig.legend(ncols=3, loc="upper center", borderaxespad=0.0)
    fn = BASE_DIR / ("_".join(keys) + "_hist.pdf")
    logger.info(f"saving {fn}")
    fig.subplots_adjust(top=0.90, bottom=0.2)
    fig.savefig(fn)
    plt.close()


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    # download()
    # select_objects_in_downloaded_legacy_survey_bricks(dr=9, sv=0)
    make_histograms()
