import gzip
import io
import logging
from itertools import batched, islice

from astropy.table import Table, vstack
from astropy.io import ascii

import wget
from matplotlib import pyplot as plt
from timewise.util.path import expand
from zipfile import ZipFile

from tqdm import tqdm

BASE_DIR = expand("$AIRGNDATA/regalade")
README_FILE_PATH = BASE_DIR / "README.md"
DATA_FILE_PATH = BASE_DIR / "regalade.dat.gz"

README_URL = "https://cdsarc.cds.unistra.fr/ftp/J/A+A/706/A284/ReadMe"
DATA_URL = "https://cdsarc.cds.unistra.fr/ftp/J/A+A/706/A284/regalade.dat.gz"

logger = logging.getLogger(__name__)


def get(columns: list[str], chunk_size: int = 10_000) -> Table:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    if not README_FILE_PATH.exists():
        logger.info(f"Downloading {README_URL}")
        wget.download(README_URL, str(README_FILE_PATH))
    if not DATA_FILE_PATH.exists():
        logger.info(f"Downloading {DATA_URL}")
        wget.download(DATA_URL, str(DATA_FILE_PATH))

    logger.info(f"Loading {DATA_FILE_PATH}")

    tables = []
    reader = ascii.get_reader(
        ascii.Cds,
        readme=README_FILE_PATH,
    )
    reader.data.table_name = "regalade.dat"
    with (
        gzip.open(DATA_FILE_PATH, "rt", encoding="utf-8") as f,
        tqdm(total=71485705) as pbar,
    ):
        while lines := list(islice(f, chunk_size)):
            tables.append(reader.read("".join(lines))[columns])
            pbar.update(chunk_size)
    return vstack(tables)


def histograms():
    table = get(["W1mag", "W2mag"]).to_pandas()
    agn_color_mask = (table["W1mag"] - table["W2mag"]) > 0.8
    bright_wise_mask = table["W1mag"] <= 15

    logger.info(f"{agn_color_mask.sum()} / {len(agn_color_mask)} AGN")
    logger.info(f"{bright_wise_mask.sum()} / {len(bright_wise_mask)} Bright in WISE")
    logger.info(f"{~agn_color_mask & bright_wise_mask} candidate objects")

    fig, ax = plt.subplots()
    ax.hist(table["W1mag"], bins=100, ec="white", alpha=0.8)
    ax.set_xlabel(r"$m_\mathrm{W1}$")
    ax.set_ylabel("Count")
    fn = BASE_DIR / "histogram_w1.pdf"
    logger.info(f"Writing to {fn}")
    fig.savefig(fn)
    plt.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    histograms()
