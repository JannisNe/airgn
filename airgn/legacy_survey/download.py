import logging
import os
import re
import requests
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from tqdm import tqdm


logger = logging.getLogger(
    __name__ if __name__ != "__main__" else "airgn.legacy_survey.download"
)
BASE_URL = "https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr10/south/sweep/"
VERSION = "10.1"
DATA_DIR = Path(os.environ["AIRGNDATA"]) / "legacy_survey"


def get_filenames() -> list[tuple[str, str]]:
    cache_file = DATA_DIR / "legacy_survey_lc_filenames.txt"

    if not cache_file.exists():
        url = BASE_URL + VERSION + "/"
        logger.debug(f"fetching lightcurve filenames from {url}")
        response = requests.get(BASE_URL + VERSION)
        response.raise_for_status()

        filenames = sorted(
            list(
                set(
                    re.findall(
                        r"sweep-\d+[mp]\d+-\d+[mp]\d+\.fits",
                        response.content.decode(),
                    )
                )
            )
        )

        logger.debug(f"caching lightcurve filenames to {cache_file}")
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            for filename in filenames:
                f.write(f"{filename}\n")

    logger.debug(f"loading lightcurve filenames from {cache_file}")
    with open(cache_file, "r") as f:
        filenames = [line.strip() for line in f.readlines()]

    logger.debug(f"found {len(filenames)} lightcurve filenames")

    full_filenames = [
        (
            VERSION + "/" + filename,
            VERSION + "-lightcurves/" + filename.replace(".fits", "-lc.fits"),
        )
        for filename in filenames
    ]

    return full_filenames


def get_local_path(filename: str) -> Path:
    cache_dir = DATA_DIR / f"legacy_survey_DR{VERSION}"
    local_filename = cache_dir / filename
    local_filename.parent.mkdir(parents=True, exist_ok=True)
    return local_filename


def download_file_by_index(
    i: int | list[int],
) -> tuple[Path, Path] | list[tuple[Path, Path]]:
    filenames = get_filenames()
    if isinstance(i, int):
        i = [i]
    paths: list[tuple[Path, Path]] = []
    for index in i:
        if index < 0 or index >= len(filenames):
            raise IndexError("Index out of range")
        i_filenames = filenames[index]
        sub_paths: tuple[Path, Path] = tuple([get_local_path(fn) for fn in i_filenames])  # type: ignore
        for sub_filename, local_path in zip(i_filenames, sub_paths):
            if not local_path.exists():
                url = BASE_URL + sub_filename
                logger.info(f"Downloading {url} to {local_path}...")
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in tqdm(
                        response.iter_content(chunk_size=int(2**20)),
                        desc=f"Downloading {sub_filename}",
                        unit="MB",
                        unit_scale=True,
                    ):
                        f.write(chunk)

        paths.append(tuple(sub_paths))

    return paths[0] if len(paths) == 1 else paths


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Download Legacy Survey lightcurve FITS files by index."
    )
    parser.add_argument(
        "indices",
        metavar="N",
        type=int,
        nargs="+",
        help="Indices of the lightcurve files to download.",
    )
    parser.add_argument("--log-level", default="INFO", help="Set the logging level")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level.upper())
    downloaded_files = download_file_by_index(args.indices)
    for file in np.atleast_1d(downloaded_files):
        print(f"Downloaded: {file}")
