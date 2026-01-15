import logging
from pathlib import Path
import os

import wget
import rubin_sim.maf as maf
from rubin_sim.data import get_baseline
from astropy.io import fits
import numpy as np
import healpy as hp
from ligo.skymap import plot
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from airgn.legacy_survey.download import BASE_DATA_DIR

logger = logging.getLogger(__name__)
DATA_DIR = BASE_DATA_DIR / "lsst_coverage"
LS_SURVEY_BRICKS_URL = "https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr{DR}/south/survey-bricks-dr{DR}-south.fits.gz"
LS_SURVEY_BRICKS_FILENAME = DATA_DIR / Path(LS_SURVEY_BRICKS_URL).name
PLOTS_DIR = DATA_DIR / "plots"


def compute_ls_healpix_map(filename, nside=64, band="g"):
    logger.info("computing LS coverage map")
    npix = hp.nside2npix(nside)
    ls_map = np.zeros(npix, dtype=int)

    logger.info(f"loading {filename}")
    with fits.open(filename) as hdul:
        bricks = hdul[1].data

    # Precompute pixel centers
    pix = np.arange(npix)
    theta, phi = hp.pix2ang(nside, pix)
    ra = np.degrees(phi)
    dec = 90.0 - np.degrees(theta)

    for brick in bricks:
        in_ra = (ra >= brick["ra1"]) & (ra <= brick["ra2"])
        in_dec = (dec >= brick["dec1"]) & (dec <= brick["dec2"])
        in_mask = in_ra & in_dec
        if any(in_mask):
            ls_map[in_ra & in_dec] = np.max(
                [
                    np.full_like(ls_map[in_ra & in_dec], brick[f"galdepth_{band}"]),
                    ls_map[in_ra & in_dec],
                ]
            )

    return ls_map


def get_ls_healpix_map(nside=64, band="g", dr=9):
    formatted_base_fn = Path(str(LS_SURVEY_BRICKS_FILENAME).format(DR=dr))
    fn = formatted_base_fn.parent / (
        formatted_base_fn.stem + f"{nside}_{band}band.fits"
    )
    if not fn.exists():
        if not formatted_base_fn.exists():
            formatted_base_fn.parent.mkdir(parents=True, exist_ok=True)
            url = LS_SURVEY_BRICKS_URL.format(DR=dr)
            logger.info(f"downloading {url}")
            wget.download(url, str(formatted_base_fn))
        ls_map = compute_ls_healpix_map(formatted_base_fn, nside, band)
        logger.info(f"writing LS coverage map to {fn}")
        hp.write_map(str(fn), ls_map)
        return ls_map
    else:
        logger.info(f"reading LS coverage map from {fn}")
        return hp.read_map(fn)


def estimate_ls_coverage(dr):
    opsdb = Path(get_baseline())
    run_name = opsdb.name.replace(".db", "")

    nside = 128
    metric = maf.Coaddm5Metric()
    slicer = maf.HealpixSlicer(nside=nside)
    constraint = "night < 366"
    plot_dict = {
        "color_min": 0,
        "color_max": 1200,
        "extend": "max",
        "x_min": 0,
        "x_max": 1200,
    }
    plot_funcs = [maf.HealpixSkyMap()]
    bundle = maf.MetricBundle(
        metric,
        slicer,
        constraint,
        run_name=run_name,
        plot_dict=plot_dict,
        plot_funcs=plot_funcs,
    )

    try:
        bundle.read(bundle.file_root + ".npz")
    except OSError:
        g = maf.MetricBundleGroup({"nvisits": bundle}, str(opsdb), verbose=True)
        g.run_all()

    rubin_map = bundle.metric_values
    ls_map = get_ls_healpix_map(nside, dr=dr)

    rubin_total = np.nansum(rubin_map > 24)
    x = np.linspace(20, 30, 100)
    y = [np.nansum(rubin_map[ls_map > ix] > 24) / rubin_total for ix in x]
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(2.5 * 1.618, 2.5))
    ax.plot(x, y)
    ax.set_xlabel(r"$m_\mathrm{lim}$")
    ax.set_ylabel("LSST covered")
    fn = PLOTS_DIR / f"lsst_coverage_by_ls{dr}.pdf"
    logger.info(f"Saving plot to {fn}")
    fig.tight_layout()
    fig.savefig(fn)
    plt.close()

    fig, axs = plt.subplots(
        nrows=2, subplot_kw={"projection": "astro degrees mollweide"}
    )
    norm = mcolors.Normalize(vmin=23, vmax=28)
    cmap = "viridis"
    for ax, m, n in zip(
        axs, [rubin_map, ls_map], ["LSST coverage, 1yr", f"LS coverage DR{dr}"]
    ):
        ax.imshow_hpx(m, cmap=cmap, norm=norm)
        ax.set_title(n)

    fig.colorbar(
        ax=axs,
        location="right",
        mappable=plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        label=r"$m_\mathrm{lim}$",
        extend="min",
        pad=0.1,
    )
    fn = PLOTS_DIR / f"lsst_coverage_by_ls{dr}_skymaps.pdf"
    logger.info(f"Saving plot to {fn}")
    fig.savefig(fn)
    plt.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    estimate_ls_coverage(dr=9)
    estimate_ls_coverage(dr=10)
