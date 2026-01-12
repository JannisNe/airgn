import logging

import wget
import rubin_sim.maf as maf
from astropy.io import fits
import numpy as np
import healpy as hp

from airgn.legacy_survey.download import DATA_DIR


logger = logging.getLogger(__name__)
LSST_BASELINE_STRATEGY_URL = "https://s3df.slac.stanford.edu/data/rubin/sim-data/sims_featureScheduler_runs5.0/baseline/baseline_v5.0.0_10yrs.db"
LSST_BASELINE_STRATEGY_FILENAME = DATA_DIR / "baseline_v5.0.0_10yrs.db"
LS_SURVEY_BROCKS_URL = "https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr10/south/survey-bricks-dr10-south.fits.gz"
LS_SURVEY_BROCKS_FILENAME = DATA_DIR / "south_survey_bricks.fits.gz"
PLOTS_DIR = DATA_DIR / "plots"


def download():
    for url, fn in zip(
        [LS_SURVEY_BROCKS_URL, LSST_BASELINE_STRATEGY_URL],
        [LS_SURVEY_BROCKS_FILENAME, LSST_BASELINE_STRATEGY_FILENAME],
    ):
        if not fn.exists():
            logger.info(f"Downloading {url}")
            wget.download(url, str(fn))


def compute_ls_healpix_map(nside=64, band="g"):
    logger.info("computing LS coverage map")
    npix = hp.nside2npix(nside)
    ls_map = np.zeros(npix, dtype=int)

    logger.info(f"loading {LS_SURVEY_BROCKS_FILENAME}")
    with fits.open(LS_SURVEY_BROCKS_FILENAME) as hdul:
        bricks = hdul[1].data

    # Precompute pixel centers
    pix = np.arange(npix)
    theta, phi = hp.pix2ang(nside, pix)
    ra = np.degrees(phi)
    dec = 90.0 - np.degrees(theta)

    for brick in bricks:
        in_ra = (ra >= brick["ra1"]) & (ra <= brick["ra2"])
        in_dec = (dec >= brick["dec1"]) & (dec <= brick["dec2"])
        ls_map[in_ra & in_dec] += brick[f"nexp_{band}"]

    return ls_map


def get_ls_healpix_map(nside=64, band="g"):
    fn = LS_SURVEY_BROCKS_FILENAME.parent / (
        LS_SURVEY_BROCKS_FILENAME.stem + f"{nside}_{band}band.fits"
    )
    if not fn.exists():
        ls_map = compute_ls_healpix_map(nside, band)
        logger.info(f"writing LS coverage map to {fn}")
        hp.write_map(str(fn), ls_map)
        return ls_map
    else:
        logger.info(f"reading LS coverage map from {fn}")
        return hp.read_map(fn)


def estimate_ls_coverage(min_n_exp=5):
    opsdb = LSST_BASELINE_STRATEGY_FILENAME
    run_name = opsdb.name.replace(".db", "")

    nside = 64
    metric = maf.CountMetric("observationStartMJD", metric_name="NVisits")
    slicer = maf.HealpixSlicer(nside=nside)
    constraint = None
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

    g = maf.MetricBundleGroup({"nvisits": bundle}, str(opsdb), verbose=True)
    g.run_all()

    rubin_map = bundle.metric_values
    ls_map = get_ls_healpix_map(nside)

    rubin_total = np.nansum(rubin_map)
    rubin_covered = np.nansum(rubin_map[ls_map > min_n_exp])

    coverage_fraction = rubin_covered / rubin_total
    coverage_percent = 100.0 * coverage_fraction

    logger.info(
        f"{coverage_percent:.2f}% of the Rubin LSST are observed by Legacy Survey at least {min_n_exp} times."
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download()
    estimate_ls_coverage(min_n_exp=2)
