import logging

import wget
import rubin_sim.maf as maf
import matplotlib.pyplot as plt

from airgn.legacy_survey.download import DATA_DIR


logger = logging.getLogger(__name__)
LSST_BASELINE_STRATEGY_URL = "https://s3df.slac.stanford.edu/data/rubin/sim-data/sims_featureScheduler_runs5.0/baseline/baseline_v5.0.0_10yrs.db"
LSST_BASELINE_STRATEGY_FILENAME = DATA_DIR / "baseline_v5.0.0_10yrs.db"
LS_SURVEY_BROCKS_URL = (
    "https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr10/survey-bricks.fits.gz"
)
LS_SURVEY_BROCKS_FILENAME = DATA_DIR / "survey_bricks.fits.gz"
PLOTS_DIR = DATA_DIR / "plots"


def tqdm_reporthook(t):
    """
    Wraps a tqdm instance so it can be used as
    urllib.request.urlretrieve reporthook.
    """
    last_block = 0

    def reporthook(block_num, block_size, total_size):
        nonlocal last_block

        if total_size > 0:
            t.total = total_size

        downloaded = block_num * block_size
        delta = downloaded - last_block
        last_block = downloaded

        t.update(delta)

    return reporthook


def download():
    for url, fn in zip(
        [LS_SURVEY_BROCKS_URL, LSST_BASELINE_STRATEGY_URL],
        [LS_SURVEY_BROCKS_FILENAME, LSST_BASELINE_STRATEGY_FILENAME],
    ):
        if not fn.exists():
            logger.info(f"Downloading {url}")
            wget.download(url, str(fn))


def estimate_ls_coverage():
    opsdb = LSST_BASELINE_STRATEGY_FILENAME
    run_name = opsdb.name.replace(".db", "")

    metric = maf.CountMetric("observationStartMJD", metric_name="NVisits")
    slicer = maf.HealpixSlicer(nside=64)
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

    plot_res = bundle.plot()
    fig = plot_res["SkyMap"]
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fn = PLOTS_DIR / f"{run_name}_ls_coverage.png"
    logger.info(f"Saving plot to {fn}")
    fig.savefig(fn, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download()
    estimate_ls_coverage()
