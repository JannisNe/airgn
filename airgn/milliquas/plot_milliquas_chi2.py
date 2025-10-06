from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2
from timewise.config import TimewiseConfig


def chi2_hist(config_path: Path, outfile: Path):
    cfg = TimewiseConfig.from_yaml(config_path)
    ampel_interface = cfg.build_ampel_interface()

    records = []
    for c in ampel_interface.t2.find({"unit": "T2CalculateChi2", "code": 0}):
        records.append(c["body"][0])

    df = pd.DataFrame(records)
    for i in range(1, 3):
        df[f"n_w{i}"] = (df[f"chi2_w{i}"] - df[f"npoints_w{i}"]) / np.sqrt(
            2 * df[f"npoints_w{i}"]
        )

    fig, axs = plt.subplots(nrows=2)

    for i in range(1, 3):
        ax = axs[i - 1]
        bins = list(np.linspace(0, 6, 100)) + [1e9]
        ax.hist(df[f"red_chi2_w{i}"], density=True, bins=bins, cumulative=False)
        ax.set_xlim(0, 6)
        x = np.linspace(*ax.get_xlim(), 1000)
        n = df[f"npoints_w{i}"].median()
        y = chi2(n - 1, 0, 1 / (n - 1)).pdf(x)
        ax.plot(x, y)

    fig.savefig(outfile)
    plt.close()


if __name__ == "__main__":
    MILLIQUAS_CONFIG = Path(__file__).parent / "milliquas.yml"
    chi2_hist(
        MILLIQUAS_CONFIG,
        Path("~/airgn_data/milliquas_plots/chi2.pdf").expanduser(),
    )
