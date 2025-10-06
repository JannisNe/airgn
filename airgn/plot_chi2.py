from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2
from timewise.config import TimewiseConfig
from argparse import ArgumentParser


def chi2_hist(config_path: Path, outfile: Path, stacked: bool):
    cfg = TimewiseConfig.from_yaml(config_path)
    ampel_interface = cfg.build_ampel_interface()

    records = []
    unit = "T2CalculateChi2" if not stacked else "T2CalculateChi2Stacked"
    for c in ampel_interface.t2.find({"unit": unit, "code": 0}):
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


def get_parser():
    parser = ArgumentParser()
    parser.add_argument("config_file", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--stacked", action="store_true")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    chi2_hist(args.config_file, args.output, args.stacked)
