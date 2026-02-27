import pandas as pd
import numpy as np


def match_distributions(s1: pd.Series, s2: pd.Series):
    """Use rejection sampling to down-sample s1 to match the distribution of s2"""

    assert (min(s2) > min(s1)) or (max(s2) < max(s1)), (
        "The target function is outside the proposal function!"
    )

    s1 = s1.sort_values(ascending=True)
    s2 = s2.sort_values(ascending=True)

    # calculate CDFs
    cdf1 = np.arange(len(s1)) / len(s1)
    cdf2 = np.arange(len(s2)) / len(s2)

    # calculate PDF, aka gradients for f and g for all values pre-sampled values of g
    cdf2_interp = np.interp(s1, s2, cdf2, left=0, right=1)
    pdf1 = np.gradient(cdf1)
    pdf2 = np.gradient(cdf2_interp)
    w = pdf2 / pdf1
    m = max([max(w), 1])

    # rejection sampling of all values of g
    rng = np.random.default_rng(42)
    u = rng.uniform(0, 1, size=len(s1))
    return s1.index[w < m * u]


def repeated_matching(s1: pd.Series, s2: pd.Series, min_samples: int = 10):
    """
    Run rejection sampling multiple times to use as much of the
    proposal distribution as possible
    """

    # set up loop variables
    sampled_indices = [[]]
    n_sampled = np.inf

    # run as long as enough objects get sampled
    while n_sampled > min_samples:
        # drop already sampled values
        drop = np.concat(sampled_indices)
        i_proposal = s1.drop(index=drop)

        # run the sampling for this step
        i_exclude_indices = match_distributions(i_proposal, s2)

        # save sampled indices
        i_sampled_indices = i_proposal.index.difference(i_exclude_indices)
        sampled_indices.append(i_sampled_indices.tolist())
        n_sampled = len(i_sampled_indices)

    # return all rejected indices
    return s1.index.difference(np.concat(sampled_indices).tolist())


if __name__ == "__main__":
    from scipy.stats import norm, kstest
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors as mcolors

    s1 = pd.Series(norm(0, 4).rvs(10000))
    s2 = pd.Series(norm(2, 0.5).rvs(1000))

    sampled_indices = [[]]
    n_sampled = np.inf
    while n_sampled > 10:
        drop = np.concat(sampled_indices)
        i_proposal = s1.drop(index=drop)
        i_exclude_indices = match_distributions(i_proposal, s2)
        i_sampled_indices = i_proposal.index.difference(i_exclude_indices)
        sampled_indices.append(i_sampled_indices)
        n_sampled = len(i_sampled_indices)
        print(n_sampled)

    s1_sampled = s1.loc[np.concat(sampled_indices)]
    s1_sampled_multi = s1.drop(index=repeated_matching(s1, s2, 10))

    pval = kstest(s1_sampled, s2).pvalue

    fig, ax = plt.subplots()
    ax.hist(s2, density=True, color="C0", ec="white", alpha=0.8)
    cmap = plt.get_cmap("viridis")
    for i in range(len(sampled_indices) + 1):
        excl = np.concat(sampled_indices[: i + 1])
        c = cmap(i / len(sampled_indices))
        ax.hist(
            s1.drop(index=excl),
            density=True,
            color=c,
            alpha=1,
            histtype="step",
            ls="-",
        )
    ax.hist(
        s1_sampled,
        density=True,
        color="C1",
        alpha=1,
        ls="-",
        histtype="step",
    )
    ax.hist(
        s1_sampled_multi,
        density=True,
        color="C1",
        alpha=0.5,
        ls="-",
        histtype="step",
        lw=4,
    )
    ax.set_title(f"P-value: {pval:.2f}")
    sm = cm.ScalarMappable(
        norm=mcolors.Normalize(vmin=0, vmax=len(sampled_indices)), cmap=cmap
    )
    fig.colorbar(
        mappable=sm,
        ax=ax,
    )
    plt.show()
