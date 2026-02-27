import pandas as pd
import numpy as np


def match_distributions(s1: pd.Series, s2: pd.Series):
    """Use rejection sampling to down-sample s1 to match the distribution of s2"""

    assert (min(s2) > min(s1)) & (max(s2) < max(s1)), (
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


if __name__ == "__main__":
    from scipy.stats import norm, kstest
    import matplotlib.pyplot as plt

    s1 = pd.Series(norm(0, 1).rvs(10000))
    s2 = pd.Series(norm(1, 0.5).rvs(1000))
    s1_sampled = s1.drop(index=match_distributions(s1, s2))

    pval = kstest(s1_sampled, s2).pvalue

    fig, ax = plt.subplots()
    ax.hist(s2, density=True, color="C0", ec="white", alpha=0.8)
    ax.hist(s1, density=True, color="C1", alpha=1, histtype="step", ls=":")
    ax.hist(
        s1_sampled,
        density=True,
        color="C1",
        alpha=1,
        ls="-",
        histtype="step",
    )
    ax.set_title(f"P-value: {pval:.2f}")
    plt.show()
