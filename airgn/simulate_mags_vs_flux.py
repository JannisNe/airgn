from scipy.stats import chi2
import matplotlib.pyplot as plt
import numpy as np


def simulate_mags_vs_flux(
    n_samples=10000,
    n_visits_per_sample=25,
    n_points_per_visit=12,
    mean_signal=20,
):
    # simulate counts from a poisson distribution
    raw_fluxes = np.random.normal(
        mean_signal,
        np.sqrt(mean_signal),
        size=(n_samples, n_visits_per_sample, n_points_per_visit),
    )

    # get flux error
    raw_flux_errors = np.full(
        (n_samples, n_visits_per_sample, n_points_per_visit), np.sqrt(mean_signal)
    )

    # check detections
    det_mask = (raw_fluxes / raw_flux_errors) > 5
    f_det = det_mask.sum() / (n_samples * n_visits_per_sample * n_points_per_visit)
    print(f"Fraction of detections: {f_det:.2%}")

    # stack exposures by visit
    fluxes = np.mean(raw_fluxes, axis=2, where=det_mask)
    flux_errors = np.sqrt(
        np.sum((raw_fluxes - fluxes[:, :, np.newaxis]) ** 2, axis=2, where=det_mask)
    ) / np.sum(det_mask, axis=2)

    # calculate chi2 per light curve based on raw fluxes
    flat_raw_fluxes = raw_fluxes.reshape(
        n_samples, n_visits_per_sample * n_points_per_visit
    )
    flat_raw_fluxes_errors = raw_flux_errors.reshape(
        n_samples, n_visits_per_sample * n_points_per_visit
    )
    mean_raw_fluxes = np.mean(flat_raw_fluxes, axis=1)
    chi2_raw_fluxes = np.nansum(
        ((flat_raw_fluxes - mean_raw_fluxes[:, np.newaxis]) / flat_raw_fluxes_errors)
        ** 2,
        axis=1,
    ) / np.sum(~np.isnan(flat_raw_fluxes), axis=1)

    # calculate raw magnitudes and magnitude errors
    mags = -2.5 * np.log10(flat_raw_fluxes)
    mag_errors = 2.5 / np.log(10) * (flat_raw_fluxes_errors / flat_raw_fluxes)

    # calculate chi2 per light curve based on fluxes only for detections
    fluxes_mean = np.nanmean(fluxes, axis=1)
    chi2_flux_det = np.nansum(
        ((fluxes - fluxes_mean[:, np.newaxis]) / flux_errors) ** 2, axis=1
    ) / np.sum(~np.isnan(fluxes), axis=1)

    # calculate chi2 per light curve based on magnitudes
    mean_mags = np.nanmean(mags, axis=0)
    chi2_mag = np.nansum(((mags - mean_mags) / mag_errors) ** 2, axis=1) / np.sum(
        ~np.isnan(mags), axis=1
    )

    # plot the distribution of chi2 values for flux and magnitude

    fig, axs = plt.subplots(1, 3, figsize=(12, 5), sharey=True, sharex=False)
    for ax, chi2_vals, title, n, xlim in zip(
        axs,
        [chi2_raw_fluxes, chi2_flux_det, chi2_mag],
        [
            "Chi2 from raw Fluxes",
            "Chi2 from stacked fluxes",
            "Chi2 from raw magnitudes",
        ],
        [
            n_visits_per_sample * n_points_per_visit,
            n_visits_per_sample,
            n_visits_per_sample * n_points_per_visit,
        ],
        [(0.5, 1.5), [0, 10], [0.5, 1.5]],
    ):
        x = np.linspace(*xlim, 100)
        chi2_exp = chi2.cdf(x, df=n - 1, scale=1 / (n - 1))
        m = ~np.isnan(chi2_vals)
        bins = np.append(np.linspace(*xlim, 50), 1e9)
        ax.hist(chi2_vals[m], bins=bins, alpha=0.7, density=True, cumulative=True)
        ax.plot(x, chi2_exp, color="red", label="Expected CDF")
        ax.set_title(title)
        ax.set_xlim(xlim)
        ax.set_xlabel("Chi2 per light curve")
        ax.set_ylabel("Count")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    simulate_mags_vs_flux()
