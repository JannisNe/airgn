import warnings

import numpy as np

from ampel.abstract.AbsLightCurveT2Unit import AbsLightCurveT2Unit
from ampel.struct.UnitResult import UnitResult
from ampel.types import UBson
from ampel.view.LightCurve import LightCurve
from ampel.timewise.util.pdutil import datapoints_to_dataframe

from timewise.process import keys
from timewise.process.stacking import FLUX_ZEROPOINTS


# AB offset to Vega for WISE bands from Jarrett et al. (2011)
# https://dx.doi.org/10.1088/0004-637X/735/2/112
WISE_AB_OFFSET = {
    "W1": 2.699,
    "W2": 3.339,
}


class T2MaggyToFluxDensity(AbsLightCurveT2Unit):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._maggy_conversion = {
            i: FLUX_ZEROPOINTS[f"w{i}"]
            * 10 ** (WISE_AB_OFFSET[f"W{i}"] / 2.5 - 9)
            * 1e3
            for i in range(1, 3)
        }

    def process(self, light_curve: LightCurve) -> UBson | UnitResult:
        columns = [
            "ra",
            "dec",
        ]
        for i in range(1, 3):
            for key in [
                "FLUX",
                "FLUX_IVAR",
                "NOBS",
                "MJD",
                "FRACFLUX",
                "RCHISQ",
                "EPOCH_INDEX",
            ]:
                columns.append(f"LC_{key}_W{i}")

        photopoints = light_curve.get_photopoints()
        if photopoints is None:
            return {}
        data, _ = datapoints_to_dataframe(photopoints, columns=columns)

        for i in range(1, 3):
            # use maggies as fluxes, not the same but that is not really important as we
            # not using the fluxes for science directly
            data[f"w{i}{keys.MEAN}{keys.FLUX_EXT}"] = data[f"LC_FLUX_W{i}"]
            data[f"w{i}{keys.FLUX_EXT}{keys.RMS}"] = np.sqrt(
                1.0 / data[f"LC_FLUX_IVAR_W{i}"]
            )

            # convert fluxes to flux densities in mJy
            data[f"w{i}{keys.MEAN}{keys.FLUX_DENSITY_EXT}"] = (
                data[f"LC_FLUX_W{i}"] * self._maggy_conversion[i]
            )
            data[f"w{i}{keys.FLUX_DENSITY_EXT}{keys.RMS}"] = (
                data[f"w{i}{keys.FLUX_EXT}{keys.RMS}"] * self._maggy_conversion[i]
            )
            data[f"w{i}{keys.FLUX_DENSITY_EXT}{keys.UPPER_LIMIT}"] = False

            # convert to magnitudes
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                data[f"w{i}{keys.MEAN}{keys.MAG_EXT}"] = (
                    22.5
                    - 2.5 * np.log10(data[f"LC_FLUX_W{i}"])
                    - WISE_AB_OFFSET[f"W{i}"]
                )

        return data.to_dict(orient="records")
