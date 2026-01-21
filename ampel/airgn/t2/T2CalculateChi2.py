from ampel.abstract.AbsLightCurveT2Unit import AbsLightCurveT2Unit
from ampel.struct.UnitResult import UnitResult
from ampel.types import UBson
from ampel.view.LightCurve import LightCurve

from ampel.timewise.util.pdutil import datapoints_to_dataframe
from ampel.airgn.t2.T2CalculateVarMetrics import red_chi2, npoints

from timewise.process import keys


class T2CalculateChi2(AbsLightCurveT2Unit):
    def process(self, light_curve: LightCurve) -> UBson | UnitResult:
        columns = [
            "ra",
            "dec",
            "mjd",
        ]
        for i in range(1, 3):
            for key in [keys.MAG_EXT, keys.FLUX_EXT]:
                columns.extend([f"w{i}{key}", f"w{i}{keys.ERROR_EXT}{key}"])

        photopoints = light_curve.get_photopoints()
        if photopoints is None:
            return {}
        data, _ = datapoints_to_dataframe(photopoints, columns=columns)

        res = {}
        for i in range(1, 3):
            nan_msak = (
                data[f"w{i}{keys.FLUX_EXT}"].notna()
                | data[f"w{i}{keys.ERROR_EXT}{keys.FLUX_EXT}"].notna()
            )
            data = data[nan_msak]
            f = data[f"w{i}{keys.FLUX_EXT}"]
            fe = data[f"w{i}{keys.ERROR_EXT}{keys.FLUX_EXT}"]
            t = data["mjd"]
            res[f"npoints_w{i}_{keys.FLUX_EXT}"] = npoints(f, fe, t)
            res[f"red_chi2_w{i}_{keys.FLUX_EXT}"] = red_chi2(f, fe, t)

        return res
