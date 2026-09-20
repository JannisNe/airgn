import numpy as np
from ampel.protocol.AmpelAlertProtocol import AmpelAlertProtocol

from ampel.timewise.t1.TimewiseFilter import TimewiseFilter

from airgn.legacy_survey.download import (
    get_downloaded_ls_brick_lightcurves,
    parse_sweep_filename,
)


class TimewiseFilterDownloadedLSBricks(TimewiseFilter):
    dr: int
    sv: int

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ranges = [
            parse_sweep_filename(fn)
            for fn in get_downloaded_ls_brick_lightcurves(self.dr, self.sv)
        ]

    def process(self, alert: AmpelAlertProtocol) -> None | bool | int:
        ra = np.array([dp["ra"] for dp in alert.datapoints])
        dec = np.array([dp["dec"] for dp in alert.datapoints])
        in_range = False
        for r in self._ranges:
            if any((ra > r[0][0]) & (ra < r[0][1]) & (dec > r[1][0]) & (dec < r[1][1])):
                in_range = True
                break

        if not in_range:
            return None

        return super().process(alert)
