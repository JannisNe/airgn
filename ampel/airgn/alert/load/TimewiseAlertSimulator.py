from typing import Dict, Generator

import numpy as np
import pandas as pd
from ampel.abstract.AbsAlertLoader import AbsAlertLoader

from timewise.process import keys
from timewise.process.stacking import MAGNITUDE_ZEROPOINTS


class TimewiseAlertSimulator(AbsAlertLoader[Dict]):
    n_sample: int
    n_visits: int
    n_detections_per_visit: int

    mag_range: tuple[float, float] = (16.0, 13.0)

    zeropoint_w1: float = MAGNITUDE_ZEROPOINTS["w1"]
    zeropoint_scatter_w1: float = 0.032
    zeropoint_w2: float = MAGNITUDE_ZEROPOINTS["w2"]
    zeropoint_scatter_w2: float = 0.037

    def simulate_alerts(self) -> Generator[pd.DataFrame, None, None]:
        n_dps = self.n_visits * self.n_detections_per_visit

        # draw MJDs for all visits
        start = 56000
        end = start + self.n_visits * 200
        mid_visit_mjd = np.linspace(start, end, self.n_visits, endpoint=True)
        offsets = np.linspace(-0.5, 0.5, self.n_detections_per_visit, endpoint=True)
        mjds = (mid_visit_mjd[:, None] + offsets).flatten()

        rnd = np.random.default_rng()
        for s in range(self.n_sample):
            df = pd.DataFrame()
            df["ra"] = np.full(n_dps, np.random.uniform(0, 360))
            sindec = rnd.uniform(-1, 1)
            df["dec"] = np.full(n_dps, np.degrees(np.arcsin(sindec)))
            df["mjd"] = mjds
            df["stock_id"] = np.full(n_dps, s)
            df["table_name"] = "neowiser_p1bs_psd"
            for i in range(1, 3):
                mag_mean = rnd.uniform(*sorted(self.mag_range))
                zp = rnd.normal(
                    loc=self.__getattribute__(f"zeropoint_w{i}"),
                    scale=self.__getattribute__(f"zeropoint_scatter_w{i}"),
                    size=n_dps,
                )
                flux_mean = 10 ** ((zp - mag_mean) / 2.5)
                f = rnd.poisson(flux_mean)
                fe = np.sqrt(f)
                df[f"w{i}{keys.FLUX_EXT}"] = f
                df[f"w{i}{keys.ERROR_EXT}{keys.FLUX_EXT}"] = fe
                df[f"w{i}{keys.MAG_EXT}"] = zp - np.log10(f) * 2.5
                df[f"w{i}{keys.ERROR_EXT}{keys.MAG_EXT}"] = (2.5 / np.log(10)) * (
                    fe / f
                )

            yield df

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self._gen = self.simulate_alerts()

    def __iter__(self):
        return self

    def __next__(self) -> pd.DataFrame:  # type: ignore
        return next(self._gen)
