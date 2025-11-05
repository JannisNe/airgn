from typing import Dict, Generator

import numpy as np
import pandas as pd
from ampel.abstract.AbsAlertLoader import AbsAlertLoader

from timewise.process import keys


class TimewiseAlertSimulator(AbsAlertLoader[Dict]):
    n_sample: int
    n_visits: int
    n_detections_per_visit: int

    mean_range: tuple[float, float] = (10.0, 20.0)
    zeropoint: float = 25.0
    zeropoint_scatter: float = 0.1

    def simulate_alerts(self) -> Generator[pd.DataFrame, None, None]:
        n_dps = self.n_visits * self.n_detections_per_visit

        # draw MJDs for all visits
        mid_visit_mjd = np.linspace(56000, 60500, self.n_visits, endpoint=True)
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
                mean = rnd.uniform(*self.mean_range)
                error = np.sqrt(mean)
                f = rnd.normal(loc=mean, scale=error, size=n_dps)
                fe = np.full(n_dps, error)
                zp = rnd.normal(
                    loc=self.zeropoint,
                    scale=self.zeropoint_scatter * self.zeropoint,
                    size=n_dps,
                )
                df[f"w{i}{keys.FLUX_EXT}"] = f
                df[f"w{i}{keys.ERROR_EXT}{keys.FLUX_EXT}"] = fe
                df[f"w{i}{keys.MAG_EXT}"] = zp - 2.5 * np.log10(f)
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
