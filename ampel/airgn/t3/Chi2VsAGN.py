from collections.abc import Generator

from pymongo import MongoClient
import pandas as pd

from ampel.abstract.AbsPhotoT3Unit import AbsPhotoT3Unit
from ampel.struct.T3Store import T3Store
from ampel.types import T3Send
from ampel.view.TransientView import TransientView


class Chi2VsAGN(AbsPhotoT3Unit):
    """
    Plot lightcurves of transients using matplotlib
    """

    path: str
    input_mongo_db_name: str
    mongo_uri: str = "mongodb://localhost:27017"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._client = MongoClient(self.mongo_uri)
        self._col = self._client[self.input_mongo_db_name]["input"]

    def process(
        self, gen: Generator[TransientView, T3Send, None], t3s: None | T3Store = None
    ) -> None:
        res = {}
        for view in gen:
            input_res = None
            for t2 in view.get_t2_views("T2CalculateChi2Stacked"):
                input_res = self._col.find_one({"orig_id": t2.stock})
                break
            if not input_res:
                continue
            ires = dict(view.get_latest_t2_body("T2CalculateChi2Stacked"))
            ires.update(input_res)
            res[view.stock["stock"]] = ires

        res = pd.DataFrame(res)
