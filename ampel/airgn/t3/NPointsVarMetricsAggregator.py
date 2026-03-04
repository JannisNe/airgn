from typing import Generator
import pandas as pd
from pymongo import MongoClient

from ampel.util.NPointsIterator import NPointsIterator
from ampel.types import T3Send
from ampel.view.TransientView import TransientView


class NPointsVarMetricsAggregator(NPointsIterator):
    # stop after iter_max iterations
    iter_max: int | None = None

    # URI to mongo db and db name to load external info
    mongo_uri: str
    input_mongo_db_name: str

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self._client = MongoClient(self.mongo_uri)
        self._col = self._client[self.input_mongo_db_name]["input"]

    def aggregate_results(
        self, gen: Generator[TransientView, T3Send, None]
    ) -> pd.DataFrame:
        res = {}
        n_iter = 0
        for view in gen:
            input_res = None
            body = None
            for t2 in view.get_t2_views("T2FeetsTimewise", code=0):
                body = dict(t2.get_payload())
                input_res = self._col.find_one({"orig_id": t2.stock})
                break

            if not input_res:
                continue

            if not body:
                continue

            # do not include objects that are outside the specified range of npoint bins
            if not all([self.is_in_range(body[c]) for c in self.n_point_cols]):
                continue

            body.update(input_res)
            mask = str(bin(int(input_res["AGN_MASKBITS"]))).replace("0b", "")[::-1]
            body["decoded_agn_mask"] = mask
            res[view.stock["stock"]] = body

            n_iter += 1
            if self.iter_max and n_iter >= self.iter_max:
                if hasattr(self, "logger"):
                    self.logger.info("iteration limit reached, stopping loop")
                break

        return pd.DataFrame.from_dict(res, orient="index")
