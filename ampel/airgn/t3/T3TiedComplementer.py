from typing import Iterable, Sequence

from ampel.abstract.AbsBufferComplement import AbsBufferComplement
from ampel.abstract.AbsT3Unit import AbsT3Unit
from ampel.core.DocBuilder import DocBuilder
from ampel.log import AmpelLogger
from ampel.model.UnitModel import UnitModel
from ampel.struct.AmpelBuffer import AmpelBuffer
from ampel.struct.T3Store import T3Store
from ampel.types import StockId
from ampel.util.hash import build_unsafe_dict_id


class T3TiedComplementer(AbsBufferComplement):
    t3_dependency: UnitModel
    t3_dependency_data_field: str
    t3_dependency_columns: list[str]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        t3unit = self.context.loader.new_safe_logical_unit(
            self.t3_dependency,
            unit_type=AbsT3Unit,
            logger=self.logger,
        )

        self._t3col = self.context.db.get_collection("t3")
        self._confid = build_unsafe_dict_id(t3unit._get_trace_content())
        res = self._t3col.find_one(
            {"confid": self._confid, "unit": self.t3_dependency.unit},
            {"_id": 1},
            sort=[("meta.run", -1)],
        )
        if res is None:
            raise RuntimeError(
                f"Could not find a T3 document with the configuration id {self._confid}!"
            )
        self._doc_id = res["_id"]

    def complement(self, it: Iterable[AmpelBuffer], t3s: T3Store) -> None:
        buffer_dict = {b["stock"]["stock"]: b for b in it}
        for res in self._t3col.aggregate(self._get_pipeline(list(buffer_dict.keys()))):
            if not res["records"]:
                continue
            for extra in res["records"]:
                stock = extra.pop("stock")
                if "extra" not in buffer_dict[stock]:
                    buffer_dict[stock]["extra"] = extra
                else:
                    buffer_dict[stock]["extra"].update(extra)

    def _get_pipeline(self, stocks: Sequence[StockId]):
        return [
            {"$match": {"_id": self._doc_id}},
            {
                "$project": {
                    "_id": 0,
                    "records": {
                        "$map": {
                            "input": {
                                "$filter": {
                                    "input": f"${self.t3_dependency_data_field}",
                                    "as": "p",
                                    "cond": {"$in": ["$$p.stock", stocks]},
                                }
                            },
                            "as": "p",
                            "in": {
                                column: f"$$p.{column}"
                                for column in self.t3_dependency_columns + ["stock"]
                            },
                        }
                    },
                }
            },
        ]
