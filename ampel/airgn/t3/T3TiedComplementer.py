from typing import Iterable, Sequence

from ampel.abstract.AbsBufferComplement import AbsBufferComplement
from ampel.struct.AmpelBuffer import AmpelBuffer
from ampel.struct.T3Store import T3Store
from ampel.types import StockId

from ampel.airgn.t3.T3Tied import T3Tied


class T3TiedComplementer(AbsBufferComplement, T3Tied):
    t3_dependency_data_field: str
    t3_dependency_columns: list[str]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._doc_id = self.get_t3_doc_id(self.context, self.logger)
        self._t3col = self.context.db.get_collection("t3")

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
