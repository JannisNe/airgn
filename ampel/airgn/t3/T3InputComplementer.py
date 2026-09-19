from collections.abc import Iterable

from ampel.abstract.AbsBufferComplement import AbsBufferComplement
from ampel.struct.AmpelBuffer import AmpelBuffer
from ampel.struct.T3Store import T3Store
from pymongo import MongoClient


class T3InputComplementer(AbsBufferComplement):
    db_name: str
    col_name: str

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._col = MongoClient(self.context.config.get("resource.mongo"))[
            self.db_name
        ][self.col_name]

    def complement(self, it: Iterable[AmpelBuffer], t3s: T3Store) -> None:
        buffer_dict = {b["stock"]["stock"]: b for b in it}
        for extra in self._col.find({"orig_id": {"$in": list(buffer_dict.keys())}}):
            buffer_dict[extra["orig_id"]]["extra"] = extra
