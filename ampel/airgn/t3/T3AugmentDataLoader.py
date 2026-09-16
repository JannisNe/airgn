from collections.abc import Iterable, Iterator

from ampel.t3.supply.load.T3SimpleDataLoader import T3SimpleDataLoader
from ampel.struct.AmpelBuffer import AmpelBuffer
from ampel.types import StockId, StrictIterable
from pymongo import MongoClient


class T3AugmentDataLoader(T3SimpleDataLoader):
    extra_resource_uri: str
    extra_resource_db_name: str

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._client = MongoClient(self.extra_resource_uri)
        self._col = self._client[self.extra_resource_db_name]["input"]

    def load(
        self, stock_ids: StockId | Iterator[StockId] | StrictIterable[StockId]
    ) -> Iterable[AmpelBuffer]:
        buffers = list(
            self.data_loader.load(
                stock_ids=stock_ids,
                directives=self.directives,
                channel=self.channel,
                codec_options=self.codec_options,
                logger=self.logger,
            )
        )
        buffer_dict = {b["stock"]["stock"]: b for b in buffers}
        for extra in self._col.find({"orig_id": {"$in": list(stock_ids)}}):
            buffer_dict[extra["orig_id"]]["extra"] = extra

        return buffers
