from typing import Iterable

from ampel.abstract.AbsT3Selector import AbsT3Selector
from ampel.log import AmpelLogger

from ampel.airgn.t3.T3Tied import T3Tied


class T3TiedSelector(AbsT3Selector, T3Tied):
    t3_dependency_data_field: str
    custom: dict | None = None

    def __init__(self, logger: AmpelLogger, **kwargs):
        super().__init__(**kwargs)
        self.logger = logger
        self._doc_id = self.get_t3_doc_id(self.context, self.logger)
        self._t3col = self.context.db.get_collection("t3")

    def fetch(self) -> None | Iterable:
        pipeline = [
            {"$match": {"_id": self._doc_id}},
            {"$unwind": "$" + self.t3_dependency_data_field},
        ]

        if self.custom:
            pipeline.append(
                {
                    "$match": {
                        f"{self.t3_dependency_data_field}.{key}": value
                        for key, value in self.custom.items()
                    }
                }
            )

        pipeline.append(
            {
                "$project": {
                    "_id": 0,
                    "stock": f"${self.t3_dependency_data_field}.stock",
                }
            }
        )

        return self._t3col.aggregate(pipeline)
