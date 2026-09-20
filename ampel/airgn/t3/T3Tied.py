from ampel.abstract.AbsT3Unit import AbsT3Unit
from ampel.core.AmpelContext import AmpelContext
from ampel.log import AmpelLogger
from ampel.model.UnitModel import UnitModel
from ampel.util.hash import build_unsafe_dict_id
from bson import ObjectId


class T3Tied:
    t3_dependency: UnitModel

    def get_t3_doc_id(self, context: AmpelContext, logger: AmpelLogger) -> ObjectId:
        t3unit = context.loader.new_safe_logical_unit(
            self.t3_dependency,
            unit_type=AbsT3Unit,
            logger=logger,
        )

        t3col = context.db.get_collection("t3")
        confid = build_unsafe_dict_id(t3unit._get_trace_content())
        res = t3col.find_one(
            {"confid": confid, "unit": self.t3_dependency.unit},
            {"_id": 1},
            sort=[("meta.run", -1)],
        )
        if res is None:
            raise RuntimeError(
                f"Could not find a T3 document with the configuration id {confid}!"
            )
        return res["_id"]
