import logging
from pathlib import Path
from pymongo import ASCENDING
import pandas as pd

from timewise.config import TimewiseConfig


CONFIG_FILE = Path(__file__).parent / "desi_agn_value_added_catalog.yml"
logger = logging.getLogger(__file__)


def import_input():
    config = TimewiseConfig.from_yaml(CONFIG_FILE)
    interface = config.build_ampel_interface()
    logger.info(
        f"importing {interface.expanded_input_csv} into {interface.input_mongo_db_name}"
    )
    col = interface.client[interface.input_mongo_db_name]["input"]

    # load the data
    df = pd.read_csv(interface.expanded_input_csv)

    # there are duplicate TARGETIDs and thus orig_ids. From the docs I did not get why:
    # https://data.desi.lbl.gov/doc/releases/dr1/vac/agnqso/
    # for now just exclude the duplicates
    m = df.orig_id.duplicated(keep=False)
    logger.info(f"Excluding {m.sum()} duplicates")
    df = df[~m]

    # create an index from stock id
    col.create_index([(interface.orig_id_key, ASCENDING)], unique=True)
    col.insert_many(df.to_dict(orient="records"))
    logger.info("done")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import_input()
