import pandas as pd
import numpy as np
import warnings


def align_legacy_survey_photometry(data: pd.DataFrame) -> pd.DataFrame:
    warnings.warn("Aligning LS photometry is not tested!")
    # The Legacy Survey forced photometry is done per band. This means
    # that it is not safe top assume that each row corresponds to the
    # same epoch across the bands.

    lc_index_cols = [f"LC_EPOCH_INDEX_W{i}" for i in range(1, 3)]
    same_index_mask = data[lc_index_cols].nunique(axis=1, dropna=False) == 1

    # in this case all indices match
    if all(same_index_mask):
        return data

    # in this case there are some non-overlapping epochs
    indices = np.unique(np.append(*[data[c].values for c in lc_index_cols]))
    aligned_data = pd.DataFrame(columns=data.columns, index=indices)
    for i in range(1, 3):
        cols = [c for c in data.columns if f"W{i}" in c.upper()]
        aligned_data.loc[data[lc_index_cols[i - 1]], cols] = data[cols].values
    return aligned_data
