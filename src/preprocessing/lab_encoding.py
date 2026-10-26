"""Consistent encoding of HbA1c / glucose "not measured" values.

In the source data a blank HbA1c or glucose result means the test was not
performed. Training used to impute blanks to the most frequent *measured*
value (">8" and "Norm"), while serving sent the literal string "None",
which the one-hot encoder turned into an all-zero vector. Both paths now
map every "not measured" spelling to one explicit category, so the model
never sees an unmeasured result disguised as a measured one.
"""

import pandas as pd

LAB_COLUMNS = ("hba1c_result", "glucose_result")
NOT_MEASURED = "Not measured"
INPUT_MAPPING_ID = "lab-not-measured-v1"

_NOT_MEASURED_TOKENS = {"", "none", "nan", "null", "not measured"}


def normalize_lab_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with unmeasured lab results set to NOT_MEASURED.

    Measured values ("Norm", ">7", ">8", ">200", ">300") and all other
    columns are left untouched.
    """

    out = df.copy()

    for column in LAB_COLUMNS:
        if column not in out.columns:
            continue

        values = out[column].astype(object)
        unmeasured = values.isna() | (
            values.astype(str).str.strip().str.lower().isin(
                _NOT_MEASURED_TOKENS
            )
        )
        out[column] = values.where(~unmeasured, NOT_MEASURED)

    return out
