from pathlib import Path
from typing import Dict

import pandas as pd


def load_uci_id_mappings(file_path: str | Path) -> Dict[str, Dict[int, str]]:
    """
    Load the UCI IDS_mapping.csv file and separate its three
    lookup sections into individual dictionaries.

    Returns:
        {
            "admission_type": {id: description},
            "discharge_disposition": {id: description},
            "admission_source": {id: description},
        }
    """

    mapping_df = pd.read_csv(file_path)

    discharge_header = mapping_df.index[
        mapping_df["admission_type_id"] == "discharge_disposition_id"
    ][0]

    source_header = mapping_df.index[
        mapping_df["admission_type_id"] == "admission_source_id"
    ][0]

    admission_type_df = mapping_df.iloc[:discharge_header].copy()

    discharge_df = mapping_df.iloc[
        discharge_header + 1 : source_header
    ].copy()

    admission_source_df = mapping_df.iloc[
        source_header + 1 :
    ].copy()

    admission_types = _section_to_dict(admission_type_df)
    discharge_dispositions = _section_to_dict(discharge_df)
    admission_sources = _section_to_dict(admission_source_df)

    return {
        "admission_type": admission_types,
        "discharge_disposition": discharge_dispositions,
        "admission_source": admission_sources,
    }


def _section_to_dict(section: pd.DataFrame) -> Dict[int, str]:
    """
    Convert one section of the UCI mapping file into:

        {id: description}
    """

    result = {}

    for _, row in section.iterrows():
        raw_id = row["admission_type_id"]
        description = row["description"]

        if pd.isna(raw_id) or pd.isna(description):
            continue

        try:
            mapping_id = int(float(raw_id))
        except (ValueError, TypeError):
            continue

        result[mapping_id] = str(description).strip()

    return result
