from typing import Optional


def categorize_diagnosis(code: Optional[str]) -> str:
    """
    Convert an ICD-9 diagnosis code into a broad diagnostic category.

    The original code is preserved elsewhere. This function creates
    a derived feature for analytical and machine-learning use.
    """

    if code is None:
        return "Missing"

    code = str(code).strip()

    if not code or code == "?":
        return "Missing"

    # ICD-9 supplementary classifications
    if code.startswith("V"):
        return "Supplementary"

    if code.startswith("E"):
        return "External Cause"

    try:
        numeric_code = float(code)
    except ValueError:
        return "Other"

    if 250 <= numeric_code < 251:
        return "Diabetes"

    if 390 <= numeric_code <= 459:
        return "Circulatory"

    if 460 <= numeric_code <= 519:
        return "Respiratory"

    if 520 <= numeric_code <= 579:
        return "Digestive"

    if 580 <= numeric_code <= 629:
        return "Genitourinary"

    if 630 <= numeric_code <= 679:
        return "Pregnancy"

    if 680 <= numeric_code <= 709:
        return "Skin"

    if 710 <= numeric_code <= 739:
        return "Musculoskeletal"

    if 740 <= numeric_code <= 759:
        return "Congenital"

    if 760 <= numeric_code <= 779:
        return "Perinatal"

    if 780 <= numeric_code <= 799:
        return "Symptoms"

    if 800 <= numeric_code <= 999:
        return "Injury"

    if 140 <= numeric_code <= 239:
        return "Neoplasms"

    if 240 <= numeric_code <= 279:
        return "Endocrine"

    if 280 <= numeric_code <= 289:
        return "Blood"

    if 290 <= numeric_code <= 319:
        return "Mental"

    if 320 <= numeric_code <= 389:
        return "Nervous"

    if 1 <= numeric_code <= 139:
        return "Infectious"

    return "Other"
