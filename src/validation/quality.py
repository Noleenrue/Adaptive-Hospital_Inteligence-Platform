from collections import Counter
from dataclasses import dataclass
from typing import List

from src.schemas.clinical import ClinicalEncounter
from src.validation.clinical import ClinicalEncounterValidator


@dataclass
class DataQualityReport:
    total_records: int
    valid_records: int
    invalid_records: int
    records_with_warnings: int
    error_counts: dict
    warning_counts: dict


class DataQualityAnalyzer:

    def __init__(self):
        self.validator = ClinicalEncounterValidator()

    def analyze(
        self,
        encounters: List[ClinicalEncounter]
    ) -> DataQualityReport:

        total_records = len(encounters)

        valid_records = 0
        invalid_records = 0
        records_with_warnings = 0

        errors = Counter()
        warnings = Counter()

        for encounter in encounters:

            result = self.validator.validate(encounter)

            if result.is_valid:
                valid_records += 1
            else:
                invalid_records += 1

            if result.warnings:
                records_with_warnings += 1

            errors.update(result.errors)
            warnings.update(result.warnings)

        return DataQualityReport(
            total_records=total_records,
            valid_records=valid_records,
            invalid_records=invalid_records,
            records_with_warnings=records_with_warnings,
            error_counts=dict(errors),
            warning_counts=dict(warnings),
        )
