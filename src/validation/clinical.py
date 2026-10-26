from dataclasses import dataclass, field
from typing import List

from src.schemas.clinical import ClinicalEncounter


@dataclass
class ValidationResult:
    """
    Represents the result of validating one clinical encounter.
    """

    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ClinicalEncounterValidator:
    """
    Validates standardized ClinicalEncounter records.

    Errors represent conditions that make a record invalid.
    Warnings represent data-quality concerns that do not
    necessarily make the record unusable.
    """

    def validate(self, encounter: ClinicalEncounter) -> ValidationResult:

        result = ValidationResult()

        # Critical identifiers
        if not encounter.patient_id:
            result.errors.append("Missing patient ID")

        if not encounter.encounter_id:
            result.errors.append("Missing encounter ID")

        # Encounter quality
        if encounter.length_of_stay is not None:
            if encounter.length_of_stay <= 0:
                result.errors.append(
                    "Length of stay must be greater than zero"
                )

        # Clinical completeness
        if encounter.admission_type is None:
            result.warnings.append(
                "Admission type is missing"
            )

        if encounter.primary_diagnosis is None:
            result.warnings.append(
                "Primary diagnosis is missing"
            )

        # Final validity
        if result.errors:
            result.is_valid = False

        return result
