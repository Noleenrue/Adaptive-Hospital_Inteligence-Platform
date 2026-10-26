from dataclasses import dataclass
from typing import Dict, List

from src.schemas.clinical import ClinicalEncounter


@dataclass
class DataReadinessReport:
    total_records: int
    completeness: Dict[str, float]
    mapping_coverage: Dict[str, float]
    target_distribution: Dict[str, int]


class DataReadinessAnalyzer:
    """
    Evaluates whether standardized clinical data is ready
    for downstream analytics and machine-learning workflows.
    """

    def analyze(
        self,
        encounters: List[ClinicalEncounter]
    ) -> DataReadinessReport:

        total = len(encounters)

        if total == 0:
            raise ValueError("Cannot analyze an empty encounter dataset.")

        completeness = self._calculate_completeness(encounters)
        mapping_coverage = self._calculate_mapping_coverage(encounters)
        target_distribution = self._calculate_target_distribution(encounters)

        return DataReadinessReport(
            total_records=total,
            completeness=completeness,
            mapping_coverage=mapping_coverage,
            target_distribution=target_distribution,
        )

    def _calculate_completeness(
        self,
        encounters: List[ClinicalEncounter]
    ) -> Dict[str, float]:

        fields = [
            "age",
            "gender",
            "race",
            "admission_type",
            "admission_source",
            "discharge_disposition",
            "primary_diagnosis",
        ]

        total = len(encounters)
        results = {}

        for field in fields:
            present = sum(
                getattr(encounter, field) is not None
                for encounter in encounters
            )

            results[field] = round(
                (present / total) * 100,
                2
            )

        return results

    def _calculate_mapping_coverage(
        self,
        encounters: List[ClinicalEncounter]
    ) -> Dict[str, float]:

        fields = [
            "admission_type",
            "admission_source",
            "discharge_disposition",
        ]

        results = {}

        for field in fields:
            values = [
                getattr(encounter, field)
                for encounter in encounters
            ]

            supplied = [
                value for value in values
                if value is not None
            ]

            if not supplied:
                results[field] = 0.0
                continue

            mapped = sum(
                not (
                    isinstance(value, str)
                    and value.startswith("Unmapped:")
                )
                for value in supplied
            )

            results[field] = round(
                (mapped / len(supplied)) * 100,
                2
            )

        return results

    def _calculate_target_distribution(
        self,
        encounters: List[ClinicalEncounter]
    ) -> Dict[str, int]:

        readmitted = sum(
            encounter.readmitted_30_days is True
            for encounter in encounters
        )

        not_readmitted = sum(
            encounter.readmitted_30_days is False
            for encounter in encounters
        )

        return {
            "readmitted_within_30_days": readmitted,
            "not_readmitted_within_30_days": not_readmitted,
        }
