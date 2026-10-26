from abc import ABC, abstractmethod
from typing import List

import pandas as pd

from src.schemas.clinical import ClinicalEncounter


class BaseHospitalAdapter(ABC):
    """
    Base contract for all hospital data adapters.

    Every hospital-specific adapter must convert its source data
    into AHIP's canonical ClinicalEncounter representation.
    """

    @abstractmethod
    def validate_source(self, data: pd.DataFrame) -> bool:
        """
        Check whether the incoming dataset contains the minimum
        source columns required by this adapter.
        """
        pass

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> List[ClinicalEncounter]:
        """
        Transform hospital-specific source data into AHIP's
        canonical clinical schema.
        """
        pass
