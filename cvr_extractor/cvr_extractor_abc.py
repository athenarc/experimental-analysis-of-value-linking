from abc import ABC, abstractmethod
from typing import List


class CVRExtractorABC(ABC):
    """Abstract base class for Candidate Value References (CVR) extractors.

    Defines the interface for implementations that extract potential
    database value references from natural language queries (NLQs)."""

    @abstractmethod
    def extract_keywords(self, input_text: str) -> List[str]:
        """Extract candidate value references from input text.

        Args:
            input_text: Natural language query string to process

        Returns:
            List of potential candidate value references"""
        pass
