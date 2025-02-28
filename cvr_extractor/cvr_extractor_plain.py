from darelabdb.nlp_value_linking.cvr_extractor.cvr_extractor_abc import CVRExtractorABC
from typing import List


class PlainExtractor(CVRExtractorABC):
    """Basic CVR extractor returning full input as single candidate.

    Simple implementation that treats the entire input string as
    a single potential value reference."""

    def extract_keywords(self, input_text: str) -> List[str]:
        return [input_text]
