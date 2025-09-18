from darelabdb.nlp_retrieval.user_query_processors.query_processor_abc import BaseUserQueryProcessor
from typing import List

class BridgeQueryProcessor(BaseUserQueryProcessor):
    def process(self, nlqs: List[str]) -> List[List[str]]:
        return [[nlq] for nlq in nlqs]