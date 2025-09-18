from typing import List
from darelabdb.nlp_retrieval.core.models import RetrievalResult
from darelabdb.nlp_retrieval.rerankers.reranker_abc import BaseReranker

class BridgeReranker(BaseReranker):
    def rerank(
        self, nlqs: List[str], results_batch: List[List[RetrievalResult]], k: int
    ) -> List[List[RetrievalResult]]:
        final_results = []
        for result_list in results_batch:
            sorted_results = sorted(result_list, key=lambda r: r.score, reverse=True)
            final_results.append(sorted_results[:k])
        return final_results