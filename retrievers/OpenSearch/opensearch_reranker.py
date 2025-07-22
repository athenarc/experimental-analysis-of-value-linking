from typing import List
from darelabdb.nlp_retrieval.core.models import RetrievalResult
from darelabdb.nlp_retrieval.rerankers.reranker_abc import BaseReranker

class OpenSearchPassthroughReranker(BaseReranker):
    """
    A simple reranker for the OpenSearch pipeline that sorts retrieved results
    by their initial similarity score and truncates the list to the top-k.

    """
    def rerank(
        self, nlqs: List[str], results_batch: List[List[RetrievalResult]], k: int
    ) -> List[List[RetrievalResult]]:
        """
        Sorts each list of results by score in descending order and truncates to k.
        """
        print("Applying passthrough reranking (sorting and truncating)...")
        final_batches = []
        for result_list in results_batch:
            # Sort by the score from the retriever
            sorted_list = sorted(result_list, key=lambda r: r.score, reverse=True)
            # Truncate to the desired size k
            final_batches.append(sorted_list[:k])
        return final_batches