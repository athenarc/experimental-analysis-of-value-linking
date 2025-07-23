from typing import List

from darelabdb.nlp_retrieval.core.models import RetrievalResult
from darelabdb.nlp_retrieval.rerankers.reranker_abc import BaseReranker
from tqdm.auto import tqdm


class OmniSQLReranker(BaseReranker):
    """
    A reranker that implements the substring matching logic from OmniSQL.

    This component re-scores candidates based on the length of the longest
    common substring between the candidate's content and the original query.
    It filters results below a certain threshold and returns a newly sorted list.
    """

    def __init__(self, score_threshold: float = 0.85):
        """
        Args:
            score_threshold: The minimum substring match score required to keep a result.
        """
        self.score_threshold = score_threshold

    @staticmethod
    def _calculate_substring_match_percentage(query: str, target: str) -> float:
        """
        Calculates the ratio of the longest common substring to the query length.
        """
        query_lower = query.lower()
        target_lower = target.lower()

        if not query_lower:
            return 0.0

        # This generates all substrings of the query.
        substrings = [
            query_lower[i:j]
            for i in range(len(query_lower))
            for j in range(i + 1, len(query_lower) + 1)
        ]

        max_matched_len = 0
        for sub in substrings:
            if sub in target_lower:
                if len(sub) > max_matched_len:
                    max_matched_len = len(sub)

        return max_matched_len / len(query_lower)

    def rerank(
        self, nlqs: List[str], results_batch: List[List[RetrievalResult]], k: int
    ) -> List[List[RetrievalResult]]:
        """
        Reranks candidates using the substring match scoring logic.
        """
        final_batches = []
        for nlq, result_list in tqdm(
            zip(nlqs, results_batch),
            total=len(nlqs),
            desc="Reranking with OmniSQL logic",
            disable=len(nlqs) < 5,
        ):
            if not result_list:
                final_batches.append([])
                continue

            rescored_results = []
            for res in result_list:
                new_score = self._calculate_substring_match_percentage(
                    nlq, res.item.content
                )
                if new_score > self.score_threshold:
                    rescored_results.append(
                        RetrievalResult(item=res.item, score=new_score)
                    )

            sorted_results = sorted(
                rescored_results,
                key=lambda r: (r.score, len(r.item.content)),
                reverse=True,
            )
            final_batches.append(sorted_results[:20])

        return final_batches