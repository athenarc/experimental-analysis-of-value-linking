from typing import List

from darelabdb.nlp_retrieval.core.models import RetrievalResult
from darelabdb.nlp_retrieval.rerankers.reranker_abc import BaseReranker
from tqdm.auto import tqdm


class OmniSQLReranker(BaseReranker):
    """
    A reranker that implements the specific filtering and sorting logic from
    the OmniSQL/csc_sql codebase.

    This component filters retrieved database values based on a substring match
    score, where the score is the ratio of the longest common substring
    (between the value and the user's question) to the length of the database
    value itself. It then sorts the surviving candidates and truncates the list.
    """

    def __init__(self, score_threshold: float = 0.85, top_k: int = 20):
        """
        Args:
            score_threshold: The minimum substring match score required to keep a result.
                             The original script uses 0.85.
            top_k: The final number of results to return after sorting.
                   The original script uses 20.
        """
        self.score_threshold = score_threshold
        self.top_k = top_k

    @staticmethod
    def _calculate_substring_match_percentage(db_value: str, question: str) -> float:
        """
        Calculates the ratio of the longest common substring to the db_value's length.
        This is a faithful reimplementation of the logic in `process_dataset.py`.
        """
        db_value_lower = db_value.lower()
        question_lower = question.lower()

        if not db_value_lower:
            return 0.0
        substrings = [
            db_value_lower[i:j]
            for i in range(len(db_value_lower))
            for j in range(i + 1, len(db_value_lower) + 1)
        ]
        
        # Safely find the max length
        max_matched_len = 0
        for sub in substrings:
            if sub in question_lower:
                if len(sub) > max_matched_len:
                    max_matched_len = len(sub)

        # Normalize by the length of the database value, as in the original script.
        return max_matched_len / len(db_value_lower)

    def rerank(
        self, nlqs: List[str], results_batch: List[List[RetrievalResult]], k: int
    ) -> List[List[RetrievalResult]]:
        """
        Filters and reranks candidates using the OmniSQL substring match logic.
        The `k` parameter from the ABC is ignored in favor of the class's
        internal `top_k`.
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
            high_score_hits = []
            for res in result_list:
                new_score = self._calculate_substring_match_percentage(
                    db_value=res.item.content, question=nlq
                )
                if new_score > self.score_threshold:
                    high_score_hits.append(
                        {
                            "result": res,
                            "score": new_score,
                            "value_len": len(res.item.content),
                        }
                    )
            sorted_hits = sorted(
                high_score_hits,
                key=lambda x: (x["score"], x["value_len"]),
                reverse=True,
            )
            final_results = [hit["result"] for hit in sorted_hits[:self.top_k]]
            final_batches.append(final_results)

        return final_batches