import re
from typing import List, Dict

from tqdm import tqdm

from nlp_retrieval.core.models import RetrievalResult
from nlp_retrieval.rerankers.reranker_abc import BaseReranker


class ValueNetReranker(BaseReranker):
    """
    Reranks retrieval results using heuristics based on ValueNet's logic.

    This reranker applies different similarity score thresholds depending on the
    type of keyword that retrieved a value. For example, a numeric keyword requires
    an exact match, while a capitalized name can have a more lenient threshold.
    """

    def __init__(self, enable_tqdm: bool = True):
        """
        Initializes the reranker.

        Args:
            enable_tqdm: If True, displays a progress bar during reranking.
        """
        self.enable_tqdm = enable_tqdm

    def _get_tolerance_for_keyword(self, keyword: str) -> float:
        """
        Determines the required similarity score based on the keyword type.
        """
        keyword = str(keyword)
        if keyword.isnumeric() or re.fullmatch(r"\d+\.\d+", keyword):
            return 1.0
        if re.fullmatch(r"[A-Z-/0-9]{2,}", keyword) and not keyword.isnumeric():
            return 1.0
        if re.fullmatch(r"'.+'", keyword) or re.fullmatch(r'".+"', keyword):
            return 0.9
        if re.fullmatch(r"[A-Z][a-zA-Z0-9-/]+", keyword):
            return 0.75
        return 0.7

    def rerank(
        self,
        nlqs: List[str],
        results_batch: List[List[RetrievalResult]],
        k: int,
    ) -> List[List[RetrievalResult]]:
        """
        Applies heuristic-based filtering and reranking.

        Args:
            nlqs: The list of original natural language queries.
            results_batch: A list of candidate `RetrievalResult` objects for each NLQ.
            k: The final number of results to return for each query.

        Returns:
            A list of reranked and sorted `RetrievalResult` lists.
        """
        final_batches = []
        for initial_results in tqdm(
            results_batch, desc="ValueNet Reranking", disable=not self.enable_tqdm
        ):
            keyword_groups: Dict[str, List[RetrievalResult]] = {}
            for res in initial_results:
                keyword = res.item.metadata.get("retrieved_by_keyword")
                if keyword:
                    if keyword not in keyword_groups:
                        keyword_groups[keyword] = []
                    keyword_groups[keyword].append(res)

            final_candidates: Dict[str, RetrievalResult] = {}

            for keyword, candidates in keyword_groups.items():
                tolerance = self._get_tolerance_for_keyword(keyword)

                for candidate in candidates:
                    if candidate.score >= tolerance:
                        item_id = candidate.item.item_id
                        if (
                            item_id not in final_candidates
                            or candidate.score > final_candidates[item_id].score
                        ):
                            final_candidates[item_id] = candidate

            final_sorted_list = sorted(
                final_candidates.values(), key=lambda r: r.score, reverse=True
            )
            final_batches.append(final_sorted_list[:k])

        return final_batches