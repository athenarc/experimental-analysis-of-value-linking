import re
from typing import List, Dict
from nlp_retrieval.core.models import RetrievalResult
from nlp_retrieval.rerankers.reranker_abc import BaseReranker
from tqdm.auto import tqdm  

class OpenSearchRuleBasedReranker(BaseReranker):
    """
    A reranker that applies a series of rule-based filters to
    the initial retrieval results, mirroring the logic of the original system.

    This reranker assumes the retriever has annotated each result with the
    keyword used to find it, stored in `result.item.metadata['retrieved_by_keyword']`.
    """
    def __init__(
        self,
        similarity_threshold: float = 0.8,
        score_proximity_filter: float = 0.3,
        max_candidates_per_group: int = 5,
        ignore_tables: List[str] = ["sqlite_sequence"],
        enable_tqdm: bool = True 
    ):
        """
        Initializes the rule-based reranker.

        Args:
            similarity_threshold (float): The minimum similarity score a result
                must have to be considered. Higher is better.
            score_proximity_filter (float): The maximum allowed score difference
                between a candidate and the best-scoring candidate for a given keyword.
            max_candidates_per_group (int): The number of top candidates to consider
                for each keyword group after initial sorting.
            ignore_tables (List[str]): A list of table names to exclude from results.
            enable_tqdm (bool): If True, displays a progress bar during reranking.
        """
        # FaissRetriever uses Inner Product on normalized vectors (higher is better).
        self.similarity_threshold = similarity_threshold
        self.score_proximity_filter = score_proximity_filter
        self.max_candidates_per_group = max_candidates_per_group
        self.ignore_tables = set(ignore_tables)
        self.enable_tqdm = enable_tqdm

    def _is_numeric(self, s: str) -> bool:
        """Checks if a string is purely numeric (integer or float)."""
        return bool(re.fullmatch(r"\d+\.?\d*", s))

    def _sort_candidates(self, candidates: List[RetrievalResult], keyword: str) -> List[RetrievalResult]:
        """
        Sorts candidates, prioritizing exact matches.

        Sorting criteria (higher is better):
        1. Is it an exact match? (True comes first)
        2. The retrieval score.
        """
        return sorted(
            candidates,
            key=lambda r: (
                r.item.content.strip().lower() == keyword.strip().lower(),
                r.score
            ),
            reverse=True # Higher score is better
        )

    def rerank(
        self,
        nlqs: List[str], # Kept for signature consistency, but not used directly
        results_batch: List[List[RetrievalResult]],
        k: int
    ) -> List[List[RetrievalResult]]:
        """
        Applies rule-based filtering and reranking to a batch of retrieval results.
        """
        final_batches = []

        for initial_results in tqdm(results_batch, desc="Reranking results", disable=not self.enable_tqdm):
            # Group results by the keyword that retrieved them
            keyword_groups: Dict[str, List[RetrievalResult]] = {}
            for res in initial_results:
                keyword = res.item.metadata.get("retrieved_by_keyword")
                if keyword:
                    if keyword not in keyword_groups:
                        keyword_groups[keyword] = []
                    keyword_groups[keyword].append(res)
            
            # This dictionary will hold the best candidates for the entire query
            final_candidates: Dict[str, RetrievalResult] = {}

            for keyword, candidates in keyword_groups.items():
                # Step 1: Sort candidates for the current keyword
                sorted_candidates = self._sort_candidates(candidates, keyword)

                # Step 2: Slice to consider only the top N
                top_candidates = sorted_candidates[:self.max_candidates_per_group]

                if not top_candidates:
                    continue

                # Step 3: Apply filtering logic
                best_score = top_candidates[0].score

                for candidate in top_candidates:
                    item_id = candidate.item.item_id
                    content = candidate.item.content
                    table = candidate.item.metadata.get("table", "")

                    if table in self.ignore_tables or self._is_numeric(content):
                        continue

                    # Since higher score is better now, the logic is reversed
                    score_is_high_enough = candidate.score >= self.similarity_threshold
                    score_is_close_to_best = (best_score - candidate.score) < self.score_proximity_filter

                    if score_is_high_enough and score_is_close_to_best:
                        if item_id not in final_candidates or candidate.score > final_candidates[item_id].score:
                            final_candidates[item_id] = candidate

            # Convert dict to a final sorted list
            final_sorted_list = sorted(final_candidates.values(), key=lambda r: r.score, reverse=True)
            final_batches.append(final_sorted_list)

        return final_batches