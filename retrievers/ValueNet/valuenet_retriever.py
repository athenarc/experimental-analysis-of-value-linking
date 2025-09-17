import os
from collections import defaultdict
from typing import List, Dict

from rapidfuzz.distance import DamerauLevenshtein
from tqdm import tqdm

from darelabdb.nlp_retrieval.core.models import RetrievalResult, SearchableItem
from darelabdb.nlp_retrieval.retrievers.retriever_abc import BaseRetriever


class ValueNetRetriever(BaseRetriever):
    """
    Retrieves database values by performing an optimized in-memory similarity search.

    This retriever operates on a pre-built index file of unique database values.
    It is optimized to avoid redundant computations by first finding all unique
    candidate strings across a batch of queries, pre-calculating their matches
    against the index, and then re-assembling the results for each original query.
    """

    INDEX_FILENAME = "values_index.jsonl"

    def __init__(self, enable_tqdm: bool = True):
        """
        Initializes the retriever.

        Args:
            enable_tqdm: If True, displays a progress bar during retrieval.
        """
        self.similarity_algorithm = DamerauLevenshtein
        self.enable_tqdm = enable_tqdm

    def index(self, items: List[SearchableItem], output_path: str) -> None:
        """
        Saves a list of SearchableItems to a JSONL file.

        This method creates the "index" that the retrieve method will use.

        Args:
            items: A list of `SearchableItem` objects.
            output_path: The directory path to save the index file.
        """
        index_path = os.path.join(output_path, self.INDEX_FILENAME)
        with open(index_path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(item.model_dump_json() + "\n")
        print(f"ValueNet index created with {len(items)} items at {index_path}")

    def retrieve(
        self, processed_queries_batch: List[List[str]], output_path: str, k: int
    ) -> List[List[RetrievalResult]]:
        """
        Retrieves relevant database values from the index file using an
        optimized batch-aware approach.

        Args:
            processed_queries_batch: A list where each inner list contains the
                candidate strings for one NLQ.
            output_path: The path to the directory containing the index file.
            k: The number of candidate results to retrieve per query.

        Returns:
            A list of lists of `RetrievalResult` objects, one list per NLQ.
        """
        index_path = os.path.join(output_path, self.INDEX_FILENAME)
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found at {index_path}")

        indexed_items = []
        with open(index_path, "r", encoding="utf-8") as f:
            for line in f:
                indexed_items.append(SearchableItem.model_validate_json(line))

        # Optimization Step 1: Pre-process indexed content to avoid repeated lower() calls.
        preprocessed_items = [
            (item, str(item.content).lower()) for item in indexed_items
        ]

        # Optimization Step 2: Find all unique candidate strings across the entire batch.
        unique_candidates = set(
            candidate.lower()
            for candidates in processed_queries_batch
            for candidate in candidates
        )

        # Optimization Step 3: Pre-compute matches for each unique candidate.
        # This is the most expensive part, but it's now done only once per unique candidate.
        candidate_to_matches: Dict[str, List[RetrievalResult]] = defaultdict(list)
        for candidate_str in tqdm(
            unique_candidates,
            desc="Computing unique candidate matches",
            disable=not self.enable_tqdm,
        ):
            for item, lower_content in preprocessed_items:
                score = self.similarity_algorithm.normalized_similarity(
                    candidate_str, lower_content
                )

                if score > 0.75:
                    result_item = item.model_copy()
                    result_item.metadata["retrieved_by_keyword"] = candidate_str
                    candidate_to_matches[candidate_str].append(
                        RetrievalResult(item=result_item, score=score)
                    )

        # Optimization Step 4: Assemble the final results for each query using the pre-computed map.
        results_batch = []
        for candidates in tqdm(
            processed_queries_batch,
            desc="Assembling query results",
            disable=not self.enable_tqdm,
        ):
            query_results: Dict[str, RetrievalResult] = {}
            for candidate_str in candidates:
                # Look up the pre-computed matches for this candidate.
                matches = candidate_to_matches.get(candidate_str.lower(), [])
                for result in matches:
                    # Keep only the best-scoring match for any given database item.
                    if (
                        result.item.item_id not in query_results
                        or result.score > query_results[result.item.item_id].score
                    ):
                        query_results[result.item.item_id] = result

            sorted_results = sorted(
                query_results.values(), key=lambda r: r.score, reverse=True
            )
            results_batch.append(sorted_results[:k])

        return results_batch