import os
import concurrent.futures
import functools
from collections import defaultdict
from typing import List, Dict, Tuple

from rapidfuzz.distance import DamerauLevenshtein
from tqdm import tqdm

from nlp_retrieval.core.models import RetrievalResult, SearchableItem
from nlp_retrieval.retrievers.retriever_abc import BaseRetriever


class ValueNetRetriever(BaseRetriever):
    """
    Retrieves database values by performing an optimized in-memory similarity search.

    This retriever operates on a pre-built index file of unique database values.
    It is optimized to avoid redundant computations by first finding all unique
    candidate strings across a batch of queries, pre-calculating their matches
    against the index, and then re-assembling the results for each original query.
    """

    INDEX_FILENAME = "values_index.jsonl"

    def __init__(self, enable_tqdm: bool = True, max_workers: int = None,similarity_threshold: float = 0.75):
        """
        Initializes the retriever.

        Args:
            enable_tqdm: If True, displays a progress bar during retrieval.
            max_workers: The maximum number of processes to use. If None, it
                         defaults to the number of processors on the machine.
        """
        self.similarity_algorithm = DamerauLevenshtein
        self.enable_tqdm = enable_tqdm
        self.max_workers = max_workers
        self.similarity_threshold = similarity_threshold

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

    @staticmethod
    def _compute_matches_for_candidate(
        args: Tuple[str, List[Tuple[SearchableItem, str]]],
        similarity_threshold: float
    ) -> Tuple[str, List[RetrievalResult]]:
        """Worker function to compute matches for a single candidate string."""
        candidate_str, preprocessed_items = args
        matches = []
        similarity_algorithm = DamerauLevenshtein

        for item, lower_content in preprocessed_items:
            score = similarity_algorithm.normalized_similarity(
                candidate_str, lower_content
            )

            if score >= similarity_threshold:
                result_item = item.model_copy()
                result_item.metadata["retrieved_by_keyword"] = candidate_str
                matches.append(RetrievalResult(item=result_item, score=score))

        return candidate_str, matches

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

        preprocessed_items = [
            (item, str(item.content).lower()) for item in indexed_items
        ]

        unique_candidates = set(
            candidate.lower()
            for candidates in processed_queries_batch
            for candidate in candidates
        )

        worker_func = functools.partial(
            self._compute_matches_for_candidate,
            similarity_threshold=self.similarity_threshold,
        )

        candidate_to_matches: Dict[str, List[RetrievalResult]] = defaultdict(list)
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            job_args = (
                (candidate, preprocessed_items) for candidate in unique_candidates
            )
            results_iterator = executor.map(worker_func, job_args)

            progress_bar = tqdm(
                results_iterator,
                total=len(unique_candidates),
                desc="Computing unique candidate matches",
                disable=not self.enable_tqdm,
            )

            for candidate_str, matches in progress_bar:
                if matches:
                    candidate_to_matches[candidate_str] = matches

        results_batch = []
        for candidates in tqdm(
            processed_queries_batch,
            desc="Assembling query results",
            disable=not self.enable_tqdm,
        ):
            query_results: Dict[str, RetrievalResult] = {}
            for candidate_str in candidates:
                matches = candidate_to_matches.get(candidate_str.lower(), [])
                for result in matches:
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