import json
import os
import subprocess
import tempfile
from typing import Dict, List

from nlp_retrieval.core.models import RetrievalResult, SearchableItem
from nlp_retrieval.retrievers.retriever_abc import BaseRetriever
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm


class PyseriniRetriever(BaseRetriever):
    """
    A sparse retriever implementation using the BM25 algorithm via Pyserini.

    """

    def __init__(self, k1: float = 0.9, b: float = 0.4, enable_tqdm: bool = True):
        """
        Initializes the Pyserini-based BM25 retriever.

        Args:
            k1: The BM25 k1 parameter. Controls term frequency saturation.
            b: The BM25 b parameter. Controls document length normalization.
            enable_tqdm: If True, displays tqdm progress bars during operations.
        """
        self.k1 = k1
        self.b = b
        self.enable_tqdm = enable_tqdm
        # Cache loaded searcher objects to avoid re-initializing
        self._searcher_cache: Dict[str, LuceneSearcher] = {}

    def index(self, items: List[SearchableItem], output_path: str) -> None:
        """
        Builds a Pyserini/Lucene index from a list of SearchableItem objects.
        """
        if os.path.exists(output_path) and os.listdir(output_path):
            print(f"Index already exists in '{output_path}'. Skipping indexing.")
            return

        # Pyserini's JsonCollection requires a directory of .jsonl files.
        # We create a temporary directory to stage the data.
        with tempfile.TemporaryDirectory() as temp_dir:
            prepared_jsonl_path = os.path.join(temp_dir, "corpus.jsonl")

            with open(prepared_jsonl_path, "w", encoding="utf-8") as outfile:
                for item in items:
                    pyserini_doc = {
                        "id": item.item_id,
                        "contents": item.content,
                        **item.metadata,
                    }
                    outfile.write(json.dumps(pyserini_doc, ensure_ascii=False) + "\n")

            num_threads = os.cpu_count() or 1
            cmd = [
                "python",
                "-m",
                "pyserini.index.lucene",
                "--collection",
                "JsonCollection",
                "--input",
                temp_dir,
                "--index",
                output_path,
                "--generator",
                "DefaultLuceneDocumentGenerator",
                "--threads",
                str(num_threads),
                "--storePositions",
                "--storeDocvectors",
                "--storeRaw",
            ]

            subprocess.run(
                cmd, check=True, capture_output=True, text=True, encoding="utf-8"
            )

    def retrieve(
        self, processed_queries_batch: List[List[str]], output_path: str, k: int
    ) -> List[List[RetrievalResult]]:
        """
        Retrieves items for a batch of processed queries using Pyserini's batch search.
        """
        if not os.path.exists(output_path):
            print(f"BM25 index not found for in {output_path}. Skipping.")
            return [[] for _ in processed_queries_batch]

        if output_path not in self._searcher_cache:
            searcher = LuceneSearcher(output_path)
            searcher.set_bm25(self.k1, self.b)
            self._searcher_cache[output_path] = searcher
        searcher = self._searcher_cache[output_path]
        flat_queries = []
        query_to_original_idx_map = []
        for i, sub_queries in enumerate(processed_queries_batch):
            if not sub_queries:
                continue
            for sub_query in sub_queries:
                if sub_query and isinstance(sub_query, str):
                    flat_queries.append(sub_query)
                    query_to_original_idx_map.append(i)

        if not flat_queries:
            return [[] for _ in processed_queries_batch]

        q_ids = [f"q{i}" for i in range(len(flat_queries))]
        batch_hits = searcher.batch_search(
            queries=flat_queries,
            qids=q_ids,
            k=k,
            threads=os.cpu_count() or 1,
        )

        if not batch_hits:
            return [[] for _ in processed_queries_batch]

        aggregated_results: List[Dict[str, RetrievalResult]] = [
            {} for _ in processed_queries_batch
        ]

        pbar_desc = "Processing BM25 Results"
        for i in tqdm(
            range(len(flat_queries)), desc=pbar_desc, disable=not self.enable_tqdm
        ):
            original_nlq_idx = query_to_original_idx_map[i]
            qid = q_ids[i]
            hits = batch_hits.get(qid, [])

            if not hits:
                continue

            for hit in hits:
                full_doc = searcher.doc(hit.docid)
                raw_doc_str = full_doc.raw()

                stored_data = json.loads(raw_doc_str)

                item_id = stored_data.get("id")
                if not item_id:
                    continue

                item = SearchableItem(
                    item_id=item_id,
                    content=stored_data.get("contents", ""),
                    metadata={
                        k: v
                        for k, v in stored_data.items()
                        if k not in ["id", "contents"]
                    },
                )
                result = RetrievalResult(item=item, score=hit.score)

                if (
                    item_id not in aggregated_results[original_nlq_idx]
                    or result.score
                    > aggregated_results[original_nlq_idx][item_id].score
                ):
                    aggregated_results[original_nlq_idx][item_id] = result

        final_batches = []
        for res_dict in aggregated_results:
            sorted_res = sorted(res_dict.values(), key=lambda r: r.score, reverse=True)
            final_batches.append(sorted_res)

        return final_batches
