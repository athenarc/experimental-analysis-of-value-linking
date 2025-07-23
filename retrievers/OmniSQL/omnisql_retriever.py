import json
import os
import subprocess
import tempfile
from typing import Dict, List

from darelabdb.nlp_retrieval.core.models import RetrievalResult, SearchableItem
from darelabdb.nlp_retrieval.retrievers.retriever_abc import BaseRetriever
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm


class OmniSQLRetriever(BaseRetriever):
    """
    A retriever for the OmniSQL system, using Pyserini's BM25.

    This retriever implements the indexing and retrieval logic from the
    OmniSQL codebase. It indexes individual cell values from a database and
    retrieves them based on lexical overlap with n-gram queries.
    """

    def __init__(self, enable_tqdm: bool = True):
        self.enable_tqdm = enable_tqdm
        self._searcher_cache: Dict[str, LuceneSearcher] = {}

    def index(self, items: List[SearchableItem], output_path: str) -> None:
        """
        Builds a Pyserini/Lucene index from a list of SearchableItem objects
        representing database cell values.
        """
        if os.path.exists(output_path) and os.listdir(output_path):
            print(f"Index already exists in '{output_path}'. Skipping indexing.")
            return

        pyserini_docs = []
        for item in items:
            pyserini_docs.append(
                {
                    "id": item.item_id,
                    "contents": item.content,
                    "metadata": item.metadata,
                }
            )

        if not pyserini_docs:
            print("No items to index.")
            return

        with tempfile.TemporaryDirectory() as temp_dir:
            # The original OmniSQL script produced a single JSON array file
            json_path = os.path.join(temp_dir, "contents.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(pyserini_docs, f, ensure_ascii=True)

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

            print("Running Pyserini indexing command...")
            result = subprocess.run(
                cmd, check=True, capture_output=True, text=True, encoding="utf-8"
            )
            print(result.stdout)
            if result.stderr:
                print("Pyserini STDERR:")
                print(result.stderr)

    def retrieve(
        self, processed_queries_batch: List[List[str]], output_path: str, k: int
    ) -> List[List[RetrievalResult]]:
        """
        Retrieves database cell values using the pre-built Pyserini index.
        """
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Index directory not found at: {output_path}")

        if output_path not in self._searcher_cache:
            self._searcher_cache[output_path] = LuceneSearcher(output_path)
        searcher = self._searcher_cache[output_path]

        flat_queries, query_to_original_idx_map = [], []
        for i, sub_queries in enumerate(processed_queries_batch):
            for sub_query in sub_queries:
                flat_queries.append(sub_query)
                query_to_original_idx_map.append(i)

        if not flat_queries:
            return [[] for _ in processed_queries_batch]

        q_ids = [str(i) for i in range(len(flat_queries))]

        batch_hits = searcher.batch_search(
            queries=flat_queries, qids=q_ids, k=k, threads=os.cpu_count() or 1
        )

        aggregated_results: List[Dict[str, RetrievalResult]] = [
            {} for _ in processed_queries_batch
        ]

        pbar_desc = "Processing BM25 Hits"
        for i, qid in enumerate(tqdm(q_ids, desc=pbar_desc, disable=not self.enable_tqdm)):
            original_nlq_idx = query_to_original_idx_map[i]
            hits = batch_hits.get(qid, [])

            for hit in hits:
                raw_doc_str = hit.lucene_document.get("raw")
                if not raw_doc_str:
                    continue

                stored_data = json.loads(raw_doc_str)
                item_id = stored_data.get("id")

                if not item_id:
                    continue

                item = SearchableItem(
                    item_id=item_id,
                    content=stored_data.get("contents", ""),
                    metadata=stored_data.get("metadata", {}),
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
        print(f"Sample from retriver: \n {final_batches[0]}")
        return final_batches