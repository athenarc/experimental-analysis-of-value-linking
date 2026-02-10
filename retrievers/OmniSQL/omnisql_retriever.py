import os
os.environ["_JAVA_OPTIONS"] = "-Xmx256g -Xms256g -XX:+UseG1GC -XX:MaxGCPauseMillis=200 -XX:ParallelGCThreads=16"
os.environ["JDK_JAVA_OPTIONS"] = "-Xmx256g -Xms256g -XX:+UseG1GC -XX:MaxGCPauseMillis=200 -XX:ParallelGCThreads=16"
import json
import subprocess
import tempfile
from typing import Dict, List, Iterable

from nlp_retrieval.core.models import RetrievalResult, SearchableItem
from nlp_retrieval.retrievers.retriever_abc import BaseRetriever
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm


class OmniSQLRetriever(BaseRetriever):
    """
    A retriever for the OmniSQL system, using Pyserini's BM25.
    """

    def __init__(self, enable_tqdm: bool = True):
        self.enable_tqdm = enable_tqdm
        self._searcher_cache: Dict[str, LuceneSearcher] = {}

    def index(self, items: Iterable[SearchableItem], output_path: str) -> None:
        if os.path.exists(output_path) and os.listdir(output_path):
            print(f"Index already exists in '{output_path}'. Skipping indexing.")
            return

        with tempfile.TemporaryDirectory() as temp_dir:
            prepared_jsonl_path = os.path.join(temp_dir, "corpus.jsonl")
            print(f"Writing corpus to temporary file: {prepared_jsonl_path}")

            count = 0
            with open(prepared_jsonl_path, "w", encoding="utf-8") as outfile:
                for item in items:
                    pyserini_doc = {
                        "id": item.item_id,
                        "contents": item.content,
                        "metadata": item.metadata,
                    }
                    outfile.write(json.dumps(pyserini_doc, ensure_ascii=False) + "\n")
                    count += 1
                    if count % 100000 == 0:
                        print(f"Written {count} docs...", end='\r')
            
            print(f"\nFinished writing {count} documents. Starting Pyserini Indexer...")

            # Cap indexing threads to 16 to be safe
            num_threads = min(os.cpu_count() or 1, 16)
            
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
        
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Index directory not found at: {output_path}")

        if output_path not in self._searcher_cache:
            self._searcher_cache[output_path] = LuceneSearcher(output_path)
        searcher = self._searcher_cache[output_path]

        flat_queries = []
        query_to_original_idx_map = []
        for i, sub_queries in enumerate(processed_queries_batch):
            for sub_query in sub_queries:
                if sub_query and isinstance(sub_query, str):
                    flat_queries.append(sub_query)
                    query_to_original_idx_map.append(i)

        if not flat_queries:
            return [[] for _ in processed_queries_batch]

        q_ids = [str(i) for i in range(len(flat_queries))]

        safe_threads = min(os.cpu_count() or 1, 8)

        batch_hits = searcher.batch_search(
            queries=flat_queries, qids=q_ids, k=k, threads=safe_threads
        )
        
        aggregated_results: List[Dict[str, RetrievalResult]] = [
            {} for _ in processed_queries_batch
        ]

        pbar_desc = "Processing BM25 Hits"
        for i in tqdm(range(len(flat_queries)), desc=pbar_desc, disable=not self.enable_tqdm):
            original_nlq_idx = query_to_original_idx_map[i]
            qid = q_ids[i]
            hits = batch_hits.get(qid, [])

            if not hits:
                continue

            for hit in hits:
                # Use searcher.doc() which is cleaner than hit.lucene_document
                try:
                    doc = searcher.doc(hit.docid)
                    if not doc: continue
                    raw_doc_str = doc.raw()
                except:
                    continue

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
            
        return final_batches