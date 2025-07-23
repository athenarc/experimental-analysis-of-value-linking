import os
import pickle
from typing import Dict, List, Set, Tuple

from datasketch import MinHash, MinHashLSH
from darelabdb.nlp_retrieval.core.models import RetrievalResult, SearchableItem
from darelabdb.nlp_retrieval.retrievers.retriever_abc import BaseRetriever
from tqdm import tqdm


class ChessMinHashLshRetriever(BaseRetriever):
    """
    A retriever that adapts the MinHash and Locality Sensitive Hashing (LSH)
    logic from the CHESS text-to-SQL framework.
    """

    LSH_INDEX_FILENAME = "chess_minhash_lsh.pkl"
    ITEMS_FILENAME = "chess_items.pkl"

    def __init__(
        self,
        num_perm: int = 100,
        n_gram: int = 3,
        threshold: float = 0.01,
        enable_tqdm: bool = True,
    ):
        """
        Initializes the ChessMinHashLshRetriever.

        Args:
            num_perm: The number of permutation functions for MinHash signatures.
                      Corresponds to `signature_size` in the CHESS codebase.
            n_gram: The n-gram size for creating shingles from text. This is
                    used in the `_create_minhash` function in CHESS.
            threshold: The Jaccard similarity threshold for the LSH index.
            enable_tqdm: If True, displays tqdm progress bars.
        """
        self.num_perm = num_perm
        self.n_gram = n_gram
        self.threshold = threshold
        self.enable_tqdm = enable_tqdm
        self._lsh_cache: Dict[str, Tuple[MinHashLSH, Dict[str, MinHash]]] = {}
        self._items_cache: Dict[str, Dict[str, SearchableItem]] = {}

    def _create_minhash(self, text: str) -> MinHash:
        """
        Creates a MinHash signature for a given text using n-grams,
        replicating the logic from CHESS's `_create_minhash`.
        """
        minhash = MinHash(num_perm=self.num_perm)
        # CHESS uses n-grams for shingling, which we replicate here.
        for d in [
            text[i : i + self.n_gram] for i in range(len(text) - self.n_gram + 1)
        ]:
            minhash.update(d.encode("utf8"))
        return minhash

    def index(self, items: List[SearchableItem], output_path: str) -> None:
        """
        Builds a MinHash LSH index from a list of SearchableItem objects.
        This adapts the logic of `make_db_lsh` from the CHESS framework.
        """
        lsh_index_path = os.path.join(output_path, self.LSH_INDEX_FILENAME)
        items_path = os.path.join(output_path, self.ITEMS_FILENAME)

        if os.path.exists(lsh_index_path) and os.path.exists(items_path):
            print(f"CHESS index already exists in '{output_path}'. Skipping.")
            return

        if not items:
            return

        lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        item_id_to_minhash_map: Dict[str, MinHash] = {}

        pbar_desc = "Creating CHESS MinHash Signatures"
        for item in tqdm(items, desc=pbar_desc, disable=not self.enable_tqdm):
            minhash = self._create_minhash(item.content)
            item_id_to_minhash_map[item.item_id] = minhash
            lsh.insert(item.item_id, minhash)

        with open(lsh_index_path, "wb") as f:
            pickle.dump((lsh, item_id_to_minhash_map), f)

        with open(items_path, "wb") as f:
            pickle.dump(items, f)

    def _load_index_and_items(
        self, output_path: str
    ) -> Tuple[MinHashLSH, Dict[str, MinHash], Dict[str, SearchableItem]]:
        """Loads the LSH index, MinHash map, and item map, caching them."""
        if output_path in self._lsh_cache:
            lsh, minhash_map = self._lsh_cache[output_path]
            items_map = self._items_cache[output_path]
            return lsh, minhash_map, items_map

        lsh_path = os.path.join(output_path, self.LSH_INDEX_FILENAME)
        items_path = os.path.join(output_path, self.ITEMS_FILENAME)

        with open(lsh_path, "rb") as f:
            lsh, minhash_map = pickle.load(f)
        with open(items_path, "rb") as f:
            items_list = pickle.load(f)
        items_map = {item.item_id: item for item in items_list}

        self._lsh_cache[output_path] = (lsh, minhash_map)
        self._items_cache[output_path] = items_map
        return lsh, minhash_map, items_map

    def retrieve(
        self, processed_queries_batch: List[List[str]], output_path: str, k: int
    ) -> List[List[RetrievalResult]]:
        """
        Retrieves items using the LSH index, adapting the logic from
        CHESS's `query_lsh` function.
        """
        lsh, minhash_map, items_map = self._load_index_and_items(output_path)
        final_batches = []

        for sub_queries in processed_queries_batch:
            aggregated_results: Dict[str, List[Tuple[float, str]]] = {}

            for query in sub_queries:
                query_minhash = self._create_minhash(query)
                retrieved_ids = lsh.query(query_minhash)

                for item_id in retrieved_ids:
                    if item_id in minhash_map and item_id in items_map:
                        item_minhash = minhash_map[item_id]
                        score = query_minhash.jaccard(item_minhash)

                        if score > self.threshold:
                            if item_id not in aggregated_results:
                                aggregated_results[item_id] = []
                            aggregated_results[item_id].append((score, query))
            
            final_results_for_batch = []
            for item_id, score_keyword_pairs in aggregated_results.items():
                best_score, best_keyword = max(score_keyword_pairs, key=lambda x: x[0])
                
                item = items_map[item_id]

                new_metadata = item.metadata.copy() if item.metadata else {}
                new_metadata['matched_keyword'] = best_keyword
                
                updated_item = SearchableItem(
                    item_id=item.item_id,
                    content=item.content,
                    metadata=new_metadata
                )

                result = RetrievalResult(item=updated_item, score=best_score)
                final_results_for_batch.append(result)

            sorted_res = sorted(
                final_results_for_batch, key=lambda r: r.score, reverse=True
            )
            final_batches.append(sorted_res)

        return final_batches