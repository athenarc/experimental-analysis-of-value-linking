import difflib
from typing import List

import numpy as np
from darelabdb.nlp_retrieval.core.models import RetrievalResult
from darelabdb.nlp_retrieval.rerankers.reranker_abc import BaseReranker
from sentence_transformers import SentenceTransformer, util
from tqdm.auto import tqdm


class ChessSimilarityReranker(BaseReranker):
    """
    A reranker that adapts the multi-stage similarity logic from the CHESS
    framework's `retrieve_entity` tool.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        edit_distance_weight: float = 0.4,
        embedding_similarity_weight: float = 0.6,
    ):
        """
        Initializes the reranker.

        Args:
            model_name: The sentence-transformer model to use for embeddings.
            edit_distance_weight: The weight for the edit distance score component.
            embedding_similarity_weight: The weight for the embedding similarity score.
        """
        self.model = SentenceTransformer(model_name)
        self.edit_weight = edit_distance_weight
        self.embed_weight = embedding_similarity_weight

    def rerank(
        self, nlqs: List[str], results_batch: List[List[RetrievalResult]], k: int
    ) -> List[List[RetrievalResult]]:
        final_batches = []
        progress_bar_desc = f"Reranking with CHESS Similarity"

        for nlq, candidate_list in tqdm(
            zip(nlqs, results_batch),
            total=len(nlqs),
            desc=progress_bar_desc,
            disable=len(nlqs) < 5,
        ):
            if not candidate_list:
                final_batches.append([])
                continue

            # 1. Calculate Edit Distance (SequenceMatcher ratio) scores
            edit_scores = [
                difflib.SequenceMatcher(
                    None, nlq.lower(), res.item.content.lower()
                ).ratio()
                for res in candidate_list
            ]

            # 2. Calculate Embedding Similarity (Cosine) scores
            query_embedding = self.model.encode(nlq, convert_to_tensor=True)
            doc_contents = [res.item.content for res in candidate_list]
            doc_embeddings = self.model.encode(
                doc_contents, convert_to_tensor=True
            )
            embedding_scores = (
                util.cos_sim(query_embedding, doc_embeddings)[0].cpu().tolist()
            )

            # 3. Combine scores and create new results
            rescored_results = []
            for i, res in enumerate(candidate_list):
                # Ensure scores are non-negative before combining
                edit_score = max(0, edit_scores[i])
                embed_score = max(0, embedding_scores[i])

                final_score = (self.edit_weight * edit_score) + (
                    self.embed_weight * embed_score
                )
                rescored_results.append(
                    RetrievalResult(item=res.item, score=final_score)
                )

            # 4. Sort by the new combined score and truncate
            sorted_results = sorted(
                rescored_results, key=lambda r: r.score, reverse=True
            )
            final_batches.append(sorted_results)

        return final_batches