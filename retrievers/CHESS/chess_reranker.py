import difflib
from typing import List
from darelabdb.nlp_retrieval.core.models import RetrievalResult
from darelabdb.nlp_retrieval.rerankers.reranker_abc import BaseReranker
from sentence_transformers import SentenceTransformer, util
from tqdm.auto import tqdm


class ChessSimilarityReranker(BaseReranker):
    """
    A reranker that adapts the multi-stage FILTERING logic from the CHESS
    framework's `_get_similar_entities` method.

    This implementation uses a sequential filtering cascade. It expects the
    retriever to have placed the keyword that found the item in the
    result's metadata under the key 'matched_keyword'.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        edit_distance_threshold: float = 0.3,
        embedding_similarity_threshold: float = 0.6,
    ):
        """
        Initializes the reranker with thresholds.

        Args:
            model_name: The sentence-transformer model to use for embeddings.
            edit_distance_threshold: The minimum SequenceMatcher ratio to pass the first filter.
            embedding_similarity_threshold: The minimum cosine similarity to pass the second filter.
        """
        self.model = SentenceTransformer(model_name,device="cuda")
        self.edit_distance_threshold = edit_distance_threshold
        self.embedding_similarity_threshold = embedding_similarity_threshold

    def rerank(
        self,
        nlqs: List[str],
        results_batch: List[List[RetrievalResult]],
        k: int,
    ) -> List[List[RetrievalResult]]:
        final_batches = []
        progress_bar_desc = "Reranking with CHESS Cascade Filter"

        for candidate_list in tqdm(
            results_batch,
            total=len(results_batch),
            desc=progress_bar_desc,
            disable=len(results_batch) < 5,
        ):
            if not candidate_list:
                final_batches.append([])
                continue
            
            # Gather all unique keywords and contents for efficient batch embedding
            keywords_to_embed = set()
            contents_to_embed = []
            for res in candidate_list:
                if res.item.metadata and 'matched_keyword' in res.item.metadata:
                    keywords_to_embed.add(res.item.metadata['matched_keyword'])
                    contents_to_embed.append(res.item.content)
            
            if not keywords_to_embed: # If no keywords were passed, can't rerank
                final_batches.append(sorted(candidate_list, key=lambda r: r.score, reverse=True)[:k])
                continue

            # Batch encode all necessary embeddings
            keyword_list = list(keywords_to_embed)
            keyword_embeddings = self.model.encode(keyword_list,batch_size=8)
            keyword_emb_map = {kw: emb for kw, emb in zip(keyword_list, keyword_embeddings)}
            
            content_embeddings = self.model.encode(contents_to_embed,batch_size=8)
            
            filtered_and_rescored = []
            for i, res in enumerate(candidate_list):
                keyword = res.item.metadata.get('matched_keyword')
                if not keyword:
                    continue # Skip if the retriever didn't provide the keyword

                candidate_content = res.item.content
                
                # STAGE 1: Edit Distance Filter
                edit_score = difflib.SequenceMatcher(
                    None, keyword.lower(), candidate_content.lower()
                ).ratio()

                if edit_score >= self.edit_distance_threshold:
                    # STAGE 2: Embedding Similarity Filter
                    keyword_embedding = keyword_emb_map[keyword]
                    content_embedding = content_embeddings[i]
                    embedding_score = util.cos_sim(keyword_embedding, content_embedding)[0].item()

                    if embedding_score >= self.embedding_similarity_threshold:
                        # Passed the cascade. The new score is the embedding similarity.
                        filtered_and_rescored.append(
                            RetrievalResult(item=res.item, score=embedding_score)
                        )

            # Sort the final list of survivors by their new score
            sorted_results = sorted(
                filtered_and_rescored, key=lambda r: r.score, reverse=True
            )
            final_batches.append(sorted_results[:k])

        return final_batches