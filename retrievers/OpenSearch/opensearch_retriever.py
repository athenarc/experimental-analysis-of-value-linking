
from typing import List
from darelabdb.nlp_retrieval.core.models import RetrievalResult, SearchableItem
from darelabdb.nlp_retrieval.retrievers.retriever_abc import BaseRetriever
from darelabdb.nlp_retrieval.retrievers.dense_retriever import FaissRetriever

class OpenSearchDenseValueRetriever(BaseRetriever):
    """
    A wrapper retriever for the OpenSearch pipeline that performs dense
    retrieval of database values using vector similarity.

    This class uses the `FaissRetriever` internally to handle embedding,
    FAISS indexing, and searching
    """
    def __init__(
        self,
        model_name_or_path: str="BAAI/bge-m3",
        embedding_backend: str = "sentence-transformers",
        **kwargs,
    ):
        """
        Initializes the dense value retriever.

        Args:
            model_name_or_path (str): The name or path of the embedding model.
            embedding_backend (str): The backend for embedding ('sentence-transformers', 'vllm', etc.).
            **kwargs: Additional arguments for the underlying FaissRetriever.
        """
        self.internal_retriever = FaissRetriever(
            model_name_or_path=model_name_or_path,
            embedding_backend=embedding_backend,
            batch_size=32,
            **kwargs
        )
        print(f"OpenSearchDenseValueRetriever initialized with model: {model_name_or_path}")

    def index(self, items: List[SearchableItem], output_path: str) -> None:
        """
        Builds a FAISS index from the `content` of the provided SearchableItems.
        """
        self.internal_retriever.index(items, output_path)

    def retrieve(
        self, processed_queries_batch: List[List[str]], output_path: str, k: int
    ) -> List[List[RetrievalResult]]:
        """
        Searches the index for keywords and returns the most similar database values.
        """
        return self.internal_retriever.retrieve(processed_queries_batch, output_path, k)