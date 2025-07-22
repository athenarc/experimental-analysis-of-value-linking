from typing import List, Optional
from darelabdb.nlp_retrieval.user_query_processors.query_processor_abc import BaseUserQueryProcessor
from darelabdb.nlp_retrieval.user_query_processors.llm_keyword_extractor_query_processor import KeywordExtractorProcessor

class OpenSearchKeywordProcessor(BaseUserQueryProcessor):
    """
    A wrapper query processor for the OpenSearch pipeline that uses a Large
    Language Model to extract key terms, entities, and phrases from queries.

    This class uses the `KeywordExtractorProcessor` internally, which handles
    prompting, batching, and caching for efficient LLM-based processing.
    """
    def __init__(
        self,
        model_name_or_path: str,
        cache_folder: Optional[str] = None,
        **kwargs,
    ):
        """
        Initializes the LLM-based keyword extractor.

        Args:
            model_name_or_path (str): The name or path of the model for VLLM.
            cache_folder (Optional[str]): A path to a folder for caching results.
            **kwargs: Additional arguments for VLLM's LLM class (e.g., tensor_parallel_size).
        """
        self.internal_processor = KeywordExtractorProcessor(
            model_name_or_path=model_name_or_path,
            cache_folder=cache_folder,
            **kwargs
        )
        print(f"OpenSearchKeywordProcessor initialized with model: {model_name_or_path}")

    def process(self, nlqs: List[str]) -> List[List[str]]:
        """
        Processes a batch of natural language queries to extract keywords.

        Returns:
            A list of lists, where each inner list contains the keywords for one query.
        """
        return self.internal_processor.process(nlqs)