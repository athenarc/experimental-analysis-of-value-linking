from darelabdb.nlp_value_linking.filtering.filtering_abc import FilterABC
from FlagEmbedding import FlagLLMReranker
from typing import List


class BGEFlagLLMRerankerFilter(FilterABC):
    """
    Filter implementation using a reranker based on the BGE model integrated with LLM capabilities.

    This filter uses the FlagLLMReranker to compute relevance scores for each query-value pair
    and returns the formatted values if their scores exceed a specified threshold.
    """

    def __init__(self, threshold=0.85, model="BAAI/bge-reranker-v2-gemma"):
        """
        Initialize the BGEFlagLLMRerankerFilter.

        Parameters:
            threshold (float): The score threshold for filtering documents.
            model (str): The model identifier used for the LLM reranker.
        """
        self.threshold = threshold
        self.model = model
        self.reranker = FlagLLMReranker(model, use_fp16=True)
        self.queries_per_keyword = {}

    def add_pair(self, keyword: str, value_pair: tuple):
        if keyword not in self.queries_per_keyword:
            self.queries_per_keyword[keyword] = []
        self.queries_per_keyword[keyword].append(value_pair)

    def filter(self) -> List[str]:
        filtered_values = []
        for keyword, value_pairs in self.queries_per_keyword.items():
            texts = [pair[0] for pair in value_pairs]
            pairs_for_scoring = [[keyword, txt] for txt in texts]
            scores = self.reranker.compute_score(pairs_for_scoring, normalize=True)

            for score, (_, second_element) in zip(scores, value_pairs):
                if score > self.threshold:
                    filtered_values.append(second_element)

        self.queries_per_keyword = {}
        return filtered_values
