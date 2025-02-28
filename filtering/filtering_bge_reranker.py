from darelabdb.nlp_value_linking.filtering.filtering_abc import FilterABC
from FlagEmbedding import FlagReranker
from typing import List


class BGEFlagRerankerFilter(FilterABC):
    """
    Filter implementation using a reranker based on BGE (retrieval-enhanced transformer models).

    This filter leverages the FlagReranker to compute relevance scores and returns values
    if their score exceeds a specified threshold.
    """

    def __init__(self, threshold=0.85, model="BAAI/bge-reranker-v2-m3"):
        """
        Initialize the BGEFlagRerankerFilter.

        Parameters:
            threshold (float): The score threshold for filtering documents.
            model (str): The model identifier used for the reranker.
        """
        self.threshold = threshold
        self.model = model
        self.reranker = FlagReranker(model)
        self.queries_per_keyword = {}

    def add_pair(self, keyword: str, value_pair: tuple):
        """Add a keyword and its corresponding value pair."""
        # Validate inputs
        if not isinstance(keyword, str):
            raise ValueError(f"Keyword must be a string. Got: {type(keyword)}")
        if not isinstance(value_pair, tuple) or len(value_pair) != 2:
            raise ValueError("Value pair must be a tuple of length 2.")
        if not isinstance(value_pair[0], (str, int, float)):
            raise ValueError(
                f"First element of value pair must be a string, int, or float. Got: {type(value_pair[0])}"
            )
        if keyword not in self.queries_per_keyword:
            self.queries_per_keyword[keyword] = []
        self.queries_per_keyword[keyword].append((str(value_pair[0]), value_pair[1]))

    def filter(self) -> List[str]:
        filtered_values = []

        for keyword, value_pairs in self.queries_per_keyword.items():
            texts = [str(pair[0]) for pair in value_pairs]
            pairs_for_scoring = [(str(keyword), txt) for txt in texts]

            try:
                # Compute scores
                scores = self.reranker.compute_score(pairs_for_scoring, normalize=True)

                # Filter based on threshold
                for score, (_, second_element) in zip(scores, value_pairs):
                    if score > self.threshold:
                        filtered_values.append(second_element)

            except Exception as e:
                print(f"Error during scoring for keyword '{keyword}': {e}")

        # Clear the queries after processing
        self.queries_per_keyword = {}
        return filtered_values
