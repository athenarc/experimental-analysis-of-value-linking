from typing import List

from darelabdb.nlp_retrieval.user_query_processors.query_processor_abc import (
    BaseUserQueryProcessor,
)
from nltk import ngrams, word_tokenize
from tqdm.auto import tqdm


class OmniSQLQueryProcessor(BaseUserQueryProcessor):
    """
    A query processor that implements the n-gram generation logic from OmniSQL.

    For each natural language question, it generates all n-grams up to a
    specified length and also includes the original question. These are then
    used as individual queries for the BM25 retriever.
    """

    def __init__(self, n: int = 8):
        """
        Args:
            n: The maximum size of the n-grams to generate.
        """
        self.n = n
        import nltk

        nltk.download("punkt", quiet=True)


    def _get_ngrams(self, text: str) -> List[str]:
        """
        Extracts all n-grams from 1 to self.n for a given text.
        """
        tokens = word_tokenize(text)
        all_ngrams = []
        for i in range(1, self.n + 1):
            n_grams_for_i = [" ".join(gram) for gram in ngrams(tokens, i)]
            all_ngrams.extend(n_grams_for_i)
        return all_ngrams

    def process(self, nlqs: List[str]) -> List[List[str]]:
        """
        Processes a batch of queries by generating n-grams for each.
        """
        results = []
        for nlq in tqdm(nlqs, desc="Generating N-grams for OmniSQL"):
            # The original code uses n-grams plus the original question
            generated_queries = self._get_ngrams(nlq) + [nlq]
            # Deduplicate while preserving order
            results.append(list(dict.fromkeys(generated_queries)))
        return results