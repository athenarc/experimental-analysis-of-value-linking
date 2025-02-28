from darelabdb.nlp_value_linking.cvr_extractor.cvr_extractor_abc import CVRExtractorABC
from nltk.util import ngrams
from nltk import word_tokenize


class NGramsExtractor(CVRExtractorABC):
    """CVR extractor generating n-grams up to specified length.

    Creates character n-grams from unigrams to n-grams (default n=4)
    to capture potential multi-word value references."""

    def __init__(self, n=4):
        self.n = n

    def extract_keywords(self, input_text):
        tokens = word_tokenize(input_text)
        all_ngrams = []
        for n in range(1, self.n + 1):
            n_grams = [" ".join(ngram) for ngram in ngrams(tokens, n)]
            all_ngrams.extend(n_grams)
        return list(set(all_ngrams))
