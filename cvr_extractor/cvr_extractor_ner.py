from cvr_extractor.cvr_extractor_abc import CVRExtractorABC
import spacy
import string
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from typing import List


class NERExtractor(CVRExtractorABC):
    """CVR extractor using Named Entity Recognition and POS tagging.

    Combines spaCy's named entity recognition with NLTK's part-of-speech tagging
    to identify potential value references. Filters nouns, proper nouns, and
    adjectives while removing stopwords and punctuation."""

    def __init__(self, nlp_model="en_core_web_sm"):
        self.nlp = spacy.load("en_core_web_sm")
        self.stop_words = set(stopwords.words("english"))
        self.punctuation = set(string.punctuation)

    def extract_keywords(self, input_text: str) -> List[str]:
        doc = self.nlp(input_text)
        entities = [ent.text for ent in doc.ents]

        tokens = word_tokenize(input_text)
        pos_tags = pos_tag(tokens)
        keywords = [
            word
            for word, pos in pos_tags
            if pos in ["NN", "NNS", "NNP", "NNPS", "JJ"]
            and word.lower() not in self.stop_words
            and word not in self.punctuation
        ]
        return list(set(entities + keywords))
