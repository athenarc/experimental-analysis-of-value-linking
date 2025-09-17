import re
from typing import List, Dict, Any

from nltk import ngrams
from more_itertools import flatten

from darelabdb.nlp_retrieval.user_query_processors.query_processor_abc import (
    BaseUserQueryProcessor,
)
from darelabdb.nlp_retrieval.user_query_processors.ner_query_processor import (
    NERQueryProcessor,
)


class ValueNetQueryProcessor(BaseUserQueryProcessor):
    """
    Extracts potential database value candidates from natural language queries.

    This processor implements ValueNet's Stage 1 logic. It combines a general-
    purpose NER processor with a suite of highly specific, rule-based heuristics
    to identify strings, numbers, dates, and other entities in a user's question
    that could correspond to values in a database.
    """

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Initializes the candidate extractor.

        Args:
            spacy_model: The name of the spaCy model to use for the underlying
                         NER processor.
        """
        self.ner_processor = NERQueryProcessor(spacy_model=spacy_model)

    def process(self, nlqs: List[str]) -> List[List[str]]:
        """
        Processes a batch of natural language queries to extract value candidates.

        Args:
            nlqs: A list of raw natural language query strings.

        Returns:
            A list of lists, where each inner list contains the unique
            candidate strings extracted for the corresponding input NLQ.
        """
        results = []
        ner_based_candidates_batch = self.ner_processor.process(nlqs)

        for i, nlq in enumerate(nlqs):
            all_candidates = set()
            question_tokens = nlq.lower().split()

            # 1. Add candidates from the general-purpose NER processor
            all_candidates.update(ner_based_candidates_batch[i])

            # 2. Add candidates from ValueNet's specialized heuristics
            all_candidates.update(self._find_values_in_quote(nlq))
            all_candidates.update(self._find_ordinals(question_tokens))
            all_candidates.update(self._find_emails(nlq))
            all_candidates.update(self._find_genders(question_tokens))
            all_candidates.update(self._find_null_empty_values(question_tokens))
            all_candidates.update(self._find_special_codes(nlq))
            all_candidates.update(self._find_single_letters(nlq))
            all_candidates.update(self._find_capitalized_words(nlq))
            all_candidates.update(self._find_numeric_values(nlq))
            all_candidates.update(self._process_multi_word_candidates(list(all_candidates)))


            results.append(list(all_candidates))
        return results
    
    def _process_multi_word_candidates(self, candidates: List[str]) -> List[str]:
        """
        For candidates with multiple words, generate n-grams.
        """
        processed_candidates = []
        for candidate in candidates:
            if " " in candidate:
                processed_candidates.extend(self._build_ngrams(candidate))
        return processed_candidates

    def _build_ngrams(self, multi_token_input: str) -> List[str]:
        """
        Creates n-grams from a multi-word string.
        Example: "New York" -> {"New York", "New", "York"}
        """
        combinations = {multi_token_input}
        tokens = multi_token_input.split()
        for n in range(1, len(tokens)):
            for t in ngrams(tokens, n):
                combinations.add(" ".join(t))
        return list(combinations)
    
    def _find_numeric_values(self, nlq: str) -> List[str]:
        """
        Extracts numeric values (integers and floats) from the query.
        """
        return re.findall(r"\b\d+(?:\.\d+)?\b", nlq)

    def _find_values_in_quote(self, question: str) -> List[str]:
        return re.findall(r"\s[\"'‘“’](.+?)[\"'’”]", question)

    def _find_ordinals(self, question_tokens: List[str]) -> List[str]:
        ordinals = {
            "once": 1, "twice": 2, "thrice": 3, "single": 1, "double": 2,
            "triple": 3, "first": 1, "second": 2, "third": 3, "fourth": 4,
            "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9,
            "tenth": 10,
        }
        tokens = flatten(map(lambda t: t.split("-"), question_tokens))
        return [str(ordinals[token]) for token in tokens if token in ordinals]

    def _find_emails(self, question: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", question)

    def _find_genders(self, question_tokens: List[str]) -> List[str]:
        gender_map = {
            "female": ["F", "female"], "females": ["F", "female"],
            "girl": ["F", "female"], "girls": ["F", "female"],
            "male": ["M", "male"], "males": ["M", "male"],
            "boy": ["M", "male"], "boys": ["M", "male"],
        }
        gender_values = []
        tokens = flatten(map(lambda t: t.split("-"), question_tokens))
        for token in tokens:
            if token in gender_map:
                gender_values.extend(gender_map[token])
        return gender_values

    def _find_null_empty_values(self, question_tokens: List[str]) -> List[str]:
        null_map = {"null": "null", "empty": ""}
        tokens = flatten(map(lambda t: t.split("-"), question_tokens))
        return [null_map[token] for token in tokens if token in null_map]

    def _find_special_codes(self, question: str) -> List[str]:
        matches = re.findall(r"[A-Z-/0-9]{2,}", question)
        return [m for m in matches if not m.isnumeric()]

    def _find_single_letters(self, question: str) -> List[str]:
        if re.search(r"\bletter(s)?\b", question, re.IGNORECASE):
            return re.findall(r"\b[A-Za-z]\b", question)
        return []

    def _find_capitalized_words(self, question: str) -> List[str]:
        q = question[1:] if question and question[0].isupper() and len(question) > 1 and question[1].islower() else question
        consecutive = [m.group() for m in re.finditer(r"(\b[A-Z0-9][A-Za-z0-9-/]+\b\s)+\b[A-Z0-9][A-Za-z0-9-/]+", q)]
        singles = [m.group() for m in re.finditer(r"\b[A-Z0-9][A-Za-z0-9-/]+\b", q)]
        
        all_caps = set(consecutive)
        for word in singles:
            if not any(word in c for c in consecutive) and not word.isnumeric():
                all_caps.add(word)
        return list(all_caps)