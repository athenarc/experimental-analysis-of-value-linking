import difflib
from typing import List, Tuple, Optional
from rapidfuzz import fuzz

from darelabdb.nlp_retrieval.core.models import RetrievalResult
from darelabdb.nlp_retrieval.rerankers.reranker_abc import BaseReranker
from tqdm.auto import tqdm

# Constants from the original script
_STOPWORDS = {'who', 'ourselves', 'down', 'only', 'were', 'him', 'at', "weren't", 'has', 'few', "it's", 'm', 'again',
              'd', 'haven', 'been', 'other', 'we', 'an', 'own', 'doing', 'ma', 'hers', 'all', "haven't", 'in', 'but',
              "shouldn't", 'does', 'out', 'aren', 'you', "you'd", 'himself', "isn't", 'most', 'y', 'below', 'is',
              "wasn't", 'hasn', 'them', 'wouldn', 'against', 'this', 'about', 'there', 'don', "that'll", 'a', 'being',
              'with', 'your', 'theirs', 'its', 'any', 'why', 'now', 'during', 'weren', 'if', 'should', 'those', 'be',
              'they', 'o', 't', 'of', 'or', 'me', 'i', 'some', 'her', 'do', 'will', 'yours', 'for', 'mightn', 'nor',
              'needn', 'the', 'until', "couldn't", 'he', 'which', 'yourself', 'to', "needn't", "you're", 'because',
              'their', 'where', 'it', "didn't", 've', 'whom', "should've", 'can', "shan't", 'on', 'had', 'have',
              'myself', 'am', "don't", 'under', 'was', "won't", 'these', 'so', 'as', 'after', 'above', 'each', 'ours',
              'hadn', 'having', 'wasn', 's', 'doesn', "hadn't", 'than', 'by', 'that', 'both', 'herself', 'his',
              "wouldn't", 'into', "doesn't", 'before', 'my', 'won', 'more', 'are', 'through', 'same', 'how', 'what',
              'over', 'll', 'yourselves', 'up', 'mustn', "mustn't", "she's", 're', 'such', 'didn', "you'll", 'shan',
              'when', "you've", 'themselves', "mightn't", 'she', 'from', 'isn', 'ain', 'between', 'once', 'here',
              'shouldn', 'our', 'and', 'not', 'too', 'very', 'further', 'while', 'off', 'couldn', "hasn't", 'itself',
              'then', 'did', 'just', "aren't"}
_COMMONWORDS = {"no", "yes", "many"}
_COMMON_DB_TERMS = {"id"}


class Match:
    """Helper class to store match information."""
    def __init__(self, start: int, size: int):
        self.start = start
        self.size = size


class CodesReranker(BaseReranker):
    """
    A reranker that implements the matching and filtering logic from the CoDeS paper.

    """

    def __init__(self, m_theta: float = 0.85, s_theta: float = 0.85, top_k: int = 20):
        """
        Args:
            m_theta: The match score threshold.
            s_theta: The secondary match score threshold.
            top_k: The number of top results to return.
        """
        self.m_theta = m_theta
        self.s_theta = s_theta
        self.top_k = top_k

    @staticmethod
    def _is_stopword(s: str) -> bool:
        return s.strip().lower() in _STOPWORDS

    @staticmethod
    def _is_commonword(s: str) -> bool:
        return s.strip().lower() in _COMMONWORDS

    @staticmethod
    def _is_common_db_term(s: str) -> bool:
        return s.strip().lower() in _COMMON_DB_TERMS

    @staticmethod
    def _is_span_separator(c: str) -> bool:
        return c in "'\"()`,.?! "

    @staticmethod
    def _split(s: str) -> List[str]:
        return [c.lower() for c in s.strip()]

    def _get_effective_match_source(self, s: str, start: int, end: int) -> Optional[Match]:
        _start = -1
        for i in range(start, start - 2, -1):
            if i < 0:
                _start = i + 1
                break
            if self._is_span_separator(s[i]):
                _start = i
                break
        if _start < 0:
            return None

        _end = -1
        for i in range(end - 1, end + 3):
            if i >= len(s):
                _end = i - 1
                break
            if self._is_span_separator(s[i]):
                _end = i
                break
        if _end < 0:
            return None

        while _start < len(s) and self._is_span_separator(s[_start]):
            _start += 1
        while _end >= 0 and self._is_span_separator(s[_end]):
            _end -= 1

        return Match(_start, _end - _start + 1)

    @staticmethod
    def _prefix_match(s1: str, s2: str) -> bool:
        i, j = 0, 0
        while i < len(s1) and CodesReranker._is_span_separator(s1[i]):
            i += 1
        while j < len(s2) and CodesReranker._is_span_separator(s2[j]):
            j += 1

        if i < len(s1) and j < len(s2):
            return s1[i].lower() == s2[j].lower()
        return i >= len(s1) and j >= len(s2)

    def _get_matched_entries(
        self, s: str, field_values: List[str]
    ) -> List[Tuple[str, Tuple[str, str, float, float, int]]]:
        if not field_values:
            return []

        n_grams = self._split(s)
        matched = {}

        for field_value in field_values:
            if not isinstance(field_value, str):
                continue

            fv_tokens = self._split(field_value)
            sm = difflib.SequenceMatcher(None, n_grams, fv_tokens)
            match = sm.find_longest_match(0, len(n_grams), 0, len(fv_tokens))

            if match.size > 0:
                source_match = self._get_effective_match_source(s, match.a, match.a + match.size)
                if source_match:
                    match_str = field_value[match.b: match.b + match.size]
                    source_match_str = s[source_match.start: source_match.start + source_match.size]
                    
                    c_match_str = match_str.lower().strip()
                    c_source_match_str = source_match_str.lower().strip()
                    c_field_value = field_value.lower().strip()

                    if not c_match_str or self._is_common_db_term(c_match_str):
                        continue
                    if self._is_stopword(c_match_str) or self._is_stopword(c_source_match_str) or self._is_stopword(c_field_value):
                        continue

                    if c_source_match_str.endswith(c_match_str + "'s"):
                        match_score = 1.0
                    elif self._prefix_match(c_field_value, c_source_match_str):
                        match_score = fuzz.ratio(c_field_value, c_source_match_str) / 100.0
                    else:
                        match_score = 0.0
                    
                    if (self._is_commonword(c_match_str) or self._is_commonword(c_source_match_str) or self._is_commonword(c_field_value)) and match_score < 1.0:
                        continue

                    s_match_score = match_score
                    if match_score >= self.m_theta and s_match_score >= self.s_theta:
                        if field_value.isupper() and match_score * s_match_score < 1.0:
                            continue
                        
                        # Use the most representative match
                        if match_str not in matched or match.size > matched[match_str][4]:
                            matched[match_str] = (field_value, source_match_str, match_score, s_match_score, match.size)

        if not matched:
            return []
        
        return sorted(
            matched.items(),
            key=lambda x: (1e16 * x[1][2] + 1e8 * x[1][3] + x[1][4]),
            reverse=True,
        )

    def rerank(
        self, nlqs: List[str], results_batch: List[List[RetrievalResult]], k: int
    ) -> List[List[RetrievalResult]]:
        """
        Reranks candidates using the CoDeS matching and scoring logic.
        """
        final_batches = []
        for nlq, result_list in tqdm(
            zip(nlqs, results_batch),
            total=len(nlqs),
            desc="Reranking with CoDeS logic",
            disable=len(nlqs) < 5,
        ):
            if not result_list:
                final_batches.append([])
                continue

            field_values = [res.item.content for res in result_list]
            
            # The original implementation processes all field values at once for a given question.
            matched_entries = self._get_matched_entries(nlq, field_values)

            # Create a map from content to its new score details
            content_to_score = {
                entry[1][0]: (entry[1][2], entry[1][3], entry[1][4])
                for entry in matched_entries
            }

            rescored_results = []
            for res in result_list:
                if res.item.content in content_to_score:
                    score_tuple = content_to_score[res.item.content]
                    # Combine scores for ranking, similar to the original key
                    new_score = 1e16 * score_tuple[0] + 1e8 * score_tuple[1] + score_tuple[2]
                    rescored_results.append(RetrievalResult(item=res.item, score=new_score))

            sorted_results = sorted(rescored_results, key=lambda r: r.score, reverse=True)
            final_batches.append(sorted_results[:self.top_k])

        return final_batches