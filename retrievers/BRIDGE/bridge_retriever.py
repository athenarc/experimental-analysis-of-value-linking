import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import difflib
from rapidfuzz import fuzz
from nlp_retrieval.core.models import RetrievalResult, SearchableItem
from nlp_retrieval.retrievers.retriever_abc import BaseRetriever

_stopwords = {'who', 'ourselves', 'down', 'only', 'were', 'him', 'at', "weren't", 'has', 'few', "it's", 'm', 'again',
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

_commonwords = {
    'no', 'yes', 'many'
}

string_types = (type(b''), type(u''))

def is_number(s):
    try:
        float(s.replace(',', ''))
        return True
    except:
        return False


def is_stopword(s):
    return s.strip() in _stopwords


def is_commonword(s):
    return s.strip() in _commonwords


def is_common_db_term(s):
    return s.strip() in ['id']


class Match(object):
    def __init__(self, start: int, size: int):
        self.start = start
        self.size = size


def _is_span_separator(c: str) -> bool:
    return c in '\'"()`,.?! '


def _split(s: str) -> List[str]:
    # original behavior: list of chars lowercased after strip
    return [c.lower() for c in s.strip()]


def _prefix_match(s1: str, s2: str) -> bool:
    # reproduce original prefix_match logic
    i, j = 0, 0
    for i in range(len(s1)):
        if not _is_span_separator(s1[i]):
            break
    for j in range(len(s2)):
        if not _is_span_separator(s2[j]):
            break
    if i < len(s1) and j < len(s2):
        return s1[i] == s2[j]
    elif i >= len(s1) and j >= len(s2):
        return True
    else:
        return False


def _get_effective_match_source(s: str, start: int, end: int) -> Optional[Match]:
    """
    Exactly replicate original get_effecitve_match_source:
    s: original string (not token list)
    start, end: token indices (end is exclusive)
    returns Match(start_index_in_s, size) or None
    """
    _start = -1
    # scan backward up to 2 positions
    for i in range(start, start - 2, -1):
        if i < 0:
            _start = i + 1
            break
        if _is_span_separator(s[i]):
            _start = i
            break
    if _start < 0:
        return None

    _end = -1
    # scan forwards up to 2 positions after end-1
    for i in range(end - 1, end + 3):
        if i >= len(s):
            _end = i - 1
            break
        if _is_span_separator(s[i]):
            _end = i
            break
    if _end < 0:
        return None

    # trim leading separators
    while (_start < len(s) and _is_span_separator(s[_start])):
        _start += 1
    # trim trailing separators
    while (_end >= 0 and _is_span_separator(s[_end])):
        _end -= 1

    # original used: Match(_start, end - start + 1)
    return Match(_start, end - start + 1)


class BridgeRetriever(BaseRetriever):
    def __init__(
        self,
        top_k_matches_per_column: int = 5,
        match_threshold: float = 0.85,
        s_match_threshold: float = 0.85,
    ):
        self.top_k_matches_per_column = top_k_matches_per_column
        self.match_threshold = match_threshold
        self.s_match_threshold = s_match_threshold
        # index: table -> column -> list[SearchableItem]
        self._index_data: Dict[str, Dict[str, List[SearchableItem]]] = defaultdict(
            lambda: defaultdict(list)
        )

    def index(self, items: List[SearchableItem], output_path: str) -> None:
        index_file = os.path.join(output_path, "bridge_index.jsonl")

        with open(index_file, "w") as f:
            for item in items:
                f.write(item.model_dump_json() + "\n")

    def _load_index(self, output_path: str) -> None:
        index_file = os.path.join(output_path, "bridge_index.jsonl")
        self._index_data.clear()

        with open(index_file, "r") as f:
            for line in f:
                item_dict = json.loads(line)
                item = SearchableItem(**item_dict)
                table = item.metadata.get("table")
                column = item.metadata.get("column")
                if table and column:
                    self._index_data[table][column].append(item)

    def _get_matched_entries(
        self,
        s: str,
        searchable_items: List[SearchableItem],
        m_theta: float = 0.85,
        s_theta: float = 0.85,
    ) -> Optional[List[Tuple[str, Tuple[str, str, float, float, int, SearchableItem]]]]:
        """
        Replicates the original get_matched_entries(s, field_values, m_theta, s_theta)

        Returns:
            Either None if no matches, or a sorted list of tuples:
              (match_str, (field_value, source_match_str, match_score, s_match_score, match.size, item))
            sorted by the original composite key:
              1e16 * match_score + 1e8 * s_match_score + match.size (descending)
        """
        if not searchable_items:
            return None

        # n_grams in original is the token list (characters) for s
        n_grams = _split(s) if isinstance(s, str) else s

        matched = dict()  # key: match_str -> value: (field_value, source_match_str, match_score, s_match_score, match.size, item)

        for item in searchable_items:
            field_value = item.metadata.get("value")
            if not isinstance(field_value, string_types):
                continue
            # tokens of field value (characters)
            fv_tokens = _split(field_value)

            sm = difflib.SequenceMatcher(None, n_grams, fv_tokens)
            match = sm.find_longest_match(0, len(n_grams), 0, len(fv_tokens))

            if match.size > 0:
                source_match = _get_effective_match_source(s, match.a, match.a + match.size)
                if source_match and source_match.size > 1:
                    # substring from field_value corresponding to matched fv_tokens
                    match_str = field_value[match.b: match.b + match.size]
                    # substring from original s corresponding to source_match
                    source_match_str = s[source_match.start: source_match.start + source_match.size]

                    c_match_str = match_str.lower().strip()
                    c_source_match_str = source_match_str.lower().strip()
                    c_field_value = field_value.lower().strip()

                    # many early filters from original
                    if c_match_str and not is_number(c_match_str) and not is_common_db_term(c_match_str):
                        if is_stopword(c_match_str) or is_stopword(c_source_match_str) or is_stopword(c_field_value):
                            continue

                        # special-case possessive "'s"
                        if c_source_match_str.endswith(c_match_str + "'s"):
                            match_score = 1.0
                        else:
                            # prefix match check: use field value and source substring as in original
                            if _prefix_match(c_field_value, c_source_match_str):
                                match_score = fuzz.ratio(c_field_value, c_source_match_str) / 100.0
                            else:
                                match_score = 0.0

                        # original sets s_match_score = match_score
                        s_match_score = match_score

                        # commonword logic: if any of the three are commonwords and score < 1, skip
                        if (is_commonword(c_match_str) or is_commonword(c_source_match_str) or is_commonword(c_field_value)) and match_score < 1:
                            continue

                        # thresholds
                        if match_score >= m_theta and s_match_score >= s_theta:
                            # uppercase filter: if field_value is all uppercase and product < 1 skip
                            if field_value.isupper() and (match_score * s_match_score) < 1:
                                continue

                            # store result keyed by match_str (as original)
                            matched[match_str] = (field_value, source_match_str, match_score, s_match_score, match.size, item)

        if not matched:
            return None
        else:
            # sort by same composite key as original:
            # key = (1e16 * match_score + 1e8 * s_match_score + match.size)
            sorted_items = sorted(
                matched.items(),
                key=lambda x: (1e16 * x[1][2] + 1e8 * x[1][3] + x[1][4]),
                reverse=True,
            )
            return sorted_items

    def retrieve(
        self, processed_queries_batch: List[List[str]], output_path: str, k: int
    ) -> List[List[RetrievalResult]]:
        """
        processed_queries_batch: List of processed queries, where each entry is a list of tokens/strings.
        For compatibility with your earlier code, we assume processed_queries_batch[i][0] is the raw NLQ string.
        """
        self._load_index(output_path)

        final_results_batch: List[List[RetrievalResult]] = []

        for nlq_queries in tqdm(processed_queries_batch, desc="Retrieving with BRIDGE"):
            raw_nlq = nlq_queries[0]  # keep same assumption as before

            all_nlq_matches: List[RetrievalResult] = []

            for table in self._index_data:
                for column in self._index_data[table]:
                    searchable_items = self._index_data[table][column]

                    matched_entries = self._get_matched_entries(raw_nlq, searchable_items, self.match_threshold, self.s_match_threshold)

                    if not matched_entries:
                        continue

                    # matched_entries is list of (match_str, (field_value, source_match_str, match_score, s_match_score, match.size, item))
                    top_matches_for_column = matched_entries[: self.top_k_matches_per_column]

                    for match_str, (field_value, source_match_str, match_score, s_match_score, match_size, item) in top_matches_for_column:
                        composite_score = 1e16 * match_score + 1e8 * s_match_score + match_size
                        all_nlq_matches.append(RetrievalResult(item=item, score=composite_score))

            # sort globally and keep top-k
            sorted_nlq_matches = sorted(all_nlq_matches, key=lambda r: r.score, reverse=True)
            final_results_batch.append(sorted_nlq_matches[:k])

        return final_results_batch