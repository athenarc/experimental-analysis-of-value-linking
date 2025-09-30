import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, NamedTuple
from tqdm import tqdm
import difflib
from rapidfuzz import fuzz
from darelabdb.nlp_retrieval.core.models import RetrievalResult, SearchableItem
from darelabdb.nlp_retrieval.retrievers.retriever_abc import BaseRetriever

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

_common_db_terms = {'id'}

_SPAN_SEPARATORS = frozenset('\'"()`,.?! ')

string_types = (str,)


def is_number(s):
    try:
        float(s.replace(',', ''))
        return True
    except ValueError:
        return False


def is_stopword(s):
    return s.strip() in _stopwords


def is_commonword(s):
    return s.strip() in _commonwords


def is_common_db_term(s):
    return s.strip() in _common_db_terms


class Match(object):
    def __init__(self, start: int, size: int):
        self.start = start
        self.size = size


def _is_span_separator(c: str) -> bool:
    return c in _SPAN_SEPARATORS


def _split(s: str) -> List[str]:
    return [c.lower() for c in s.strip()]


def _prefix_match(s1: str, s2: str) -> bool:
    i, j = 0, 0
    len_s1, len_s2 = len(s1), len(s2)
    while i < len_s1 and _is_span_separator(s1[i]):
        i += 1
    while j < len_s2 and _is_span_separator(s2[j]):
        j += 1
    
    if i < len_s1 and j < len_s2:
        return s1[i] == s2[j]
    return i >= len_s1 and j >= len_s2


def _get_effective_match_source(s: str, start: int, end: int) -> Optional[Match]:
    _start = -1
    for i in range(start, start - 3, -1):
        if i < 0:
            _start = 0
            break
        if _is_span_separator(s[i]):
            _start = i
            break
    if _start < 0:
        return None

    _end = -1
    len_s = len(s)
    for i in range(end - 1, end + 3):
        if i >= len_s:
            _end = len_s - 1
            break
        if _is_span_separator(s[i]):
            _end = i
            break
    if _end < 0:
        return None

    while _start < len_s and _is_span_separator(s[_start]):
        _start += 1
    while _end >= 0 and _is_span_separator(s[_end]):
        _end -= 1
    return Match(_start, end - start)
class _PrecomputedItem(NamedTuple):
    original_item: SearchableItem
    field_value: str
    fv_tokens: List[str]
    c_field_value: str
    is_fv_common: bool
    is_fv_stopword: bool
    is_fv_upper: bool


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
        self._index_data: Dict[str, Dict[str, List[_PrecomputedItem]]] = defaultdict(
            lambda: defaultdict(list)
        )

    def index(self, items: List[SearchableItem], output_path: str) -> None:
        index_file = os.path.join(output_path, "bridge_index.jsonl")

        with open(index_file, "w") as f:
            for item in items:
                f.write(item.model_dump_json() + "\n")

    def _load_index(self, output_path: str) -> None:
        """
        OPTIMIZATION: This method is updated to pre-compute and cache values
        (like tokenized forms, lowercase versions, etc.) for each item upon loading.
        This prevents these expensive operations from being repeated for every query.
        """
        index_file = os.path.join(output_path, "bridge_index.jsonl")
        if self._index_data: # Avoid reloading if already loaded
            return
            
        with open(index_file, "r") as f:
            for line in f:
                item_dict = json.loads(line)
                item = SearchableItem(**item_dict)
                table = item.metadata.get("table")
                column = item.metadata.get("column")

                if not (table and column):
                    continue

                field_value = item.metadata.get("value")
                # Pre-computation only applies to string values, mirroring the original logic's filter.
                if not isinstance(field_value, string_types):
                    continue

                c_field_value = field_value.lower().strip()

                precomputed = _PrecomputedItem(
                    original_item=item,
                    field_value=field_value,
                    fv_tokens=_split(field_value),
                    c_field_value=c_field_value,
                    is_fv_common=is_commonword(c_field_value),
                    is_fv_stopword=is_stopword(c_field_value),
                    is_fv_upper=field_value.isupper()
                )
                self._index_data[table][column].append(precomputed)

    def _get_matched_entries(
        self,
        s: str,
        precomputed_items: List[_PrecomputedItem],
        m_theta: float = 0.85,
        s_theta: float = 0.85,
    ) -> Optional[List[Tuple[str, Tuple[str, str, float, float, int, SearchableItem]]]]:
        """
        OPTIMIZATION: Replicates the original get_matched_entries logic but is significantly
        faster because it operates on the _PrecomputedItem objects, avoiding repeated
        string operations and function calls inside the main loop.
        """
        if not precomputed_items:
            return None

        n_grams = _split(s)
        len_n_grams = len(n_grams)
        matched = {}

        for p_item in precomputed_items:
            # All item-specific values are now read from the precomputed p_item.
            len_fv_tokens = len(p_item.fv_tokens)
            sm = difflib.SequenceMatcher(None, n_grams, p_item.fv_tokens, autojunk=False)
            match = sm.find_longest_match(0, len_n_grams, 0, len_fv_tokens)

            if match.size > 0:
                source_match = _get_effective_match_source(s, match.a, match.a + match.size)
                if source_match and source_match.size > 1:
                    match_str = p_item.field_value[match.b: match.b + match.size]
                    source_match_str = s[source_match.start: source_match.start + source_match.size]

                    c_match_str = match_str.lower().strip()
                    c_source_match_str = source_match_str.lower().strip()

                    if c_match_str and not is_number(c_match_str) and not is_common_db_term(c_match_str):
                        if is_stopword(c_match_str) or is_stopword(c_source_match_str) or p_item.is_fv_stopword:
                            continue

                        if c_source_match_str.endswith(f"{c_match_str}'s"):
                            match_score = 1.0
                        else:
                            if _prefix_match(p_item.c_field_value, c_source_match_str):
                                match_score = fuzz.ratio(p_item.c_field_value, c_source_match_str) / 100.0
                            else:
                                match_score = 0.0

                        s_match_score = match_score

                        if (is_commonword(c_match_str) or is_commonword(c_source_match_str) or p_item.is_fv_common) and match_score < 1:
                            continue

                        if match_score >= m_theta and s_match_score >= s_theta:
                            if p_item.is_fv_upper and (match_score * s_match_score) < 1:
                                continue
                            match_key = (match_str, p_item.field_value)
                            matched[match_key] = (p_item.field_value, source_match_str, match_score, s_match_score, match.size, p_item.original_item)

        if not matched:
            return None
        else:
            # The original key was x[0] (match_str). We now use match_key[0] for the same effect.
            sorted_items = sorted(
                matched.items(),
                key=lambda x: (1e16 * x[1][2] + 1e8 * x[1][3] + x[1][4]),
                reverse=True,
            )
            # Reformat the output to match the original structure
            return [(item[0][0], item[1]) for item in sorted_items]

    def retrieve(
        self, processed_queries_batch: List[List[str]], output_path: str, k: int
    ) -> List[List[RetrievalResult]]:
        """
        processed_queries_batch: List of processed queries, where each entry is a list of tokens/strings.
        For compatibility, we assume processed_queries_batch[i][0] is the raw NLQ string.
        """
        self._load_index(output_path)

        final_results_batch: List[List[RetrievalResult]] = []

        for nlq_tokens in tqdm(processed_queries_batch, desc="Retrieving with BRIDGE"):
            # The first element is assumed to be the raw, unprocessed query string.
            raw_nlq = nlq_tokens[0] 

            all_nlq_matches: List[RetrievalResult] = []

            for table in self._index_data:
                for column in self._index_data[table]:
                    precomputed_items = self._index_data[table][column]

                    matched_entries = self._get_matched_entries(
                        raw_nlq, 
                        precomputed_items, 
                        self.match_threshold, 
                        self.s_match_threshold
                    )

                    if not matched_entries:
                        continue

                    top_matches_for_column = matched_entries[: self.top_k_matches_per_column]

                    for match_str, (field_value, source_match_str, match_score, s_match_score, match_size, item) in top_matches_for_column:
                        composite_score = 1e16 * match_score + 1e8 * s_match_score + match_size
                        all_nlq_matches.append(RetrievalResult(item=item, score=composite_score))

            sorted_nlq_matches = sorted(all_nlq_matches, key=lambda r: r.score, reverse=True)
            final_results_batch.append(sorted_nlq_matches[:k])

        return final_results_batch