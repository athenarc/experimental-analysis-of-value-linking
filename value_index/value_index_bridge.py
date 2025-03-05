from typing import List, Tuple
from utils.sqlite_db import DatabaseSqlite
import difflib
from typing import List, Optional, Tuple, Any
from value_index.value_index_abc import ValueIndexABC
from rapidfuzz import fuzz
from filtering.filtering_abc import FilterABC
from pathlib import Path
INDEXES_CACHE_PATH = str(Path.home()) + "/.cache/value_linking/"

class Match(object):
    def __init__(self, start: int, size: int) -> None:
        self.start = start
        self.size = size


class BRIDGEIndex(ValueIndexABC):
    def __init__(self, top_k=1, threshold=0.85):
        self.picklist_indexes = dict()
        self.match_threshold = threshold
        self.stopwords = {
            "who",
            "ourselves",
            "down",
            "only",
            "were",
            "him",
            "at",
            "weren't",
            "has",
            "few",
            "it's",
            "m",
            "again",
            "d",
            "haven",
            "been",
            "other",
            "we",
            "an",
            "own",
            "doing",
            "ma",
            "hers",
            "all",
            "haven't",
            "in",
            "but",
            "shouldn't",
            "does",
            "out",
            "aren",
            "you",
            "you'd",
            "himself",
            "isn't",
            "most",
            "y",
            "below",
            "is",
            "wasn't",
            "hasn",
            "them",
            "wouldn",
            "against",
            "this",
            "about",
            "there",
            "don",
            "that'll",
            "a",
            "being",
            "with",
            "your",
            "theirs",
            "its",
            "any",
            "why",
            "now",
            "during",
            "weren",
            "if",
            "should",
            "those",
            "be",
            "they",
            "o",
            "t",
            "of",
            "or",
            "me",
            "i",
            "some",
            "her",
            "do",
            "will",
            "yours",
            "for",
            "mightn",
            "nor",
            "needn",
            "the",
            "until",
            "couldn't",
            "he",
            "which",
            "yourself",
            "to",
            "needn't",
            "you're",
            "because",
            "their",
            "where",
            "it",
            "didn't",
            "ve",
            "whom",
            "should've",
            "can",
            "shan't",
            "on",
            "had",
            "have",
            "myself",
            "am",
            "don't",
            "under",
            "was",
            "won't",
            "these",
            "so",
            "as",
            "after",
            "above",
            "each",
            "ours",
            "hadn",
            "having",
            "wasn",
            "s",
            "doesn",
            "hadn't",
            "than",
            "by",
            "that",
            "both",
            "herself",
            "his",
            "wouldn't",
            "into",
            "doesn't",
            "before",
            "my",
            "won",
            "more",
            "are",
            "through",
            "same",
            "how",
            "what",
            "over",
            "ll",
            "yourselves",
            "up",
            "mustn",
            "mustn't",
            "she's",
            "re",
            "such",
            "didn",
            "you'll",
            "shan",
            "when",
            "you've",
            "themselves",
            "mightn't",
            "she",
            "from",
            "isn",
            "ain",
            "between",
            "once",
            "here",
            "shouldn",
            "our",
            "and",
            "not",
            "too",
            "very",
            "further",
            "while",
            "off",
            "couldn",
            "hasn't",
            "itself",
            "then",
            "did",
            "just",
            "aren't",
        }
        self.commonwords = {"no", "yes", "many"}
        self.m_theta = threshold
        self.s_theta = threshold

    def is_number(self, s: str) -> bool:
        try:
            float(s.replace(",", ""))
            return True
        except:
            return False

    def is_stopword(self, s: str) -> bool:
        return s.strip() in self.stopwords

    def is_commonword(self, s: str) -> bool:
        return s.strip() in self.commonwords

    def is_common_db_term(self, s: str) -> bool:
        return s.strip() in ["id"]

    def is_span_separator(self, c: str) -> bool:
        return c in "'\"()`,.?! "

    def split_to_chars(self, s: str) -> List[str]:
        return [c.lower() for c in s.strip()]

    def prefix_match(self, s1: str, s2: str) -> bool:
        i, j = 0, 0
        for i in range(len(s1)):
            if not self.is_span_separator(s1[i]):
                break
        for j in range(len(s2)):
            if not self.is_span_separator(s2[j]):
                break
        if i < len(s1) and j < len(s2):
            return s1[i] == s2[j]
        elif i >= len(s1) and j >= len(s2):
            return True
        else:
            return False

    def get_effective_match_source(self, s: str, start: int, end: int) -> Match:
        _start = -1

        for i in range(start, start - 2, -1):
            if i < 0:
                _start = i + 1
                break
            if self.is_span_separator(s[i]):
                _start = i
                break

        if _start < 0:
            return None

        _end = -1
        for i in range(end - 1, end + 3):
            if i >= len(s):
                _end = i - 1
                break
            if self.is_span_separator(s[i]):
                _end = i
                break

        if _end < 0:
            return None

        while _start < len(s) and self.is_span_separator(s[_start]):
            _start += 1
        while _end >= 0 and self.is_span_separator(s[_end]):
            _end -= 1

        return Match(_start, _end - _start + 1)

    def get_matched_entries(
        self, s: str, field_values: List[str]
    ) -> Optional[List[Tuple[str, Tuple[str, str, float, float, int]]]]:
        if not field_values:
            return None

        if isinstance(s, str):
            n_grams = self.split_to_chars(s)
        else:
            n_grams = s

        matched = dict()
        for field_value in field_values:
            if not isinstance(field_value, str):
                continue
            fv_tokens = self.split_to_chars(field_value)
            sm = difflib.SequenceMatcher(None, n_grams, fv_tokens)
            match = sm.find_longest_match(0, len(n_grams), 0, len(fv_tokens))
            if match.size > 0:
                source_match = self.get_effective_match_source(
                    n_grams, match.a, match.a + match.size
                )
                if source_match:
                    match_str = field_value[match.b : match.b + match.size]
                    source_match_str = s[
                        source_match.start : source_match.start + source_match.size
                    ]
                    c_match_str = match_str.lower().strip()
                    c_source_match_str = source_match_str.lower().strip()
                    c_field_value = field_value.lower().strip()
                    if c_match_str and not self.is_common_db_term(c_match_str):
                        if (
                            self.is_stopword(c_match_str)
                            or self.is_stopword(c_source_match_str)
                            or self.is_stopword(c_field_value)
                        ):
                            continue
                        if c_source_match_str.endswith(c_match_str + "'s"):
                            match_score = 1.0
                        else:
                            if self.prefix_match(c_field_value, c_source_match_str):
                                match_score = (
                                    fuzz.ratio(c_field_value, c_source_match_str) / 100
                                )
                            else:
                                match_score = 0
                        if (
                            self.is_commonword(c_match_str)
                            or self.is_commonword(c_source_match_str)
                            or self.is_commonword(c_field_value)
                        ) and match_score < 1:
                            continue
                        s_match_score = match_score
                        if (
                            match_score >= self.m_theta
                            and s_match_score >= self.s_theta
                        ):
                            if (
                                field_value.isupper()
                                and match_score * s_match_score < 1
                            ):
                                continue
                            matched[match_str] = (
                                field_value,
                                source_match_str,
                                match_score,
                                s_match_score,
                                match.size,
                            )

        if not matched:
            return None
        else:
            return sorted(
                matched.items(),
                key=lambda x: (1e16 * x[1][2] + 1e8 * x[1][3] + x[1][4]),
                reverse=True,
            )

    def create_index(
        self, database: DatabaseSqlite, output_path=INDEXES_CACHE_PATH
    ):
        pass

    def query_index(
        self,
        keywords: str,
        index_path=INDEXES_CACHE_PATH,
        top_k=1,
        filter_instance: FilterABC = None,
        database: DatabaseSqlite = None,
    ):
        nl_query = keywords[0]
        matched_values = []
        schema = database.get_tables_and_columns()
        tables = [table for table in schema["tables"] if table != "sqlite_sequence"]

        for table in tables:
            # Get columns for current table
            columns_info = database.execute(f'PRAGMA table_info("{table}");', limit=-1)
            columns = columns_info.to_dict("records")

            for column in columns:
                # Check cache first
                col_name = column["name"]
                cache_key = (table, col_name)

                # Build picklist if not cached
                if cache_key not in self.picklist_indexes:
                    # Get value counts from database
                    print(f"Creating picklist for {table}.{col_name}")
                    try:
                        query = f"SELECT DISTINCT `{col_name}` FROM `{table}` WHERE `{col_name}` IS NOT NULL;"

                        values_df = database.execute(query, limit=-1)
                        values = values_df[col_name].dropna().tolist()

                        self.picklist_indexes[cache_key] = values
                    except Exception as e:
                        print(f"Error processing {table}.{col_name}: {e}")
                        continue

                # Get matches for current picklist
                picklist = self.picklist_indexes.get(cache_key, [])
                if not picklist:
                    continue
                matches = self.get_matched_entries(nl_query, picklist)
                if matches:
                    added = 0
                    for match in matches:
                        if added >= top_k:
                            break

                        value = match[0]  # Adjust based on match structure
                        matched_values.append(f"{table}.{col_name}.{value}")
                        added += 1

        return list(set(matched_values))
