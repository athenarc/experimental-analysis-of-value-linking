import json
import random
import string
import re
import nltk
from nltk.corpus import wordnet
from typing import List, Dict, Tuple, Optional
import sqlglot
import sqlglot.expressions as exp
import os


nltk.download("wordnet", quiet=True)


class ValueLinkingDatasetProcessor:
    """Processes dataset for value linking tasks including formatting, typos, synonyms, and predictions."""

    def format_value_strings(self, input_path, output_path):
        """Formats values into 'table.column.value' strings and saves them.

        Args:
            input_path (str): Path to input JSON file
            output_path (str): Path to save formatted JSON
        """
        with open(input_path, "r") as file:
            data = json.load(file)

        results = []
        for record in data:
            value_strings = [
                f"{v['table']}.{v['column']}.{v['value']}".lower()
                for v in record["values"]
            ]
            results.append(value_strings)

        with open(output_path, "w") as output_file:
            json.dump(results, output_file, indent=4)

    def filter_records(self, input_path, output_path):
        """Filters records not containin values in WHERE clauses.

        Args:
            input_path (str): Path to input JSON file
            output_path (str): Path to save filtered JSON
        """
        with open(input_path, "r") as f:
            data = json.load(f)

        filtered = []
        for record in data:
            if record.get("values"):
                all_strings = all(
                    isinstance(v.get("value"), str)
                    and re.search(r"[a-zA-Z]", v.get("value"))
                    for v in record["values"]
                )
                if all_strings:
                    filtered.append(
                        {
                            "values": record.get("values"),
                            "question": record.get("question"),
                            "db_id": record.get("schema", {}).get("db_id"),
                        }
                    )

        with open(output_path, "w") as f:
            json.dump(filtered, f, indent=4)

    def _introduce_spelling_error(self, word):
        """Randomly adds or removes a character from a word."""
        if len(word) < 1:
            return word

        if random.choice([True, False]):
            pos = random.randint(0, len(word))
            return word[:pos] + random.choice(string.ascii_letters) + word[pos:]
        return (
            word[: random.randint(0, len(word) - 1)]
            + word[random.randint(0, len(word) - 1) + 1 :]
        )

    def introduce_typos(self, input_path, output_path):
        """Introduces spelling errors to value tokens in questions.

        Args:
            input_path (str): Path to input JSON file
            output_path (str): Path to save modified JSON
        """
        with open(input_path, "r") as f:
            data = json.load(f)

        output = []
        for record in data:
            tokens = {
                token.lower()
                for value in record["values"]
                for token in value["value"].split()
            }
            modified_words = []
            changed = False

            for word in record["question"].split():
                lower_word = word.lower()
                if lower_word in tokens:
                    modified = self._introduce_spelling_error(word)
                    changed |= modified != word
                    modified_words.append(modified)
                else:
                    modified_words.append(word)

            if changed:
                output.append(
                    {
                        "question": " ".join(modified_words),
                        "values": record["values"],
                        "db_id": record["db_id"],
                    }
                )

        with open(output_path, "w") as f:
            json.dump(output, f, indent=4)

    def _get_most_relevant_synonym(self, word):
        """Finds most semantically relevant synonym using WordNet."""
        synsets = wordnet.synsets(word)
        if not synsets:
            return word

        primary = synsets[0]
        candidates = []

        for syn in synsets:
            for lemma in syn.lemmas():
                synonym = lemma.name().replace("_", " ")
                if synonym.lower() != word:
                    if synset := wordnet.synsets(synonym):
                        similarity = primary.path_similarity(synset[0])
                        if similarity:
                            candidates.append((synonym, similarity))

        return max(candidates, key=lambda x: x[1], default=(word, 0))[0]

    def introduce_synonyms(self, input_path, output_path):
        """Replaces value tokens with synonyms in questions.

        Args:
            input_path (str): Path to input JSON file
            output_path (str): Path to save modified JSON
        """
        with open(input_path, "r") as f:
            data = json.load(f)

        output = []
        for record in data:
            tokens = {
                token.lower(): token
                for value in record["values"]
                for token in value["value"].split()
            }
            modified_words = []
            changed = False

            for word in record["question"].split():
                stripped = word.strip(",.!?;:")
                if stripped.lower() in tokens:
                    synonym = self._get_most_relevant_synonym(stripped.lower())
                    formatted = (
                        synonym.lower() if word.islower() else synonym.capitalize()
                    )
                    changed |= formatted != stripped
                    modified_words.append(formatted)
                else:
                    modified_words.append(word)

            if changed:
                output.append(
                    {
                        "question": " ".join(modified_words),
                        "values": record["values"],
                        "db_id": record["db_id"],
                    }
                )

        with open(output_path, "w") as f:
            json.dump(output, f, indent=4)

    def generate_predictions_with_precision(
        self, pred_path, gt_path, precision, output_path
    ):
        """Generates predictions calibrated to target precision.

        Args:
            pred_path (str): Path to predicted JSON file
            gt_path (str): Path to ground truth JSON file
            precision (float): Target precision between 0-1
            output_path (str): Path to save calibrated predictions
        """
        if not 0 <= precision <= 1:
            raise ValueError("Precision must be between 0 and 1")

        with open(pred_path) as pred_file, open(gt_path) as gt_file:
            pred_data = json.load(pred_file)
            gt_data = json.load(gt_file)

        if len(pred_data) != len(gt_data):
            raise ValueError("Input files must have same number of records")

        results = []
        for preds, truths in zip(pred_data, gt_data):
            combined = set(truths)
            if precision < 1:
                required = int(len(truths) / precision)
                available = set(preds) - combined
                combined.update(
                    random.sample(
                        list(available), min(required - len(combined), len(available))
                    )
                )
            results.append(list(combined))

        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)

    def build_schema_mapping(self, raw_schemas: List[Dict]) -> Dict[str, Dict]:
        """Convert raw schema data into a standardized format for query processing.

        Args:
            raw_schemas: List of schema definitions from different databases

        Returns:
            Dictionary mapping database IDs to their normalized schemas
        """
        schema_map = {}
        for schema in raw_schemas:
            db_id = schema["db_id"]
            tables = {}

            # Process columns into table-based structure
            for col_info in schema["column_names"]:
                table_idx = col_info[0]
                if table_idx == -1:  # Skip special columns
                    continue

                table_name = schema["table_names"][table_idx].lower()
                column_name = col_info[1].lower()

                if table_name not in tables:
                    tables[table_name] = []
                tables[table_name].append(column_name)

            # Convert to final schema format
            schema_items = [
                {"table_name": table, "column_names": columns}
                for table, columns in tables.items()
            ]

            schema_map[db_id] = {"schema_items": schema_items}

        return schema_map

    def process_sql_query(
        self,
        sql_query: str,  # First parameter after self
        schema: Dict,  # Second parameter
        dialect: str = "sqlite",  # Should come last
    ) -> Tuple[List[str], List[str], List[Dict]]:
        """Extract tables, columns, and values from a SQL query using database schema.

        Args:
            sql_query: Input SQL query string
            schema: Preprocessed schema for the target database
            dialect: SQL dialect for parsing

        Returns:
            Tuple containing:
            - List of table names
            - List of column names (with table prefixes)
            - List of value conditions
        """

        def resolve_table_for_column(
            column_name: str, base_tables: List[str], schema_items: List[Dict]
        ) -> Optional[str]:
            """Resolve table name for columns without explicit table references."""
            if len(base_tables) == 1:
                return base_tables[0]

            for table in schema_items:
                if (
                    column_name in table["column_names"]
                    and table["table_name"] in base_tables
                ):
                    return table["table_name"]
            return None

        def resolve_table_name(
            table_alias: str, table_map: Dict[str, str], tables: List[str]
        ) -> Optional[str]:
            """Resolve table aliases to real names using context."""
            if table_alias in table_map:
                return table_map[table_alias]
            return table_alias if table_alias in tables else None

        def extract_from_expression(
            expression: exp.Expression, schema_items: List[Dict], cte_aliases: List[str]
        ) -> Tuple[List[str], List[str], List[Dict]]:
            """Recursively process SQL expressions."""
            tables = []
            columns = []
            values = []
            operator_map = {
                "eq": "=",
                "neq": "!=",
                "gt": ">",
                "lt": "<",
                "gte": ">=",
                "lte": "<=",
                "like": "LIKE",
                "in": "IN",
            }

            # Extract base tables
            base_tables = [
                t.name.lower()
                for t in expression.find_all(exp.Table)
                if t.name.lower() not in cte_aliases
            ]

            # Build alias mapping
            table_aliases = {
                t.alias.lower(): t.name.lower()
                for t in expression.find_all(exp.Table)
                if t.alias
            }

            # Process columns
            for col in expression.find_all(exp.Column):
                col_table = col.table.lower() if col.table else ""
                col_name = col.name.lower()

                # Resolve table name
                resolved_table = resolve_table_name(
                    col_table, table_aliases, base_tables
                )
                if not resolved_table:
                    # Try column-based disambiguation
                    for table in schema_items:
                        if (
                            col_name in table["column_names"]
                            and table["table_name"] in base_tables
                        ):
                            resolved_table = table["table_name"]
                            break

                if resolved_table:
                    columns.append(f"{resolved_table}.{col_name}")

            # Process conditions
            condition_types = (
                exp.EQ,
                exp.NEQ,
                exp.GT,
                exp.LT,
                exp.GTE,
                exp.LTE,
                exp.Like,
                exp.In,
            )
            for condition in expression.find_all(exp.Condition):
                if isinstance(condition, condition_types):
                    operator = operator_map.get(
                        condition.__class__.__name__.lower(),
                        condition.key,  # fallback to SQLGlot's built-in key
                    )
                    left = (
                        condition.left if hasattr(condition, "left") else condition.this
                    )
                    right = (
                        condition.right
                        if hasattr(condition, "right")
                        else condition.expressions
                    )

                    if isinstance(left, exp.Column):
                        # Handle both explicit and implicit table references
                        col_table = left.table.lower() if left.table else ""
                        col_name = left.name.lower()

                        # First try explicit table resolution
                        table_name = resolve_table_name(
                            col_table, table_aliases, base_tables
                        )

                        # If that fails, try schema-based resolution
                        if not table_name:
                            table_name = resolve_table_for_column(
                                col_name, base_tables, schema_items
                            )

                        # Process value if we resolved the table
                        if table_name:
                            if isinstance(condition, exp.In):
                                left = condition.this
                                right = condition.expressions
                                for expr in right:
                                    if isinstance(expr, exp.Literal):
                                        values.append(
                                            {
                                                "table": table_name,
                                                "column": col_name,
                                                "value": expr.this.strip("'\""),
                                                "condition": operator,
                                            }
                                        )
                            elif isinstance(right, exp.Literal):
                                values.append(
                                    {
                                        "table": table_name,
                                        "column": col_name,
                                        "value": right.this.strip("'\""),
                                        "condition": operator,
                                    }
                                )

            # Recursively process subqueries
            for subquery in expression.find_all((exp.Subquery, exp.CTE)):
                sub_tables, sub_columns, sub_values = extract_from_expression(
                    subquery.this, schema_items, cte_aliases
                )
                tables += sub_tables
                columns += sub_columns
                values += sub_values

            return base_tables, columns, values

        # Main processing logic
        if not schema or "schema_items" not in schema:
            return [], [], []

        try:
            parsed = sqlglot.parse_one(sql_query, read=dialect)
            cte_aliases = [cte.alias.lower() for cte in parsed.find_all(exp.CTE)]
            tables, columns, values = extract_from_expression(
                parsed, schema["schema_items"], cte_aliases
            )
            return (
                list(set(tables)),
                list(set(columns)),
                list({v["value"]: v for v in values}.values()),  # Deduplicate
            )
        except sqlglot.errors.ParseError:
            return [], [], []



# Load JSON files
with open("dev_20240627/dev.json", "r") as f:
    train_data = json.load(f)
with open("dev_20240627/dev_tables.json", "r") as f:
    schema_data = json.load(f)
# Initialize the SQLQueryProcessor with schema data
processor = ValueLinkingDatasetProcessor()
schema_mapping = processor.build_schema_mapping(schema_data)


def process_queries(queries: List[Dict], output_path: str):
    results = []
    for query in queries:
        db_id = query["db_id"]
        schema = schema_mapping.get(db_id, {})
        tables, columns, values = processor.process_sql_query(
            sql_query=query["SQL"],
            schema=schema_mapping[db_id],  # Schema passed as positional argument
            dialect="mysql"                # Dialect as explicit keyword
        )
        
        results.append({
            "question": query["question"],
            "SQL": query["SQL"],
            "tables": tables,
            "columns": columns,
            "values": values
        })
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

output_file_path = "assets/value_linking_dataset.json"

process_queries(train_data, output_file_path)
print(f"Results have been saved to {output_file_path}")

output_file_path_list = "assets/value_linking_dataset_list.json"
processor.format_value_strings(output_file_path, output_file_path_list)