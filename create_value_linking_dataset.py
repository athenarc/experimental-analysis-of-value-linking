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
    def __init__(self, schema_data: List[Dict]):
        # Initialize schema mapping from schema data
        self.schema_mapping = self._build_schema_mapping(schema_data)
        
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

    @staticmethod
    def _build_schema_mapping(schema_data: List[Dict]) -> Dict[str, Dict]:
        # Build a mapping of database IDs to their schema details
        schema_mapping = {}
        for schema in schema_data:
            schema_mapping[schema["db_id"]] = {
                "schema_items": [
                    {
                        "table_name": schema["table_names"][col[0]],
                        "column_names": [
                            schema["column_names"][idx][1].lower()
                            for idx in range(len(schema["column_names"]))
                            if schema["column_names"][idx][0] == col[0]
                        ],
                    }
                    for col in schema["column_names"]
                    if col[0] != -1
                ]
            }
        return schema_mapping

    def extract_tables_columns_and_values(
        self, sql_query: str, db_id: str, dialect: str = "sqlite"
    ) -> Tuple[List[str], List[str], List[Dict[str, str]]]:
        # Retrieve schema for the given database ID
        schema = self.schema_mapping.get(db_id, None)
        if not schema:
            return [], [], []

        def get_subquery_tables_columns_and_values(expression, cte_aliases):
            # Extract table names from the query, excluding CTE aliases
            tables = [
                t.name.lower()
                for t in expression.find_all(exp.Table)
                if t.name.lower() not in cte_aliases
            ]

            # Map table aliases to their original table names
            table_aliases = {
                t.alias.lower(): t.name.lower()
                for t in expression.find_all(exp.Table)
                if t.alias != ""
            }

            columns = []
            values = []
            # Extract columns from the query
            for c in expression.find_all(exp.Column):
                column_name = c.name.lower()
                table_name_or_alias = c.table.lower()

                if table_name_or_alias == "":
                    # Disambiguate columns when table name is not provided
                    if len(tables) == 1:
                        table_name = tables[0]
                    else:
                        table_name = ""
                        for table in schema["schema_items"]:
                            if (
                                column_name in table["column_names"]
                                and table["table_name"] in tables
                            ):
                                table_name = table["table_name"]
                                break
                        if table_name == "":
                            continue
                elif table_name_or_alias in table_aliases:
                    table_name = table_aliases[table_name_or_alias]
                elif table_name_or_alias in tables:
                    table_name = table_name_or_alias
                else:
                    continue

                columns.append(f"{table_name}.{column_name}")

            # Extract values from conditions in the query
            for condition in expression.find_all(exp.Condition):
                if isinstance(
                    condition,
                    (exp.EQ, exp.NEQ, exp.GT, exp.LT, exp.GTE, exp.LTE, exp.Like, exp.In),
                ):
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
                    operator = operator_map.get(
                        condition.__class__.__name__.lower(),
                        condition.__class__.__name__.lower(),
                    )

                    if isinstance(condition, exp.In):
                        left = condition.this
                        right = condition.expressions

                        if isinstance(left, exp.Column):
                            column_name = left.name.lower()
                            table_name = left.table.lower()

                            if table_name == "" and len(tables) == 1:
                                table_name = tables[0]
                            elif table_name in table_aliases:
                                table_name = table_aliases[table_name]

                            for literal in right:
                                if isinstance(literal, exp.Literal):
                                    values.append({
                                        "table": table_name,
                                        "column": column_name,
                                        "value": str(literal).strip("'\""),
                                    })
                    else:
                        left = condition.left
                        right = condition.right

                        if isinstance(left, exp.Column) and isinstance(right, exp.Literal):
                            column_name = left.name.lower()
                            table_name = left.table.lower()

                            if table_name == "" and len(tables) == 1:
                                table_name = tables[0]
                            elif table_name in table_aliases:
                                table_name = table_aliases[table_name]

                            values.append({
                                "table": table_name,
                                "column": column_name,
                                "value": str(right).strip("'\""),
                                "condition": operator,
                            })

            return tables, columns, values

        # Parse the SQL query
        expression = sqlglot.parse_one(sql_query, read=dialect)
        # Collect CTE aliases to distinguish them from actual tables
        cte_aliases = [cte.alias for cte in expression.find_all(exp.CTE)]

        # Collect sub-queries and process them in reverse order
        sub_queries = list(expression.find_all((exp.Subquery, exp.CTE), bfs=False))
        sub_queries.reverse()
        sub_queries.append(expression)

        tables = []
        columns = []
        values = []

        for sub_query in sub_queries:
            sub_tables, sub_columns, sub_values = get_subquery_tables_columns_and_values(
                sub_query, cte_aliases
            )
            sub_query.pop()
            tables.extend(sub_tables)
            columns.extend(sub_columns)
            values.extend(sub_values)

        return list(set(tables)), list(set(columns)), values


# Load JSON files
with open("dev_20240627/dev.json", "r") as f:
    train_data = json.load(f)
with open("dev_20240627/dev_tables.json", "r") as f:
    schema_data = json.load(f)
# Initialize the SQLQueryProcessor with schema data
processor = ValueLinkingDatasetProcessor(schema_data)


results = []
for query in train_data:
    sql_query = query["SQL"]
    db_id = query["db_id"]
    # Extract tables, columns, and values for the SQL query
    tables, columns, values = processor.extract_tables_columns_and_values(sql_query, db_id)
    results.append({
        "question": query["question"],
        "SQL": sql_query,
        "tables": tables,
        "columns": columns,
        "values": values,
    })

# Save results to a JSON file
output_file_path = "assets/value_linking_dataset.json"
with open(output_file_path, "w") as outfile:
    json.dump(results, outfile, indent=4)
    
print(f"Value linkning dataset has been saved to {output_file_path}")

output_file_path_list = "assets/value_linking_dataset_list.json"
processor.format_value_strings(output_file_path, output_file_path_list)