from darelabdb.nlp_value_linking.filtering.filtering_abc import FilterABC
import json
import ast
import requests
from typing import List


class LLMFilter(FilterABC):
    """
    Filter implementation that uses an LLM to determine
    which values are likely to be used in a SQL WHERE clause based on a natural language query.

    The filter sends a prompt to the LLM with a query and candidate values, and parses the response
    to extract a list of values that are highly relevant.
    """

    def __init__(self):
        """
        Initialize the LLMFilter with configuration for the LLM model and endpoint.

        The configuration includes the model name, generation mode, options (such as temperature),
        and the base URL for the API.
        """
        self.OLLAMA_CONFIG = {
            "MODEL_NAME": "llama3.1:70b",
            "MODE": "generate",
            "OPTIONS": {
                "temperature": 0.01,
            },
        }
        self.OLLAMA_BASE_URL = "http://gaia-gpu-2.imsi.athenarc.gr:11434"
        self.queries_per_query = {}

    def pose_query(self, nl_question: str):
        # function to pose a query to the LLM model
        request_body = {
            "model": self.OLLAMA_CONFIG["MODEL_NAME"],
            "prompt": nl_question,
            "stream": False,
        }
        if "OPTIONS" in self.OLLAMA_CONFIG:
            request_body.update({"options": self.OLLAMA_CONFIG["OPTIONS"]})

        try:
            response = requests.post(
                f"{self.OLLAMA_BASE_URL}/api/generate",
                data=json.dumps(request_body),
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()

            generated_text = response.json()["response"]
            return generated_text

        except requests.exceptions.HTTPError as e:
            print(f"HTTPError: {e}")
            return None

    def add_pair(self, keyword: str, value_pair: tuple):
        if keyword not in self.queries_per_query:
            self.queries_per_query[keyword] = []
        self.queries_per_query[keyword].append(value_pair)

    def filter(self) -> List[str]:
        for query, value_pairs in self.queries_per_query.items():
            texts = [pair[0] for pair in value_pairs]
            # convert to a string
            texts_string = repr(texts)
            prompt = f"""You are part of a text-to-SQL system. Given a natural language query and a list of values from database columns (in 'table.column.value' format), your task is to filter and return ONLY the values that are highly likely to be used in the WHERE clause of the final SQL query.

                        Key requirements:
                        - Return values ONLY if they are highly relevant to the query's intent
                        - Focus on precision - avoid false positives
                        - Return an empty list if no values are clearly relevant
                        - Never return more than 5 values, as typically fewer values are correct
                        - Only return the Python list, no explanations
                        - If a value is exactly the same as the one mentioned in the query (case-insensitive), it should be included in the list (by values, we mean the 'value' part of 'table.column.value')
                        - Return the original string in the 'table.column.value' format
                        Query: {query}
                        Available values: {texts_string}

                        Return only a Python list containing the filtered values:"""
            response = self.pose_query(prompt)
            if response:
                try:
                    # First try to evaluate as Python literal
                    result_list = ast.literal_eval(response.strip())
                    if not isinstance(result_list, list):
                        result_list = []
                except (SyntaxError, ValueError):
                    try:
                        # Fallback to JSON parsing
                        result_list = json.loads(response.strip())
                    except json.JSONDecodeError:
                        print(f"Failed to parse response for question: {query}")
                        print(response)
                        result_list = []
        self.queries_per_query = {}
        return result_list
