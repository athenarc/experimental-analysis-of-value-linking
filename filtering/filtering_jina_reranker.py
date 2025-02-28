from darelabdb.nlp_value_linking.filtering.filtering_abc import FilterABC
import requests
from typing import List


class JinaAiRerankerFiltering(FilterABC):
    """
    Filter implementation that uses the Jina AI Reranker API.

    This filter sends queries and candidate value references to the Jina AI API and retains the candidate value references with
    relevance scores above a specified threshold.
    """

    def __init__(
        self, threshold=0.85, top_n=5, model="jina-reranker-v2-base-multilingual"
    ):
        """
        Initialize the JinaAiRerankerFiltering filter.

        Parameters:
            threshold (float): The relevance score threshold for filtering candidate value references .
            top_n (int): The number of top candidate value references to retrieve from the API.
            model (str): The model identifier to use for the reranker.
        """
        self.threshold = threshold
        self.top_n = top_n
        self.url = "https://api.jina.ai/v1/rerank"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer jina_a38d1327ae9b47a7b8dbb44c57e6a56bhEVC_cuFSIItiQSfQFmC729SUsKR",
        }
        self.model = model
        self.queries_per_keyword = {}

    def add_pair(self, keyword: str, value_pair: tuple):
        if keyword not in self.queries_per_keyword:
            self.queries_per_keyword[keyword] = []
        self.queries_per_keyword[keyword].append(value_pair)

    def filter(self) -> List[str]:
        filtered_values = []

        for keyword, value_pairs in self.queries_per_keyword.items():
            # Validate and sanitize inputs
            documents = [
                str(value_pair[0])
                for value_pair in value_pairs
                if isinstance(value_pair[0], (str, int, float))
            ]
            if not documents:
                continue
            # Prepare data for the API request
            data = {
                "model": self.model,
                "query": keyword,
                "top_n": self.top_n,
                "documents": documents,
            }

            try:
                # Make the HTTP POST request
                response = requests.post(self.url, headers=self.headers, json=data)

                # Check if the request was successful
                if response.status_code != 200:
                    print(
                        f"Error: Received status code {response.status_code} for keyword '{keyword}'."
                    )
                    continue

                # Parse JSON response
                response_data = response.json()

                # Validate response structure
                if "results" not in response_data:
                    continue

                # Process results
                for item in response_data["results"]:
                    if item.get("relevance_score", 0) > self.threshold:
                        document_text = item.get("document", {}).get("text", "")
                        if isinstance(document_text, str) and document_text.strip():
                            filtered_values.append(document_text)
                        else:
                            print(f"Invalid or empty document text for item: {item}")
            except requests.RequestException as e:
                print(f"HTTP request failed for keyword '{keyword}': {e}")
            except ValueError as e:
                print(f"Error parsing JSON response for keyword '{keyword}': {e}")

        # Clear queries after processing
        self.queries_per_keyword = {}
        return filtered_values
