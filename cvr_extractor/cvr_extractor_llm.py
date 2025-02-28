from darelabdb.nlp_value_linking.cvr_extractor.cvr_extractor_abc import CVRExtractorABC
import json
import requests
from typing import List


class LLMExtractor(CVRExtractorABC):
    """CVR extractor using LLM via Ollama API.

    Uses LLM capabilities to identify value references through
    structured prompting."""

    def __init__(self):
        self.OLLAMA_CONFIG = {
            "MODEL_NAME": "llama3.1:70b",
            "MODE": "generate",
            "OPTIONS": {
                "temperature": 0.01,
            },
        }
        self.OLLAMA_BASE_URL = "http://gaia-gpu-2.imsi.athenarc.gr:11434"

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

    def extract_keywords(self, input_text: str) -> List[str]:
        prompt = f"""
                    Objective: Analyze the given question to identify and extract keywords, keyphrases, and named entities. These elements are crucial for understanding the core components of the inquiry and the guidance provided. This process involves recognizing and isolating significant terms and phrases that could be instrumental in formulating searches or queries related to the posed question.

                    Instructions:

                    Read the Question Carefully: Understand the primary focus and specific details of the question. Look for any named entities (such as organizations, locations, etc.), technical terms, and other phrases that encapsulate important aspects of the inquiry.

                    List Keyphrases and Entities: Based on your findings from the question create a single Python list. This list should contain:

                    Keywords: Single words that capture essential aspects of the question.
                    Keyphrases: Short phrases or named entities that represent specific concepts, locations, organizations, or other significant details.
                    Ensure to maintain the original phrasing or terminology used in the question.

                    Example 1:
                    Question: "What is the annual revenue of Acme Corp in the United States for 2022?"

                    ["annual revenue", "Acme Corp", "United States", "2022"]

                    Example 2:
                    Question: "In the Winter and Summer Olympics of 1988, which game has the most number of competitors? Find the difference of the number of competitors between the two games."

                    ["Winter Olympics", "Summer Olympics", "1988", "1988 Summer", "Summer", "1988 Winter", "Winter", "number of competitors", "difference", "games"]

                    Example 3:
                    Question: "How many Men's 200 Metres Freestyle events did Ian James Thorpe compete in?"

                    ["Men's 200 metres Freestyle", "Ian James Thorpe", "Ian", "James", "Thorpe", "compete in", "event"]

                    Task:
                    Given the following question, identify and list all relevant keywords, keyphrases, and named entities.

                    Question: {input_text}

                    Please provide your findings as a Python list, capturing the essence of the question through the identified terms and phrases. 
                    Only output the Python list, no explanations needed. 
                """  # prompt to extract keywords from the input text follwoing CHESS
        response = self.pose_query(prompt)
        result_list = []
        # clear cache
        if response:
            try:
                result_list = json.loads(response)
            except json.JSONDecodeError:
                print(f"Failed to parse response for question: {input_text}")
                result_list = []
        return result_list
