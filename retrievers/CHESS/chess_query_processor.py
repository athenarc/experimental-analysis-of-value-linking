import ast
import json
import os
import re
from typing import Dict, List, Optional

from darelabdb.nlp_retrieval.user_query_processors.query_processor_abc import (
    BaseUserQueryProcessor,
)
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


class ChessQueryProcessor(BaseUserQueryProcessor):
    """
    A query processor that adapts the CHESS framework's LLM-based keyword
    extraction tool using VLLM for high throughput.
    """

    def __init__(
        self,
        model_name_or_path: str,
        cache_folder: Optional[str] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        **kwargs,
    ):
        """
        Initializes the VLLM-based keyword extraction processor.

        Args:
            model_name_or_path: The name or path of the model for VLLM.
            cache_folder: An optional path to a folder for caching results.
            tensor_parallel_size: The number of GPUs for tensor parallelism.
            gpu_memory_utilization: Fraction of GPU memory for VLLM.
            **kwargs: Additional arguments for VLLM's LLM class or SamplingParams.
        """
        self.model_name_or_path = model_name_or_path
        self.cache_folder = cache_folder
        self.keywords_cache: Optional[Dict[str, List[str]]] = None

        if self.cache_folder:
            os.makedirs(self.cache_folder, exist_ok=True)
            self.cache_file = os.path.join(
                self.cache_folder, "chess_keywords_cache.json"
            )
            self.keywords_cache = self._load_cache()

        self.vllm_args = {
            "trust_remote_code": kwargs.pop("trust_remote_code", True),
            "max_model_len": kwargs.pop("max_model_len", 4096),
        }
        self.sampling_args = {
            "temperature": kwargs.pop("temperature", 0.0),
            "top_p": kwargs.pop("top_p", 1.0),
            "max_tokens": kwargs.pop("max_tokens", 512),
        }
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tokenizer = None
        self.llm = None
        self.sampling_params = None
        self._build_prompt_template()

    def _build_prompt_template(self):
        """
        Constructs the few-shot prompt for keyword extraction, adapted from
        the CHESS framework's `template_extract_keywords.txt`.
        """
        system_prompt = """You are an expert at keyword extraction. Your goal is to analyze the given question and hint to identify and extract keywords, keyphrases, and named entities. These elements are crucial for understanding the core components of the inquiry. List all relevant keywords, keyphrases, and named entities as a Python list of strings.

Only output the Python list. Do not include any introduction, explanation, or markdown formatting."""

        examples_data = [
            {
                "input": "Question: \"What is the annual revenue of Acme Corp in the United States for 2022?\"\nHint: \"Focus on financial reports and U.S. market performance for the fiscal year 2022.\"",
                "output": '["annual revenue", "Acme Corp", "United States", "2022", "financial reports", "U.S. market performance", "fiscal year"]',
            },
            {
                "input": "Question: \"In the Winter and Summer Olympics of 1988, which game has the most number of competitors? Find the difference of the number of competitors between the two games.\"\nHint: \"the most number of competitors refer to MAX(COUNT(person_id)); SUBTRACT(COUNT(person_id where games_name = '1988 Summer'), COUNT(person_id where games_name = '1988 Winter'));\"",
                "output": '["Winter Olympics", "Summer Olympics", "1988", "1988 Summer", "Summer", "1988 Winter", "Winter", "number of competitors", "difference", "MAX(COUNT(person_id))", "games_name", "person_id"]',
            },
            {
                "input": "Question: \"How many Men's 200 Metres Freestyle events did Ian James Thorpe compete in?\"\nHint: \"Men's 200 Metres Freestyle events refer to event_name = 'Swimming Men''s 200 metres Freestyle'; events compete in refers to event_id;\"",
                "output": '["Swimming Men\'s 200 metres Freestyle", "Ian James Thorpe", "Ian", "James", "Thorpe", "compete in", "event_name", "event_id"]',
            },
        ]

        self.formatted_examples = []
        for ex in examples_data:
            self.formatted_examples.extend(
                [
                    {"role": "user", "content": ex["input"]},
                    {"role": "assistant", "content": ex["output"]},
                ]
            )
        self.system_message = {"role": "system", "content": system_prompt}

    def _load_cache(self) -> Dict[str, List[str]]:
        if self.cache_file and os.path.exists(self.cache_file):
            with open(self.cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        if self.cache_file and self.keywords_cache is not None:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.keywords_cache, f, indent=2, ensure_ascii=False)

    def _parse_llm_output(self, raw_output: str) -> List[str]:
        """
        Parses the raw LLM output to extract a Python list of strings.
        This adapts the `PythonListOutputParser` from CHESS.
        """
        try:
            # Find the string representation of the list
            match = re.search(r"\[.*\]", raw_output, re.DOTALL)
            if match:
                list_str = match.group(0)
                # Use ast.literal_eval for safe parsing
                parsed_list = ast.literal_eval(list_str)
                if isinstance(parsed_list, list):
                    return [str(item) for item in parsed_list]
        except (ValueError, SyntaxError):
            # Fallback if parsing fails
            return []
        return []

    def process(self, nlqs: List[str]) -> List[List[str]]:
        """
        Processes a batch of queries to extract keywords using VLLM.
        The final output for each query is a list containing the original query
        plus all extracted keywords.
        """
        if not nlqs:
            return []
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=self.vllm_args["trust_remote_code"],
            )
        if self.llm is None:
            self.llm = LLM(
                model=self.model_name_or_path,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                **self.vllm_args,
            )
        if self.sampling_params is None:
            self.sampling_params = SamplingParams(**self.sampling_args)
            
        final_results: List[Optional[List[str]]] = [None] * len(nlqs)
        prompts_to_generate, indices_to_generate, nlqs_to_generate = [], [], []

        for i, nlq in enumerate(tqdm(nlqs, desc="Preparing CHESS keyword prompts")):
            if self.keywords_cache is not None and nlq in self.keywords_cache:
                final_results[i] = self.keywords_cache[nlq]
                continue
            
            # The CHESS prompt combines Question and Hint. We will use the nlq as the main input.
            user_content = f'Question: "{nlq}"\nHint: ""'
            messages = (
                [self.system_message]
                + self.formatted_examples
                + [{"role": "user", "content": user_content}]
            )
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts_to_generate.append(prompt_text)
            indices_to_generate.append(i)
            nlqs_to_generate.append(nlq)

        if prompts_to_generate:
            vllm_outputs = self.llm.generate(prompts_to_generate, self.sampling_params)
            for i, output in enumerate(tqdm(vllm_outputs, desc="Extracting CHESS keywords")):
                original_nlq_idx = indices_to_generate[i]
                current_nlq = nlqs_to_generate[i]
                raw_output = output.outputs[0].text
                extracted_keywords = self._parse_llm_output(raw_output)

                # Combine original query with keywords for a richer search context
                final_processed_list = [current_nlq] + extracted_keywords
                # Deduplicate while preserving order
                deduplicated_list = list(dict.fromkeys(final_processed_list))
                
                final_results[original_nlq_idx] = deduplicated_list
                if self.keywords_cache is not None:
                    self.keywords_cache[current_nlq] = deduplicated_list

        if self.keywords_cache is not None and prompts_to_generate:
            self._save_cache()

        # Fallback for any queries that failed processing
        return [
            res if res is not None else [nlqs[i]]
            for i, res in enumerate(final_results)
        ]