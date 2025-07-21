from typing import Any, Dict, List

from langchain_core.exceptions import OutputParserException
from langchain.output_parsers import OutputFixingParser
from concurrent.futures import ThreadPoolExecutor
from llm.engine_configs import ENGINE_CONFIGS
from runner.logger import Logger
from threading_utils import ordered_concurrent_function_calls
import concurrent
def get_llm_chain(engine_name: str, temperature: float = 0, base_uri: str = None) -> Any:
    """
    Returns the appropriate LLM chain based on the provided engine name and temperature.

    Args:
        engine (str): The name of the engine.
        temperature (float): The temperature for the LLM.
        base_uri (str, optional): The base URI for the engine. Defaults to None.

    Returns:
        Any: The LLM chain instance.

    Raises:
        ValueError: If the engine is not supported.
    """
    if engine_name not in ENGINE_CONFIGS:
        raise ValueError(f"Engine {engine_name} not supported")
    
    config = ENGINE_CONFIGS[engine_name]
    constructor = config["constructor"]
    params = config["params"]
    if temperature:
        params["temperature"] = temperature
    
    # Adjust base_uri if provided
    if base_uri and "openai_api_base" in params:
        params["openai_api_base"] = f"{base_uri}/v1"
    
    model = constructor(**params)
    if "preprocess" in config:
        llm_chain = config["preprocess"] | model
    else:
        llm_chain = model
    return llm_chain

def call_llm_chain(prompt: Any, engine: Any, parser: Any, request_kwargs: Dict[str, Any], step: int, max_attempts: int = 12, backoff_base: int = 2, jitter_max: int = 60) -> Any:
    """
    Calls the LLM chain with exponential backoff and jitter on failure.

    Args:
        prompt (Any): The prompt to be passed to the chain.
        engine (Any): The engine to be used in the chain.
        parser (Any): The parser to parse the output.
        request_kwargs (Dict[str, Any]): The request arguments.
        step (int): The current step in the process.
        max_attempts (int, optional): The maximum number of attempts. Defaults to 12.
        backoff_base (int, optional): The base for exponential backoff. Defaults to 2.
        jitter_max (int, optional): The maximum jitter in seconds. Defaults to 60.

    Returns:
        Any: The output from the chain.

    Raises:
        Exception: If all attempts fail.
    """
    logger = Logger()
    for attempt in range(max_attempts):
        try:
            # chain = prompt | engine | parser
            chain = prompt | engine
            prompt_text = prompt.invoke(request_kwargs).messages[0].content
            output = chain.invoke(request_kwargs)
            if isinstance(output, str):
                if output.strip() == "":
                    engine = get_llm_chain("gemini-1.5-flash")
                    raise OutputParserException("Empty output")
            else:
                if output.content.strip() == "":    
                    engine = get_llm_chain("gemini-1.5-flash")
                    raise OutputParserException("Empty output")
            output = parser.invoke(output)
            logger.log_conversation(
                [
                    {
                        "text": prompt_text,
                        "from": "Human",
                        "step": step
                    },
                    {
                        "text": output,
                        "from": "AI",
                        "step": step
                    }
                ]
            )
            return output
        except OutputParserException as e:
            logger.log(f"OutputParserException: {e}", "warning")
            new_parser = OutputFixingParser.from_llm(parser=parser, llm=engine)
            chain = prompt | engine | new_parser
            if attempt == max_attempts - 1:
                logger.log(f"call_chain: {e}", "error")
                raise e
        except Exception as e:
            # if attempt < max_attempts - 1:
            #     logger.log(f"Failed to invoke the chain {attempt + 1} times.\n{type(e)}\n{e}", "warning")
            #     sleep_time = (backoff_base ** attempt) + random.uniform(0, jitter_max)
            #     time.sleep(sleep_time)
            # else:
            logger.log(f"Failed to invoke the chain {attempt + 1} times.\n{type(e)} <{e}>\n", "error")
            raise e

def async_llm_chain_call(
    prompt: Any, 
    engine: Any, 
    parser: Any, 
    request_list: List[Dict[str, Any]], 
    step: int, 
    sampling_count: int = 1
) -> List[List[Any]]:
    logger = Logger() # Assuming Logger can be instantiated without args here for general logging
    call_list = []
    engine_id = 0

    if not request_list: # ADD THIS CHECK
        logger.log(f"async_llm_chain_call for step {step}: received empty request_list. Returning empty results.", "warning")
        return [] # Return empty list if there are no requests to process

    for request_id, request_kwargs in enumerate(request_list):
        for _ in range(sampling_count):
            call_list.append({
                'function': call_llm_chain,
                'kwargs': {
                    'prompt': prompt,
                    'engine': engine[engine_id % len(engine)] if isinstance(engine,list) else engine,
                    'parser': parser,
                    'request_kwargs': request_kwargs,
                    'step': step
                }
            })
            engine_id += 1

    # num_workers for ThreadPoolExecutor should be at least 1 if call_list is not empty
    num_workers_for_pool = len(call_list)
    if num_workers_for_pool == 0: # Should be caught by the earlier `if not request_list:`
        logger.log(f"async_llm_chain_call for step {step}: call_list became empty unexpectedly. Returning empty.", "warning")
        return [[] for _ in range(len(request_list))] # Match expected output structure for empty

    # Execute the functions concurrently
    # The original ordered_concurrent_function_calls might have its own ThreadPoolExecutor.
    # If using that directly:
    # results = ordered_concurrent_function_calls(call_list)
    
    # If implementing ThreadPoolExecutor here as implied by max_workers error:
    results_placeholder = [None] * len(call_list)
    try:
        with ThreadPoolExecutor(max_workers=num_workers_for_pool) as executor:
            future_to_id = {executor.submit(call['function'], **call['kwargs']): i for i, call in enumerate(call_list)}
            for future in concurrent.futures.as_completed(future_to_id):
                call_id = future_to_id[future]
                try:
                    results_placeholder[call_id] = future.result()
                except Exception as exc:
                    logger.log(f"async_llm_chain_call: Generated an exception for call_id {call_id}, step {step}: {exc}", "error", exc_info=True)
                    results_placeholder[call_id] = None # Store None or a specific error marker
    except Exception as pool_exc: # Catch errors from ThreadPoolExecutor itself
        logger.log(f"async_llm_chain_call: ThreadPoolExecutor error for step {step}: {pool_exc}", "error", exc_info=True)
        # Fill results with None if the pool failed catastrophically
        results_placeholder = [None] * len(call_list)


    # Group results by sampling_count
    grouped_results = []
    if request_list: # Ensure request_list is not empty before trying to group
        for i in range(len(request_list)):
            start_index = i * sampling_count
            end_index = (i + 1) * sampling_count
            # Ensure slice indices are valid and handle cases where results_placeholder might not be fully populated due to errors
            group = [results_placeholder[j] for j in range(start_index, end_index) if j < len(results_placeholder)]
            grouped_results.append(group)
    
    return grouped_results


def call_engine(message: str, engine: Any, max_attempts: int = 12, backoff_base: int = 2, jitter_max: int = 60) -> Any:
    """
    Calls the LLM chain with exponential backoff and jitter on failure.

    Args:
        message (str): The message to be passed to the chain.
        engine (Any): The engine to be used in the chain.
        max_attempts (int, optional): The maximum number of attempts. Defaults to 12.
        backoff_base (int, optional): The base for exponential backoff. Defaults to 2.
        jitter_max (int, optional): The maximum jitter in seconds. Defaults to 60.

    Returns:
        Any: The output from the chain.

    Raises:
        Exception: If all attempts fail.
    """
    logger = Logger()
    for attempt in range(max_attempts):
        try:
            output = engine.invoke(message)
            return output.content
        except Exception as e:
            # if attempt < max_attempts - 1:
            #     logger.log(f"Failed to invoke the chain {attempt + 1} times.\n{type(e)}\n{e}", "warning")
            #     sleep_time = (backoff_base ** attempt) + random.uniform(0, jitter_max)
            #     time.sleep(sleep_time)
            # else:
            logger.log(f"Failed to invoke the chain {attempt + 1} times.\n{type(e)} <{e}>\n", "error")
            raise e