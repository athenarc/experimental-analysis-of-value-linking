from typing import List, Dict, Any
from threading import Lock
import logging
from vllm import LLM, SamplingParams


class VLLMManager:
    """
    A singleton manager to handle a single instance of a vLLM model
    for direct, in-process batch generation.
    """
    _instance = None
    _lock = Lock()
    _model: 'LLM' = None
    _initialized = False

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(VLLMManager, cls).__new__(cls)
            return cls._instance

    @classmethod
    def initialize_model(cls, model_path: str, tensor_parallel_size: int = 2):
        """
        Loads the vLLM model into memory. Should be called once per process.
        """
        if LLM is None:
            raise ImportError("vllm is not installed. Please run 'pip install vllm'.")
            
        with cls._lock:
            if not cls._initialized:
                if not model_path:
                    raise ValueError("A model path must be provided to initialize vLLM batch mode.")
                
                logging.info(f"Initializing vLLM model: {model_path}...")
                cls._model = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size,gpu_memory_utilization=0.8,download_dir="/data/hdd1/vllm_models/",max_model_len=30000)
                cls._initialized = True
                logging.info("vLLM model initialized successfully.")

    @classmethod
    def is_initialized(cls) -> bool:
        """Check if the model has been loaded."""
        return cls._initialized

    @classmethod
    def generate(cls, prompts: List[str], sampling_params_dict: Dict[str, Any]) -> List[str]:
        """
        Generates completions for a batch of prompts.
        """
        if not cls.is_initialized() or cls._model is None:
            raise RuntimeError("VLLMManager has not been initialized. Call initialize_model() first.")

        # Convert the dictionary to a vllm SamplingParams object
        sampling_params = SamplingParams(**sampling_params_dict)
        
        logging.info(f"Submitting batch of {len(prompts)} prompts to vLLM...")
        outputs = cls._model.generate(prompts, sampling_params)
        
        # Extract the text from the output objects
        generated_texts = [output.outputs[0].text for output in outputs]
        logging.info("vLLM batch generation complete.")
        
        return generated_texts