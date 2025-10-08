import inspect
from threading import Lock
from typing import Any, Dict, Tuple
from sentence_transformers import SentenceTransformer 
class PipelineManager:
    _instance = None
    _lock = Lock()

    def __new__(cls, pipeline_setup: Dict[str, Any] = None, args: Any = None):
        # ... (new method is unchanged)
        if pipeline_setup is not None:
            with cls._lock:
                cls._instance = super(PipelineManager, cls).__new__(cls)
                cls._instance.pipeline_setup = pipeline_setup
                cls._instance.args = args
                # +++ ADD ATTRIBUTES FOR SHARED MODELS +++
                cls._instance.bert_model = None
                # +++++++++++++++++++++++++++++++++++++++++
                cls._instance._init(pipeline_setup)
        elif cls._instance is None:
            raise ValueError("pipeline_setup dictionary must be provided for initialization")
        return cls._instance

    def _init(self, pipeline_setup: Dict[str, Any]):
        """
        Custom initialization logic using the pipeline_setup dictionary.

        Args:
            pipeline_setup (Dict[str, Any]): The setup dictionary for the pipeline.
        """
        self.generate_db_schema = pipeline_setup.get("generate_db_schema", {})
        self.extract_col_value = pipeline_setup.get("extract_col_value", {})
        self.extract_query_noun = pipeline_setup.get("extract_query_noun", {})
        self.column_retrieve_and_other_info = pipeline_setup.get("column_retrieve_and_other_info", {})
        self.candidate_generate = pipeline_setup.get("candidate_generate", {})
        self.align_correct = pipeline_setup.get("align_correct", {})
        self.vote = pipeline_setup.get("vote", {})
    def initialize_shared_models(self):
        """
        Initializes models that are shared across multiple pipeline nodes.
        This should be called once at the beginning of a run.
        """
        # Find a node that defines the bert_model config (any of them will do)
        bert_model_name = None
        device = 'cuda:0' # Default device
        for node_config in self.pipeline_setup.values():
            if "bert_model" in node_config:
                bert_model_name = node_config["bert_model"]
                device = node_config.get("device", device)
                break
        
        if bert_model_name and self.bert_model is None:
            print(f"Initializing shared SentenceTransformer model: {bert_model_name}")
            self.bert_model = SentenceTransformer(bert_model_name, device="cuda:0")
            print("Shared model loaded successfully.")

    def get_bert_model(self) -> SentenceTransformer:
        """Returns the shared SentenceTransformer instance."""
        if self.bert_model is None:
            raise Exception("Shared BERT model has not been initialized. Call initialize_shared_models() first.")
        return self.bert_model
    # +++ ADD A GETTER FOR RUNTIME ARGS +++
    def get_runtime_args(self) -> Any:
        """
        Returns the runtime arguments stored during initialization.
        """
        return self.args
    
    def get_model_para(self, node_name: str = None, **kwargs: Any) -> Tuple[Dict[str, Any], str]:
        """
        Retrieves the configuration for a specific node.

        Args:
            node_name (str, optional): The name of the node to get config for. 
                                       If None, it attempts to infer from the call stack.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[Dict[str, Any], str]: The node's setup dictionary and its name.
        """
        if node_name is None:
            # Original logic for when called from within a node
            frame = inspect.currentframe()
            caller_frame = frame.f_back
            node_name = caller_frame.f_code.co_name
        
        node_setup = self.pipeline_setup.get(node_name, {})
                
        return node_setup, node_name
    
