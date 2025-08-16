import argparse
import json
import re
from transformers import AutoTokenizer
import ray
from packaging.version import Version
from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig

# This function is from the original infer.py
def parse_response(response):
    pattern = r"```sql\s*(.*?)\s*```"
    sql_blocks = re.findall(pattern, response, re.DOTALL)
    if sql_blocks:
        return sql_blocks[-1].strip()
    else:
        return ""

def run_inference_with_ray(args):
    """
    Runs batch inference using Ray Data and vLLM without a server.
    """
    assert Version(ray.__version__) >= Version("2.44.1"), "Ray version must be at least 2.44.1"

    print("Starting Ray Data inference for arguments:", args)

    # 1. Load and prepare the input data
    input_dataset = json.load(open(args.input_file))
    # Convert to Ray Dataset
    ds = ray.data.from_items(input_dataset)

    # 2. Prepare tokenizer and stop strings (needed for the preprocess step)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, trust_remote_code=True)
    
    # This logic is copied directly from the original infer.py
    if "Qwen2.5-" in args.pretrained_model_name_or_path:
        stop_token_ids = [151645]
    elif "OmniSQL-" in args.pretrained_model_name_or_path:
        stop_token_ids = [151645]
    # ... (add all other elif conditions from infer.py for different models)
    else:
        print("Use Qwen2.5's stop tokens by default.")
        stop_token_ids = [151645]
    
    stop_strings = [tokenizer.decode(token_id) for token_id in stop_token_ids]
    max_output_len = 2048 # From original script

    # 3. Configure the vLLM engine
    config = vLLMEngineProcessorConfig(
        model_source=args.pretrained_model_name_or_path,
        engine_kwargs={
            "tensor_parallel_size": args.tensor_parallel_size,
            "max_model_len": 8192,  # Hardcoded based on our analysis
            "gpu_memory_utilization": 0.92,
        },
        concurrency=1, # Number of parallel vLLM replicas. 1 is usually enough for a single machine.
        batch_size=64, # Ray Data will batch inputs before sending to vLLM.
    )

    # 4. Build the LLM Processor
    vllm_processor = build_llm_processor(
        config,
        # Preprocess each row to create the prompt and sampling params
        preprocess=lambda row: dict(
            messages=[{"role": "user", "content": row["input_seq"]}],
            sampling_params=dict(
                temperature=args.temperature,
                n=args.n,
                max_tokens=max_output_len,
                stop=stop_strings,
            ),
        ),
        # Postprocess the output to parse the SQL
        postprocess=lambda row: dict(
            pred_sql=parse_response(row["generated_text"]),
            response=row["generated_text"],
            **row  # Keep original columns
        ),
    )

    # 5. Run the inference
    print("Applying vLLM processor to the dataset...")
    ds_with_results = vllm_processor(ds)
    
    # 6. Collect and format results
    # The output needs to be in the same format as the original script for the evaluation to work
    print("Inference complete. Collecting and formatting results...")
    
    # Group results by the original data index
    results_by_index = {}
    for i, row in enumerate(ds_with_results.iter_rows()):
        # Since n>1 creates multiple output rows for one input row, we need to find the original
        original_index = i // args.n
        if original_index not in results_by_index:
            # Reconstruct the original data object
            original_data = {k: v for k, v in row.items() if k not in ['generated_text', 'pred_sql', 'response']}
            original_data["responses"] = []
            original_data["pred_sqls"] = []
            results_by_index[original_index] = original_data
        
        results_by_index[original_index]["responses"].append(row["response"])
        results_by_index[original_index]["pred_sqls"].append(row["pred_sql"])

    final_results = [results_by_index[i] for i in sorted(results_by_index.keys())]

    # 7. Save the final output file
    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(final_results, indent=2, ensure_ascii=False))
    
    print(f"Results saved to {args.output_file}")