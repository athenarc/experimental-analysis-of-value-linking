import os
import json
import argparse
import evaluate_bird
import evaluate_spider2
import evaluate_spider
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys # Add this
from pathlib import Path # Add this

def visualize(eval_name, acc_dict, ylabel, file_path):
    plt.figure(figsize=(10, 6))

    ckpt_ids = list(range(len(acc_dict)))
    values = list(acc_dict.values())

    if isinstance(values[0], list): # Spider has two metrics: EX acc and TS acc
        num_lines = len(values[0])
        labels = ["EX", "TS"]
        assert num_lines == len(labels)
        for i in range(num_lines):
            line_values = [v[i] for v in values]
            plt.plot(ckpt_ids, line_values, marker='o', linestyle='-', label=labels[i])
    else:
        plt.plot(ckpt_ids, values, marker='o', linestyle='-', label="EX")

    plt.title(eval_name)
    plt.xlabel('ckpt-id')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()

    plt.savefig(file_path)
    plt.close()

def save_evaluation_results(file_path, acc_dict):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(acc_dict, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_ckpt_dir", type = str, default = "./ckpts")
    parser.add_argument('--multiple_models', action='store_true', help='Evaluate multiple models from a folder.')
    parser.add_argument("--source", type = str, default = "bird")

    parser.add_argument("--visible_devices", type = str, default = "0,1")
    parser.add_argument("--input_file", type = str, help = "input file path (prompts)")
    parser.add_argument("--eval_name", type = str, help = "name of the evaluation set")
    parser.add_argument("--tensor_parallel_size", type = int, help = "the number of used GPUs", default = 1)
    parser.add_argument("--n", type = int, help = "sampling number", default = 16)

    parser.add_argument("--gold_file", type = str, help = "gold sql path")
    parser.add_argument("--db_path", type = str, help = "database path")
    parser.add_argument("--ts_db_path", type = str, default = "", help = "test suite database path (required by Spider)")
    parser.add_argument("--gold_result_dir", type = str, help = "gold sql execution results (required by Spider2.0)")
    parser.add_argument("--eval_standard", type = str, help = "evaluation standard (required by Spider2.0)")
    
    parser.add_argument("--use_vllm_server", action="store_true", help="Use vLLM server for inference in infer.py")
    parser.add_argument("--vllm_server_url", type=str, default="http://0.0.0.0:5001/v1", help="URL of the vLLM server for infer.py")
    parser.add_argument("--vllm_server_model_name", type=str, default="seeklhy/OmniSQL-32B", help="Model name used by the vLLM server for infer.py")

    opt = parser.parse_args()
    print(opt)

    assert opt.source in ["spider", "bird", "spider2.0"]

    # Get the directory of the current script (auto_evaluation.py)
    script_dir = Path(sys.argv[0]).parent.resolve()
    infer_script_path = script_dir / "infer.py"


    if opt.multiple_models:
        ckpt_ids = os.listdir(opt.output_ckpt_dir)
        ckpt_ids = sorted(ckpt_ids, key=lambda x: int(x.split("-")[1]))
        print(ckpt_ids)
    else:
        ckpt_ids = [""]

    greedy_search_acc_dict = dict()
    pass_at_k_acc_dict = dict()
    major_voting_acc_dict = dict()

    os.makedirs(os.path.join("results", opt.eval_name), exist_ok=True)
    os.makedirs(os.path.join("evaluation_results", opt.eval_name), exist_ok=True)

    vllm_server_args = ""
    if opt.use_vllm_server:
        vllm_server_args = f" --use_vllm_server --vllm_server_url {opt.vllm_server_url} --vllm_server_model_name {opt.vllm_server_model_name}"


    for ckpt_id in tqdm(ckpt_ids):
        print("Evaluating ckpt:", ckpt_id)
        
        model_path_or_name = os.path.join(opt.output_ckpt_dir, ckpt_id) if ckpt_id else opt.output_ckpt_dir


        if ckpt_id not in greedy_search_acc_dict.keys():
            # greedy decoding
            gs_pred_file = f"results/{opt.eval_name}/greedy_search_{ckpt_id if ckpt_id else model_path_or_name.split('/')[-1]}.json"
            greedy_search_cmd = f"CUDA_VISIBLE_DEVICES={opt.visible_devices} python3 {infer_script_path} \
                --pretrained_model_name_or_path {model_path_or_name} \
                --input_file {opt.input_file} \
                --output_file {gs_pred_file} \
                --tensor_parallel_size {opt.tensor_parallel_size} \
                --n 1 \
                --temperature 0.0{vllm_server_args}"
            print(f"Executing: {greedy_search_cmd}")
            os.system(greedy_search_cmd)

            # evaluate greedy search
            if opt.source == "spider2.0":
                # warm up
                evaluate_spider2.evaluate("greedy_search", opt.gold_result_dir, opt.eval_standard, 
                                        opt.gold_file, gs_pred_file, opt.db_path, True)
                # record evaluation results
                gs_acc, _ = evaluate_spider2.evaluate("greedy_search", opt.gold_result_dir, opt.eval_standard, 
                                        opt.gold_file, gs_pred_file, opt.db_path, True)
            elif opt.source == "bird":
                # warm up
                evaluate_bird.run_eval(opt.gold_file, gs_pred_file, opt.db_path, "greedy_search", True)
                # record evaluation results
                gs_acc, _ = evaluate_bird.run_eval(opt.gold_file, gs_pred_file, opt.db_path, "greedy_search", True)
            elif opt.source == "spider": # for "spider"
                # warm up
                evaluate_spider.run_spider_eval(opt.gold_file, gs_pred_file, opt.db_path, 
                                            opt.ts_db_path, "greedy_search", True)
                # record evaluation results
                ex_score, ts_score = evaluate_spider.run_spider_eval(opt.gold_file, gs_pred_file, opt.db_path, 
                                            opt.ts_db_path, "greedy_search", True)
                if ts_score is None:
                    gs_acc = ex_score
                else:
                    gs_acc = [ex_score, ts_score]
            
            greedy_search_acc_dict[ckpt_id if ckpt_id else model_path_or_name.split('/')[-1]] = gs_acc
            print(greedy_search_acc_dict)
            visualize(opt.eval_name, greedy_search_acc_dict, "greedy_search",
                os.path.join("evaluation_results", opt.eval_name, "greedy_search.png"))
            save_evaluation_results(os.path.join("evaluation_results", opt.eval_name, "greedy_search.json"), greedy_search_acc_dict)
        else:
            print(f"skip {ckpt_id if ckpt_id else model_path_or_name.split('/')[-1]} greedy search")

        if ckpt_id not in major_voting_acc_dict.keys():
            # sampling
            sampling_pred_file = f"results/{opt.eval_name}/sampling_{ckpt_id if ckpt_id else model_path_or_name.split('/')[-1]}.json"
            sampling_cmd = f"CUDA_VISIBLE_DEVICES={opt.visible_devices} python3 {infer_script_path} \
                --pretrained_model_name_or_path {model_path_or_name} \
                --input_file {opt.input_file} \
                --output_file {sampling_pred_file} \
                --tensor_parallel_size {opt.tensor_parallel_size} \
                --n {opt.n} \
                --temperature 0.8{vllm_server_args}"
            print(f"Executing: {sampling_cmd}")
            os.system(sampling_cmd)

            # evaluate pass@k (we do not evaluate pass@k for spider and its variants)
            if opt.source in ["bird", "spider2.0"]:
                if opt.source == "spider2.0":
                    # warm up
                    evaluate_spider2.evaluate("pass@k", opt.gold_result_dir, opt.eval_standard, 
                        opt.gold_file, sampling_pred_file, opt.db_path, True)
                    # record evaluation results
                    pass_at_k_acc, _ = evaluate_spider2.evaluate("pass@k", opt.gold_result_dir, opt.eval_standard, 
                                            opt.gold_file, sampling_pred_file, opt.db_path, True)
                elif opt.source == "bird":
                    # warm up
                    evaluate_bird.run_eval(opt.gold_file, sampling_pred_file, opt.db_path, "pass@k", True)
                    # record evaluation results
                    pass_at_k_acc, _ = evaluate_bird.run_eval(opt.gold_file, sampling_pred_file, opt.db_path, "pass@k", True)
                pass_at_k_acc_dict[ckpt_id if ckpt_id else model_path_or_name.split('/')[-1]] = pass_at_k_acc
                print(pass_at_k_acc_dict)
                visualize(opt.eval_name, pass_at_k_acc_dict, "pass_at_k",
                    os.path.join("evaluation_results", opt.eval_name, "pass_at_k.png"))
                save_evaluation_results(os.path.join("evaluation_results", opt.eval_name, "pass_at_k.json"), pass_at_k_acc_dict)
            
            # evaluate major voting
            if opt.source == "spider2.0":
                # warm up
                evaluate_spider2.evaluate("major_voting", opt.gold_result_dir, opt.eval_standard, 
                                            opt.gold_file, sampling_pred_file, opt.db_path, True)
                # record evaluation results
                major_voting_acc, _ = evaluate_spider2.evaluate("major_voting", opt.gold_result_dir, opt.eval_standard, 
                                            opt.gold_file, sampling_pred_file, opt.db_path, True)
            elif opt.source == "bird":
                # warm up
                evaluate_bird.run_eval(opt.gold_file, sampling_pred_file, opt.db_path, "major_voting", True)
                # record evaluation results
                major_voting_acc, _ = evaluate_bird.run_eval(opt.gold_file, sampling_pred_file, opt.db_path, "major_voting", True)
            else: # spider
                # warm up
                evaluate_spider.run_spider_eval(opt.gold_file, sampling_pred_file, opt.db_path, 
                                            opt.ts_db_path, "major_voting", True)
                # record evaluation results
                ex_score, ts_score = evaluate_spider.run_spider_eval(opt.gold_file, sampling_pred_file, opt.db_path, 
                                            opt.ts_db_path, "major_voting", True)
                if ts_score is None:
                    major_voting_acc = ex_score
                else:
                    major_voting_acc = [ex_score, ts_score]
            
            major_voting_acc_dict[ckpt_id if ckpt_id else model_path_or_name.split('/')[-1]] = major_voting_acc
            print(major_voting_acc_dict)
            visualize(opt.eval_name, major_voting_acc_dict, "major_voting",
                os.path.join("evaluation_results", opt.eval_name, "major_voting.png"))
            save_evaluation_results(os.path.join("evaluation_results", opt.eval_name, "major_voting.json"), major_voting_acc_dict)
        else:
            print(f"skip {ckpt_id if ckpt_id else model_path_or_name.split('/')[-1]} pass at k and major voting")