import json
import os
from typing import Dict, List, Tuple

import pandas as pd
import wandb
from darelabdb.nlp_retrieval.core.models import RetrievalResult, SearchableItem
from darelabdb.nlp_retrieval.evaluation.eval_models import EvaluationSummary
from darelabdb.nlp_retrieval.evaluation.evaluator import RetrievalEvaluator
from darelabdb.nlp_retrieval.searcher import Searcher
# --- CHESS Imports ---
from development.experimental_analysis_of_value_linking.retrievers.CHESS.chess_db_loader import ChessDBLoader
from development.experimental_analysis_of_value_linking.retrievers.CHESS.chess_query_processor import ChessQueryProcessor
from development.experimental_analysis_of_value_linking.retrievers.CHESS.chess_reranker import ChessSimilarityReranker
from development.experimental_analysis_of_value_linking.retrievers.CHESS.chess_retriever import ChessMinHashLshRetriever

# --- OmniSQL Imports ---
from development.experimental_analysis_of_value_linking.retrievers.OmniSQL.omnisql_loader import OmniSQLLoader
from development.experimental_analysis_of_value_linking.retrievers.OmniSQL.omnisql_query_processor import OmniSQLQueryProcessor
from development.experimental_analysis_of_value_linking.retrievers.OmniSQL.omnisql_reranker import OmniSQLReranker
from development.experimental_analysis_of_value_linking.retrievers.OmniSQL.omnisql_retriever import OmniSQLRetriever

# --- OpenSearch (Dense) Imports ---
from development.experimental_analysis_of_value_linking.retrievers.OpenSearch.opensearch_loader import OpenSearchValueLoader
from development.experimental_analysis_of_value_linking.retrievers.OpenSearch.opensearch_query_processor import OpenSearchKeywordProcessor
from development.experimental_analysis_of_value_linking.retrievers.OpenSearch.opensearch_reranker import OpenSearchPassthroughReranker
from development.experimental_analysis_of_value_linking.retrievers.OpenSearch.opensearch_retriever import OpenSearchDenseValueRetriever

# --- Configuration ---
BASE_PATH = "development/experimental_analysis_of_value_linking/assets"
DATABASES_ROOT = os.path.join(BASE_PATH, "retrievers", "databases")
INDEXES_ROOT = os.path.join(BASE_PATH, "retrievers", "indexes")
BENCHMARK_FILE = os.path.join(BASE_PATH, "all_benchmarks_human/all_dump_good.json")

# Weights & Biases Configuration
WANDB_ENTITY = "darelab"
WANDB_PROJECT = "value_linking"

# Evaluation Configuration
K_VALUES = [1, 5, 10, 20, 50]
RETRIEVAL_DEPTH = 100 # Retrieve a fixed large number for full evaluation

# Model Configuration
LLM_MODEL_PATH = "mlx-community/gemma-3n-E4B-it-4bit"
EMBEDDING_MODEL_PATH = "BAAI/bge-m3"

def load_and_group_benchmark_data(file_path: str) -> Dict[str, Tuple[List[str], List[List[RetrievalResult]]]]:
    """
    Loads benchmark data from the specified JSON file and groups it by db_id.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        all_tasks = json.load(f)

    grouped_data = {}
    for task in all_tasks:
        db_id = task.get("db_id")
        query = task.get("new_question_correct_value")
        gold_values = task.get("values")

        if not db_id or not query or not gold_values:
            continue

        if db_id not in grouped_data:
            grouped_data[db_id] = ([], [])

        # Append query
        grouped_data[db_id][0].append(query)

        # Create and append gold standard results
        current_gold_list = []
        for gold_meta in gold_values:
            # We only need table, column, and value for matching
            relevant_meta = {
                "table": gold_meta.get("table"),
                "column": gold_meta.get("column"),
                "value": gold_meta.get("value"),
            }
            gold_item = SearchableItem(item_id="gold", content="gold", metadata=relevant_meta)
            current_gold_list.append(RetrievalResult(item=gold_item, score=1.0))
        
        grouped_data[db_id][1].append(current_gold_list)

    print(f"Loaded benchmark data for {len(grouped_data)} databases from {os.path.basename(file_path)}.")
    return grouped_data

def get_system_configs():
    """Returns a dictionary of all system configurations."""
    
    # This is a factory function that will be called with a db_id
    def get_chess_config(db_id):
        db_dir_path = os.path.join(DATABASES_ROOT, db_id)
        return {
            "loader": ChessDBLoader(db_directory_path=db_dir_path),
            "searcher": Searcher(
                query_processor=ChessQueryProcessor(model_name_or_path=LLM_MODEL_PATH, cache_folder="./cache/keywords_chess", tensor_parallel_size=2, gpu_memory_utilization=0.35),
                retrievers=[ChessMinHashLshRetriever()],
                reranker=ChessSimilarityReranker(model_name=EMBEDDING_MODEL_PATH)
            ),
            "index_path": os.path.join(INDEXES_ROOT, "chess", db_id)
        }

    def get_omnisql_config(db_id):
        db_file_path = os.path.join(DATABASES_ROOT, db_id, f"{db_id}.sqlite")
        return {
            "loader": OmniSQLLoader(db_file_path=db_file_path),
            "searcher": Searcher(
                query_processor=OmniSQLQueryProcessor(n=8),
                retrievers=[OmniSQLRetriever()],
                reranker=OmniSQLReranker(score_threshold=0.8)
            ),
            "index_path": os.path.join(INDEXES_ROOT, "omnisql", db_id)
        }

    def get_opensearch_config(db_id):
        db_file_path = os.path.join(DATABASES_ROOT, db_id, f"{db_id}.sqlite")
        return {
            "loader": OpenSearchValueLoader(db_path=db_file_path, db_id=db_id),
            "searcher": Searcher(
                query_processor=OpenSearchKeywordProcessor(model_name_or_path=LLM_MODEL_PATH, cache_folder="./cache/keywords_open_search", tensor_parallel_size=2, gpu_memory_utilization=0.35),
                retrievers=[OpenSearchDenseValueRetriever(model_name_or_path=EMBEDDING_MODEL_PATH)],
                reranker=OpenSearchPassthroughReranker()
            ),
            "index_path": os.path.join(INDEXES_ROOT, "opensearch", db_id)
        }

    return {
        "CHESS": get_chess_config,
        "OmniSQL": get_omnisql_config,
        "OpenSearch": get_opensearch_config,
    }


def aggregate_summaries(summaries: List[EvaluationSummary]) -> Dict:
    """Aggregates multiple EvaluationSummary objects into a single set of micro-averaged metrics."""
    if not summaries:
        return {}

    total_tp = sum(q_metric.true_positives for s in summaries for q_metric in s.per_query_details)
    total_fp = sum(q_metric.false_positives for s in summaries for q_metric in s.per_query_details)
    total_fn = sum(q_metric.false_negatives for s in summaries for q_metric in s.per_query_details)
    
    num_queries = sum(s.num_queries for s in summaries)
    perfect_recall_sum = sum(s.perfect_recall_rate * s.num_queries for s in summaries)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    prr = perfect_recall_sum / num_queries if num_queries > 0 else 0.0

    return {
        "Overall Precision": precision,
        "Overall Recall": recall,
        "Overall F1 Score": f1,
        "Overall Perfect Recall Rate": prr,
        "Total Queries": num_queries
    }


def main():
    benchmark_data = load_and_group_benchmark_data(BENCHMARK_FILE)
    system_configs = get_system_configs()
    evaluator = RetrievalEvaluator()

    # --- Outer loop: Iterate over each system ---
    for system_name, config_factory in system_configs.items():
        print(f"\n{'='*30}\nRUNNING BENCHMARK FOR SYSTEM: {system_name}\n{'='*30}")

        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=f"{system_name}-Benchmark-Report",
            reinit=True,
        )

        all_db_summaries = []
        per_db_table_data = []

        # --- Inner loop: Iterate over each database for the current system ---
        for db_id, (queries, gold_standard) in benchmark_data.items():
            print(f"\n--- Evaluating on DB: {db_id} for System: {system_name} ---")

            # Get the specific configuration for this system and db_id
            config = config_factory(db_id)
            searcher = config["searcher"]
            index_path = config["index_path"]
            
            if not os.path.exists(index_path) or not os.listdir(index_path):
                print(f"Index not found for {db_id} at {index_path}. Skipping.")
                # You might want to run indexing here if it's missing
                # searcher.index(loader=config["loader"], output_path=index_path)
                continue

            # Run search to get predicted results
            predicted_results = searcher.search(
                nlqs=queries, output_path=index_path, k=RETRIEVAL_DEPTH
            )

            # Evaluate performance for this database on the full set of results
            summary = evaluator.evaluate(predicted_results, gold_standard)
            all_db_summaries.append(summary)

            # Log per-DB results
            db_metrics = {
                "Database ID": db_id,
                "Num Queries": summary.num_queries,
                "Precision": summary.overall_precision,
                "Recall": summary.overall_recall,
                "F1 Score": summary.overall_f1_score,
                "Perfect Recall Rate": summary.perfect_recall_rate,
            }
            per_db_table_data.append(list(db_metrics.values()))
            
            print(f"  Recall on {db_id}: {summary.overall_recall:.4f}")

        # --- After all databases are processed for the system ---
        if per_db_table_data:
            # Log the detailed per-database performance table
            columns = ["Database ID", "Num Queries", "Precision", "Recall", "F1 Score", "Perfect Recall Rate"]
            per_db_table = wandb.Table(columns=columns, data=per_db_table_data)
            wandb.log({f"performance_by_database": per_db_table})

            # Calculate and log the aggregated summary
            aggregated_metrics = aggregate_summaries(all_db_summaries)
            wandb.summary.update(aggregated_metrics)
            
            print("\n--- Aggregated Summary for System: {system_name} ---")
            summary_df = pd.DataFrame([aggregated_metrics])
            print(summary_df.to_markdown(index=False))

        wandb.finish()

    print("\n\nBenchmarking finished for all systems.")

if __name__ == "__main__":
    main()