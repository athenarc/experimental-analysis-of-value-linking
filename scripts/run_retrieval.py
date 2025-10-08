import json
import os
import time  # Added for timing
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd
import wandb
from darelabdb.nlp_retrieval.core.models import RetrievalResult, SearchableItem
from darelabdb.nlp_retrieval.evaluation.eval_models import (
    EvaluationSummary,
    PerQueryMetrics,
)
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
from development.experimental_analysis_of_value_linking.retrievers.OmniSQL.codes_reranker import CodesReranker

# --- OpenSearch (Dense) Imports ---
from development.experimental_analysis_of_value_linking.retrievers.OpenSearch.opensearch_loader import OpenSearchValueLoader
from development.experimental_analysis_of_value_linking.retrievers.OpenSearch.opensearch_query_processor import OpenSearchQueryProcessor
from development.experimental_analysis_of_value_linking.retrievers.OpenSearch.opensearch_reranker import OpenSearchRuleBasedReranker
from development.experimental_analysis_of_value_linking.retrievers.OpenSearch.opensearch_retriever import OpenSearchDenseValueRetriever

# --- ValueNet Imports ---
from development.experimental_analysis_of_value_linking.retrievers.ValueNet.valuenet_loader import ValueNetLoader
from development.experimental_analysis_of_value_linking.retrievers.ValueNet.valuenet_retriever import ValueNetRetriever
from development.experimental_analysis_of_value_linking.retrievers.ValueNet.valuenet_query_processor import ValueNetQueryProcessor
from development.experimental_analysis_of_value_linking.retrievers.ValueNet.valuenet_reranker import ValueNetReranker

from development.experimental_analysis_of_value_linking.retrievers.BRIDGE.bridge_retriever import BridgeRetriever
from development.experimental_analysis_of_value_linking.retrievers.BRIDGE.bridge_query_processor import BridgeQueryProcessor
from development.experimental_analysis_of_value_linking.retrievers.BRIDGE.bridge_reranker import BridgeReranker

# --- Configuration ---
BASE_PATH = "development/experimental_analysis_of_value_linking/assets"
DATABASES_ROOT = os.path.join(BASE_PATH, "retrievers", "databases")
INDEXES_ROOT = os.path.join(BASE_PATH, "retrievers", "indexes")
BENCHMARK_FILE = os.path.join(BASE_PATH, "all_benchmarks_human/all_dump_good.json")
MISSED_ITEMS_FILE = os.path.join(BASE_PATH, "temp/missed_items.json")

# Weights & Biases Configuration
WANDB_ENTITY = "darelab"
WANDB_PROJECT = "value_linking"
GROUP_NAME="test_run"

# Evaluation Configuration
K_VALUES = [1, 2, 3, 5, 10, 20]
RETRIEVAL_DEPTH = 10000 # Retrieve a fixed large number for full evaluation

# Model Configuration
LLM_MODEL_PATH = "gaunernst/gemma-3-12b-it-int4-awq"
EMBEDDING_MODEL_PATH = "BAAI/bge-m3"

def load_and_group_benchmark_data(
    file_path: str,
) -> Dict[str, Tuple[List[str], List[List[RetrievalResult]], List[str], List[str]]]:
    """
    Loads benchmark data, groups it by db_id, and extracts categories and sources.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        all_tasks = json.load(f)

    grouped_data = {}
    for task in all_tasks:
        db_id = task.get("db_id")
        query = task.get("new_question_correct_value")
        # new_question_correct_value
        gold_values = task.get("values")
        changes_info = task.get("changes_information")
        source = task.get("source", "other")

        if not db_id or not query or not gold_values:
            continue

        if db_id not in grouped_data:
            grouped_data[db_id] = (
                [],
                [],
                [],
                [],
            )  # queries, gold, categories, sources

        # Append query
        grouped_data[db_id][0].append(query)

        # Determine the category
        category = "uncategorized"
        if isinstance(changes_info, dict):
            category_keys = [k for k in changes_info if k != "original_value"]
            if category_keys:
                category = category_keys[0]
        grouped_data[db_id][2].append(category)

        # Determine the source
        source_key = "other"
        if "bird" in source.lower():
            source_key = "bird"
        elif "spider" in source.lower():
            source_key = "spider"
        grouped_data[db_id][3].append(source_key)

        # Create and append gold standard results
        current_gold_list = []
        for gold_meta in gold_values:
            relevant_meta = {
                "table": gold_meta.get("table"),
                "column": gold_meta.get("column"),
                "value": gold_meta.get("value"),
            }
            gold_item = SearchableItem(
                item_id="gold", content="gold", metadata=relevant_meta
            )
            current_gold_list.append(RetrievalResult(item=gold_item, score=1.0))

        grouped_data[db_id][1].append(current_gold_list)

    print(f"Loaded benchmark data for {len(grouped_data)} databases from {os.path.basename(file_path)}.")
    return grouped_data


def get_system_configs():
    """
    Returns a dictionary of all system configurations, with expensive components
    pre-initialized.
    """
    print("Initializing all system searchers. This may take a moment...")

    # Pre-initialize all expensive components once
    #chess_searcher = Searcher(
    #    query_processor=ChessQueryProcessor(model_name_or_path=LLM_MODEL_PATH, cache_folder="./cache/keywords_chess", tensor_parallel_size=2, gpu_memory_utilization=0.20),
    #    retrievers=[ChessMinHashLshRetriever()],
    #    reranker=ChessSimilarityReranker(model_name=EMBEDDING_MODEL_PATH,embedding_similarity_threshold=0.8)
    #)

    omnisql_searcher = Searcher(
        query_processor=OmniSQLQueryProcessor(n=4),
        retrievers=[OmniSQLRetriever()]
    )

    #opensearch_searcher = Searcher(
    #    query_processor=OpenSearchQueryProcessor(model_name_or_path=LLM_MODEL_PATH, cache_folder="./cache/keywords_open_search", tensor_parallel_size=2, gpu_memory_utilization=0.65),
    #    retrievers=[OpenSearchDenseValueRetriever(model_name_or_path=EMBEDDING_MODEL_PATH)],
    #    reranker = OpenSearchRuleBasedReranker(similarity_threshold=0,score_proximity_filter=1,max_candidates_per_group=100000)
    #)

    #value_net_searcher = Searcher(
    #    query_processor=ValueNetQueryProcessor(),
    #    retrievers=[ValueNetRetriever()],
    #    reranker=ValueNetReranker()
    #)

    #bridge_searcher = Searcher(
    #    query_processor=BridgeQueryProcessor(),
    #    retrievers=[BridgeRetriever()],
    #    reranker=BridgeReranker()
    #)
    configs = {
        #"CHESS": {
        #    "searcher": chess_searcher,
        #    "get_db_specifics": lambda db_id: {
        #        "loader": ChessDBLoader(db_directory_path=os.path.join(DATABASES_ROOT, db_id)),
        #        "index_path": os.path.join(INDEXES_ROOT, "chess", db_id)
        #    }
        #},
        "OmniSQL": {
            "searcher": omnisql_searcher,
            "get_db_specifics": lambda db_id: {
                "loader": OmniSQLLoader(db_file_path=os.path.join(DATABASES_ROOT, db_id, f"{db_id}.sqlite")),
                "index_path": os.path.join(INDEXES_ROOT, "omnisql", db_id)
            }
        },
        #"OpenSearch": {
        #    "searcher": opensearch_searcher,
        #    "get_db_specifics": lambda db_id: {
        #       "loader": OpenSearchValueLoader(db_path=os.path.join(DATABASES_ROOT, db_id, f"{db_id}.sqlite")),
        #       "index_path": os.path.join(INDEXES_ROOT, "opensearch", db_id)
        #    }
        #}
        #"ValueNet": {
        #    "searcher": value_net_searcher,
        #    "get_db_specifics": lambda db_id: {
        #        "loader": None,
        #        "index_path": os.path.join(INDEXES_ROOT, "valuenet", db_id)
        #    }
        #}
        #"BRIDGE": {
        #    "searcher": bridge_searcher,
        #    "get_db_specifics": lambda db_id: {
        #        "loader":None,
        #        "index_path": os.path.join(INDEXES_ROOT, "bridge", db_id)
        #    }
        #}
    }
    print("All systems initialized.")
    return configs


def aggregate_metrics_from_details(
    query_metrics_details: List[PerQueryMetrics],
) -> Dict:
    """Aggregates a list of PerQueryMetrics objects into a single summary dictionary."""
    if not query_metrics_details:
        return {}

    total_tp = sum(m.true_positives for m in query_metrics_details)
    total_fp = sum(m.false_positives for m in query_metrics_details)
    total_fn = sum(m.false_negatives for m in query_metrics_details)
    num_queries = len(query_metrics_details)

    perfect_recall_sum = sum(m.perfect_recall for m in query_metrics_details)
    recall_non_numerical_sum = sum(m.recall_non_numerical for m in query_metrics_details)
    perfect_recall_non_numerical_sum = sum(m.perfect_recall_non_numerical for m in query_metrics_details)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    prr = perfect_recall_sum / num_queries if num_queries > 0 else 0.0
    recall_non_numerical = recall_non_numerical_sum / num_queries if num_queries > 0 else 0.0
    prr_non_numerical = perfect_recall_non_numerical_sum / num_queries if num_queries > 0 else 0.0

    return {
        "Overall Precision": precision,
        "Overall Recall": recall,
        "Overall F1 Score": f1,
        "Overall Perfect Recall Rate": prr,
        "Overall Recall (Non-Numerical)": recall_non_numerical,
        "Overall Perfect Recall Rate (Non-Numerical)": prr_non_numerical,
        "Total Queries": num_queries,
    }


def aggregate_by_category(
    all_query_details_with_cat: List[Tuple[PerQueryMetrics, str]]
) -> List[List]:
    """Aggregates metrics for each category."""

    metrics_by_category = defaultdict(list)
    for metric, category in all_query_details_with_cat:
        metrics_by_category[category].append(metric)

    results_for_table = []
    for category, metrics_list in metrics_by_category.items():
        agg_metrics = aggregate_metrics_from_details(metrics_list)
        results_for_table.append([
            category,
            agg_metrics["Overall Precision"],
            agg_metrics["Overall Recall"],
            agg_metrics["Overall F1 Score"],
            agg_metrics["Overall Perfect Recall Rate"],
            agg_metrics["Total Queries"]
        ])

    results_for_table.sort(key=lambda x: x[0])
    return results_for_table


def _calculate_and_prepare_k_table(
    metrics_at_k: Dict[int, Dict[str, float]],
    k_values: List[int],
    total_queries: int,
) -> List[List]:
    """Helper to calculate final metrics from aggregated counts for a wandb.Table."""
    if total_queries == 0:
        return []

    table_data = []
    for k in sorted(k_values):
        counts = metrics_at_k[k]
        total_tp = counts.get('tp', 0)
        total_fp = counts.get('fp', 0)
        total_fn = counts.get('fn', 0)
        perfect_recalls_non_numerical = counts.get('perfect_recalls_non_numerical', 0)

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        prr = perfect_recalls_non_numerical / total_queries if total_queries > 0 else 0.0

        table_data.append([k, precision, recall, f1, prr])
    return table_data

def generate_wandb_report(
    run_name: str,
    all_query_details_with_cat: List[Tuple[PerQueryMetrics, str]],
    all_db_summaries: List[EvaluationSummary],
    per_db_table_data: List[List],
    performance_at_k_data: List[List],
    avg_seconds_per_query: float
):
    """Initializes a W&B run and logs all aggregated tables and summaries."""
    if not all_query_details_with_cat:
        print(f"Skipping W&B report for {run_name} as there is no data.")
        return

    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=run_name,
        reinit=True,
        group=GROUP_NAME
    )

    # Log the detailed per-database performance table
    if per_db_table_data:
        columns_db = ["Database ID", "Num Queries", "Precision", "Recall", "F1 Score", "Perfect Recall Rate", "Recall (Non-Numerical)", "Perfect Recall Rate (Non-Numerical)"]
        per_db_table = wandb.Table(columns=columns_db, data=per_db_table_data)
        wandb.log({f"performance_by_database": per_db_table})

    # Calculate and log the performance by category
    category_table_data = aggregate_by_category(all_query_details_with_cat)
    if category_table_data:
        columns_cat = ["Category", "Precision", "Recall", "F1 Score", "Perfect Recall Rate", "# Queries"]
        per_cat_table = wandb.Table(columns=columns_cat, data=category_table_data)
        wandb.log({"performance_by_category": per_cat_table})

    # Log the performance at k table
    if performance_at_k_data:
        columns_k = ["@k", "Precision", "Recall", "F1 Score", "Perfect Recall Rate (Non-Numerical)"]
        perf_at_k_table = wandb.Table(columns=columns_k, data=performance_at_k_data)
        wandb.log({"performance_at_k": perf_at_k_table})

    # Calculate and log the overall aggregated summary for this specific run
    all_query_metrics = [metric for metric, cat in all_query_details_with_cat]
    aggregated_metrics = aggregate_metrics_from_details(all_query_metrics)
    aggregated_metrics["Avg Seconds per Query"] = avg_seconds_per_query
    wandb.summary.update(aggregated_metrics)

    print(f"\n--- Aggregated Summary for Report: {run_name} ---")
    summary_df = pd.DataFrame([aggregated_metrics])
    print(summary_df.to_markdown(index=False))

    wandb.finish()


def main():
    benchmark_data = load_and_group_benchmark_data(BENCHMARK_FILE)
    system_configs = get_system_configs()
    evaluator = RetrievalEvaluator()

    all_systems_failures = []

    # --- Outer loop: Iterate over each system ---
    for system_name, system_config in system_configs.items():
        print(f"\n{'='*30}\nRUNNING BENCHMARK FOR SYSTEM: {system_name}\n{'='*30}")
        json_output_for_system = []
        searcher = system_config["searcher"]

        # Data structures to collect results for the entire system
        system_all_db_summaries = []
        system_per_db_table_data = []
        system_all_query_details_with_cat = []

        # Data structures for metrics@k
        system_total_execution_time = 0.0
        system_total_queries = 0
        metrics_at_k_overall = defaultdict(lambda: defaultdict(int))
        metrics_at_k_bird = defaultdict(lambda: defaultdict(int))
        metrics_at_k_spider = defaultdict(lambda: defaultdict(int))
        num_queries_overall, num_queries_bird, num_queries_spider = 0, 0, 0

        # --- Inner loop: Iterate over each database ---
        for db_id, (queries, gold_standard, categories, sources) in benchmark_data.items():
            print(f"\n--- Evaluating on DB: {db_id} for System: {system_name} ---")

            db_specifics = system_config["get_db_specifics"](db_id)
            index_path = db_specifics["index_path"]

            if not os.path.exists(index_path) or not os.listdir(index_path):
                print(f"Index not found for {db_id} at {index_path}. Skipping.")
                continue

            # Update query counts
            num_queries_overall += len(queries)
            num_queries_bird += sum(1 for s in sources if s == 'bird')
            num_queries_spider += sum(1 for s in sources if s == 'spider')

            start_time = time.time()
            predicted_results = searcher.search(
                nlqs=queries, output_path=index_path, k=RETRIEVAL_DEPTH
            )
            for i, nlq in enumerate(queries):
                # Format the gold standard values into a set for easy lookup
                gold_results = gold_standard[i]
                gold_strings_set = {
                    f"{res.item.metadata.get('table')}.{res.item.metadata.get('column')}.{res.item.metadata.get('value')}"
                    for res in gold_results
                }
                # Use the first gold value as the representative "Correct value"
                correct_value_str = list(gold_strings_set)[0] if gold_strings_set else "N/A"

                # Format predicted values and filter out any that match the gold standard
                predicted_for_query = predicted_results[i]
                predicted_values_except_correct = []
                for pred_res in predicted_for_query:
                    pred_str = f"{pred_res.item.metadata.get('table')}.{pred_res.item.metadata.get('column')}.{pred_res.item.metadata.get('value')}"
                    if pred_str not in gold_strings_set:
                        predicted_values_except_correct.append(pred_str)
                
                # Create the record for this query
                record = {
                    "NLQ": nlq,
                    "Correct value": correct_value_str,
                    "Predicted values except correct value": predicted_values_except_correct
                }
                json_output_for_system.append(record)

            total_time = time.time() - start_time
            system_total_execution_time += total_time
            if queries:
                system_total_queries += len(queries)

            # --- End Timing & Log ---
            if len(queries) > 0:
                print(f"  Query Speed: {total_time / len(queries):.4f} seconds/query ({len(queries)} queries in {total_time:.2f}s)")


            # --- Evaluate metrics at different k values ---
            for k in K_VALUES:
                predictions_at_k = []
                for i, pred_list_for_query in enumerate(predicted_results):
                    gold_list_for_query = gold_standard[i]
                    granularity_fields = evaluator._get_granularity_fields_for_query(gold_list_for_query)
                    deduplicated_list = evaluator.deduplicate_by_granularity(pred_list_for_query, granularity_fields)
                    sliced_list = deduplicated_list[:k]
                    predictions_at_k.append(sliced_list)

                summary_at_k = evaluator.evaluate(predictions_at_k, gold_standard)
                for i, q_metric in enumerate(summary_at_k.per_query_details):
                    source = sources[i]

                    metrics_at_k_overall[k]['tp'] += q_metric.true_positives
                    metrics_at_k_overall[k]['fp'] += q_metric.false_positives
                    metrics_at_k_overall[k]['fn'] += q_metric.false_negatives
                    metrics_at_k_overall[k]['perfect_recalls_non_numerical'] += q_metric.perfect_recall_non_numerical

                    if source == 'bird':
                        metrics_at_k_bird[k]['tp'] += q_metric.true_positives
                        metrics_at_k_bird[k]['fp'] += q_metric.false_positives
                        metrics_at_k_bird[k]['fn'] += q_metric.false_negatives
                        metrics_at_k_bird[k]['perfect_recalls_non_numerical'] += q_metric.perfect_recall_non_numerical
                    elif source == 'spider':
                        metrics_at_k_spider[k]['tp'] += q_metric.true_positives
                        metrics_at_k_spider[k]['fp'] += q_metric.false_positives
                        metrics_at_k_spider[k]['fn'] += q_metric.false_negatives
                        metrics_at_k_spider[k]['perfect_recalls_non_numerical'] += q_metric.perfect_recall_non_numerical

            # --- Evaluate on full results for other metrics ---
            summary = evaluator.evaluate(predicted_results, gold_standard)
            system_all_db_summaries.append(summary)

            # Collect detailed results and tag with category and source
            for i, query_metric in enumerate(summary.per_query_details):
                system_all_query_details_with_cat.append((query_metric, categories[i], sources[i]))

                if query_metric.perfect_recall_non_numerical == 0.0:
                    all_systems_failures.append({
                        "system": system_name,
                        "db_id": db_id,
                        "category": categories[i],
                        "source": sources[i],
                        "query_index_in_db": query_metric.query_index,
                        "query": queries[i],
                        "missed_gold_items": query_metric.missed_items,
                        "retrieved_items": query_metric.retrieved_items,
                    })

            # Prepare data for the per-DB table
            system_per_db_table_data.append([
                db_id, summary.num_queries, summary.overall_precision, summary.overall_recall,
                summary.overall_f1_score, summary.perfect_recall_rate,
                summary.overall_recall_non_numerical, summary.perfect_recall_rate_non_numerical,
            ])

            print(f"  Recall on {db_id}: {summary.overall_recall:.4f}")
            print(f"  Recall (Non-Numerical) on {db_id}: {summary.overall_recall_non_numerical:.4f}")
        output_json_path = os.path.join(BASE_PATH, "predicted_values.json")
        print(f"\nWriting predicted values analysis for {system_name} to {output_json_path}...")
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(json_output_for_system, f, indent=4, ensure_ascii=False)
        print("Done.")
        # --- After all DBs are processed, generate the W&B reports for the system ---
        if system_all_query_details_with_cat:
            avg_seconds_per_query = system_total_execution_time / system_total_queries if system_total_queries > 0 else 0.0
            # 1. Overall Report
            overall_k_table_data = _calculate_and_prepare_k_table(metrics_at_k_overall, K_VALUES, num_queries_overall)
            generate_wandb_report(
                f"{system_name}-Overall-Report",
                [(m, c) for m, c, s in system_all_query_details_with_cat],
                system_all_db_summaries,
                system_per_db_table_data,
                overall_k_table_data,
                avg_seconds_per_query 
            )

            # 2. BIRD Report
            bird_query_details = [(m, c) for m, c, s in system_all_query_details_with_cat if s == 'bird']
            bird_db_summaries = [s for s, db_id in zip(system_all_db_summaries, benchmark_data.keys()) if any('bird' in src for src in benchmark_data[db_id][3])]
            bird_per_db_data = [row for row, db_id in zip(system_per_db_table_data, benchmark_data.keys()) if any('bird' in src for src in benchmark_data[db_id][3])]
            bird_k_table_data = _calculate_and_prepare_k_table(metrics_at_k_bird, K_VALUES, num_queries_bird)
            generate_wandb_report(f"{system_name}-BIRD-Report", bird_query_details, bird_db_summaries, bird_per_db_data, bird_k_table_data,avg_seconds_per_query)

            # 3. SPIDER Report
            spider_query_details = [(m, c) for m, c, s in system_all_query_details_with_cat if s == 'spider']
            spider_db_summaries = [s for s, db_id in zip(system_all_db_summaries, benchmark_data.keys()) if any('spider' in src for src in benchmark_data[db_id][3])]
            spider_per_db_data = [row for row, db_id in zip(system_per_db_table_data, benchmark_data.keys()) if any('spider' in src for src in benchmark_data[db_id][3])]
            spider_k_table_data = _calculate_and_prepare_k_table(metrics_at_k_spider, K_VALUES, num_queries_spider)
            generate_wandb_report(f"{system_name}-SPIDER-Report", spider_query_details, spider_db_summaries, spider_per_db_data, spider_k_table_data,avg_seconds_per_query)

    # Save all collected failure cases to the specified JSON file
    if all_systems_failures:
        output_dir = os.path.dirname(MISSED_ITEMS_FILE)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(MISSED_ITEMS_FILE, "w", encoding="utf-8") as f:
            json.dump(all_systems_failures, f, indent=4, ensure_ascii=False)
        print(f"\nSaved {len(all_systems_failures)} total failure cases to {MISSED_ITEMS_FILE}")

    print("\n\nBenchmarking finished for all systems.")

if __name__ == "__main__":
    main()