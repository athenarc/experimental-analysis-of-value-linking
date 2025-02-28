import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from darelabdb.nlp_value_linking.cvr_extractor.cvr_extractor_llm import LLMExtractor
from darelabdb.nlp_value_linking.cvr_extractor.cvr_extractor_ner import NERExtractor
from darelabdb.nlp_value_linking.cvr_extractor.cvr_extractor_plain import PlainExtractor
from darelabdb.nlp_value_linking.cvr_extractor.cvr_extractor_ngrams import (
    NGramsExtractor,
)
from darelabdb.nlp_value_linking.filtering.filtering_bge_reranker import (
    BGEFlagRerankerFilter,
)
from darelabdb.nlp_value_linking.filtering.filtering_bridge import BridgeFilter
from darelabdb.nlp_value_linking.filtering.filtering_jina_reranker import (
    JinaAiRerankerFiltering,
)
from darelabdb.nlp_value_linking.filtering.filtering_llm_reranker import (
    BGEFlagLLMRerankerFilter,
)
from darelabdb.nlp_value_linking.filtering.filtering_llm import LLMFilter
from darelabdb.nlp_value_linking.value_index.value_index_bm25_pyserini import BM25Index
from darelabdb.nlp_value_linking.filtering.filtering_cosine import CosineFilter
from darelabdb.nlp_value_linking.value_index.value_index_colbert import ColbertIndex
from darelabdb.nlp_value_linking.value_index.value_index_faiss_flat import (
    FaissFlatIndex,
)
from darelabdb.nlp_value_linking.value_index.value_index_minhashlsh import (
    MinHashForestIndex,
)
from darelabdb.nlp_value_linking.value_index.value_index_picklist import PicklistIndex
from darelabdb.nlp_value_linking.value_index.value_index_abc import ValueLinker
from development.value_linking.ValueIndexABC_devlopment import (
    DartSQLIndex,
    BRIDGEIndex,
    CHESSIndex,
)
from darelabdb.nlp_metrics.value_linking_accuracy import ValueLinkingPerformance
import json
import wandb
from io import StringIO
import sys
from darelabdb.utils_database_connector.sqlite_db import DatabaseSqlite
import time

# make oonly the second gpu visible


def main():
    # All parameters are defined here

    # Comprehensive experiment parameters
    experiment_params = {
        "delimiter": " ",
        "delimiter_to_log": "space" if " " else "custom",
        "per_value": False,
        "skip_non_text": False,
        "index_type": "mix",
        "ngrams_n": 4,
        "keywords_method": "LLM",
        "top_k": 10,
        "chess_edit_distance": 0.3,
        "chess_embedding_similarity_threshold": 0.6,
        "minhash_threshold": 0.5,
        "minhash_signature_size": 128,
        "minhash_ngrams": 4,
        "filter_instance": "bridge",
        "filter_threshold": 0.9,
    }

    # Initialize wandb with the full parameter set
    wandb.init(
        project="value_linking_bird_v3",
        name="bm25_minhash_faiss_space_top10_LLM_bridge",
        config=experiment_params,
        entity="darelab",
        group="nl_experiments",
        settings=wandb.Settings(init_timeout=120),
    )

    # Extract parameters for use in the script
    delimiter = experiment_params["delimiter"]
    per_value = experiment_params["per_value"]
    skip_non_text = experiment_params["skip_non_text"]
    index_type = experiment_params["index_type"]
    keywords_method = experiment_params["keywords_method"]
    top_k = experiment_params["top_k"]
    chess_edit_distance = experiment_params["chess_edit_distance"]
    chess_embedding_similarity_threshold = experiment_params[
        "chess_embedding_similarity_threshold"
    ]

    # Set up the index based on `index_type`
    if index_type == "bm25":
        index = [BM25Index(per_value=True, delimeter=delimiter)]
    elif index_type == "faiss":
        index = [
            FaissFlatIndex(
                delimeter=delimiter, per_value=per_value, skip_non_text=skip_non_text
            )
        ]
    elif index_type == "minhash":
        minhash_signature_size = experiment_params["minhash_signature_size"]
        minhash_threshold = experiment_params["minhash_threshold"]
        index = [
            MinHashForestIndex(
                per_value=per_value,
                delimeter=delimiter,
                skip_non_text=skip_non_text,
                minhash_signature_size=minhash_signature_size,
            )
        ]
    elif index_type == "picklist":
        categorical_size = 100
        threshold = 0.8
        index = [
            PicklistIndex(
                per_value=per_value,
                categorical_size=categorical_size,
                threshold=threshold,
            )
        ]
    elif index_type == "chess":
        index = [CHESSIndex()]
    elif index_type == "colbert":
        index = [
            ColbertIndex(
                delimeter=delimiter, per_value=per_value, skip_non_text=skip_non_text
            )
        ]
    elif index_type == "mix":
        minhash_signature_size = experiment_params["minhash_signature_size"]
        minhash_threshold = experiment_params["minhash_threshold"]
        categorical_size = 100
        threshold = 0.8
        index = [
            BM25Index(per_value=per_value, delimeter=delimiter),
            FaissFlatIndex(
                delimeter=delimiter, per_value=per_value, skip_non_text=skip_non_text
            ),
            MinHashForestIndex(
                per_value=per_value,
                delimeter=delimiter,
                skip_non_text=skip_non_text,
                minhash_signature_size=minhash_signature_size,
            ),
        ]
        # index = [BM25Index(per_value=per_value, delimeter=delimiter),MinHashForestIndex(per_value=per_value, delimeter=delimiter, minhash_threshold=minhash_threshold,skip_non_text=skip_non_text, minhash_signature_size=minhash_signature_size)]
        # index = [BM25Index(per_value=per_value, delimeter=delimiter),FaissFlatIndex(delimeter=delimiter, per_value=per_value, skip_non_text=skip_non_text)]
        # index = [FaissFlatIndex(delimeter=delimiter, per_value=per_value, skip_non_text=skip_non_text),MinHashForestIndex(per_value=per_value, delimeter=delimiter, minhash_threshold=minhash_threshold,skip_non_text=skip_non_text, minhash_signature_size=minhash_signature_size)]
        # index = [PicklistIndex(per_value=per_value, categorical_size=categorical_size,threshold=threshold),BM25Index(per_value=per_value, delimeter=delimiter),FaissFlatIndex(delimeter=delimiter, per_value=per_value, skip_non_text=skip_non_text),MinHashForestIndex(per_value=per_value, delimeter=delimiter, minhash_threshold=minhash_threshold,skip_non_text=skip_non_text, minhash_signature_size=minhash_signature_size)]

    elif index_type == "bridge":
        index = [BRIDGEIndex()]
    elif index_type == "dart":
        index = [DartSQLIndex()]

    if keywords_method == "ngrams":
        n = experiment_params["ngrams_n"]
        keyword_extractor = NGramsExtractor(n=n)
    elif keywords_method == "ner":
        keyword_extractor = NERExtractor()
    elif keywords_method == "LLM":
        keyword_extractor = LLMExtractor()
    elif keywords_method == "plain":
        keyword_extractor = PlainExtractor()
    elif keywords_method == "none":
        keyword_extractor = None

    filter_instance = experiment_params["filter_instance"]
    filter_threshold = experiment_params["filter_threshold"]
    if filter_instance == "bridge":
        filter = BridgeFilter(filter_threshold=filter_threshold)
    elif filter_instance == "jina":
        filter = JinaAiRerankerFiltering(threshold=filter_threshold)
    elif filter_instance == "bgeflag":
        filter = BGEFlagRerankerFilter(threshold=filter_threshold)
    elif filter_instance == "bgeflagllm":
        filter = BGEFlagLLMRerankerFilter(threshold=filter_threshold)
    elif filter_instance == "none":
        filter = None
    elif filter_instance == "llama":
        filter = LLMFilter, ()
    elif filter_instance == "bert":
        filter = CosineFilter(threshold=filter_threshold)
    # end of parameter definition
    databases_folder = (
        "/data/hdd1/users/akouk/BIRD-dev/dev_20240627/dev_databases/dev_databases"
    )
    base_path = ""
    if index_type == "faiss":
        base_path = "/data/hdd1/users/akouk/DarelabDB/development/value_linking/faiss_indexes_bird"
    elif index_type == "bm25":
        base_path = "/data/hdd1/users/akouk/DarelabDB/development/value_linking/bm25_indexes_bird"
    elif index_type == "minhash":
        base_path = "/data/hdd1/users/akouk/DarelabDB/development/value_linking/minhash_indexes_bird"
    elif index_type == "chess":
        base_path = "/data/hdd1/users/akouk/DarelabDB/development/value_linking/chess_indexes_bird"
    elif index_type == "colbert":
        base_path = "/data/hdd1/users/akouk/DarelabDB/development/value_linking/colbert_indexes_bird"
    elif index_type == "mix":
        base_path = "/data/hdd1/users/akouk/DarelabDB/development/value_linking/mix_indexes_bird"
    elif index_type == "bridge":
        base_path = "/data/hdd1/users/akouk/DarelabDB/development/value_linking/bridge_indexes_bird"
    elif index_type == "dart":
        base_path = "/data/hdd1/users/akouk/DarelabDB/development/value_linking/dart_indexes_bird"
    elif index_type == "picklist":
        base_path = "/data/hdd1/users/akouk/DarelabDB/development/value_linking/picklist_indexes_bird"

    if per_value:
        output_folder = os.path.join(base_path, "per_value")
    elif per_value is None:
        # do nothing
        output_folder = base_path
    else:
        # then output file is deliimeter_{delimiter}
        if delimiter == " ":
            delimiter = "space"
        output_folder = os.path.join(base_path, f"delimiter_{delimiter}")

    # if the folder does not exist, perform the indexing

    linker = ValueLinker(index, keyword_extractor=keyword_extractor)
    if index != "bridge":
        for db_folder in os.listdir(databases_folder):
            if db_folder.startswith("."):
                continue
            db_path = os.path.join(databases_folder, db_folder, f"{db_folder}.sqlite")
            output_folder_temp = os.path.join(output_folder, db_folder)
            if not os.path.exists(output_folder_temp):
                os.makedirs(output_folder_temp)
                db = DatabaseSqlite(db_path)
                linker.create_indexes(db, output_folder_temp)

    query_path = "/data/hdd1/users/akouk/BIRD-dev/dev_20240627/dev.json"
    input_folder = output_folder
    all_results = []
    previous_db_id = None
    db = None
    start = time.time()
    with open(query_path, "r") as f:
        json_data = json.load(f)
        for record in json_data:
            db_id = record.get("db_id")
            query = record.get("question")
            print(f"Processing query: {query}")
            if index_type != "bridge":
                index_path = os.path.join(input_folder, db_id)
            else:
                index_path = None
            if index_type == "chess" or index_type == "bridge":
                if db_id != previous_db_id:
                    db_path = os.path.join(databases_folder, db_id, f"{db_id}.sqlite")
                    db = DatabaseSqlite(db_path)
                    previous_db_id = db_id
                results = linker.query_indexes(
                    input_text=query,
                    index_path=index_path,
                    top_k=top_k,
                    filter_instance=filter,
                    database=db,
                )
            else:
                results = linker.query_indexes(
                    input_text=query,
                    index_path=index_path,
                    top_k=top_k,
                    filter_instance=filter,
                )
            all_results.append(results)
    query_time = time.time() - start
    linker.print_timers()

    ground_truth_strings = (
        "/data/hdd1/users/akouk/BIRD-dev/dev_20240627/dev_strings_lower.json"
    )
    results_folder = (
        "/data/hdd1/users/akouk/DarelabDB/development/value_linking/results_bird"
    )
    calculator = ValueLinkingPerformance(ground_truth_strings, results_folder)

    temp_path = "/data/hdd1/users/akouk/DarelabDB/development/value_linking/temp.json"
    # Lowercase all the results
    all_results = [[x.lower() for x in result] for result in all_results]
    with open(temp_path, "w") as f:
        json.dump(all_results, f, indent=4)

    output_file = os.path.join(results_folder, "results.json")
    (
        acc_with_filter,
        prec_with_filter,
        prefix_acc_with_filter,
        acc_without_filter,
        prec_without_filter,
        prefix_acc_without_filter,
        average_partial_recall_with_filter,
        average_partial_recall_without_filter,  # Renamed variables
    ) = calculator.calculate_accuracy_and_log(
        predicted_file=temp_path, output_file=output_file, remove_values=True
    )

    # Log all metrics with consistent naming
    wandb.log(
        {
            "accuracy_with_filter": acc_with_filter,
            "precision_with_filter": prec_with_filter,
            "column_accuracy_with_filter": prefix_acc_with_filter,
            "average_partial_recall_with_filter": average_partial_recall_with_filter,  # Updated key
            "accuracy_without_filter": acc_without_filter,
            "precision_without_filter": prec_without_filter,
            "column_accuracy_without_filter": prefix_acc_without_filter,
            "average_partial_recall_without_filter": average_partial_recall_without_filter,  # Updated key
            "actual_query_time": query_time,
        }
    )

    def parse_last_timers(timer_output):
        lines = timer_output.strip().split("\n")
        timers = {}
        for line in lines[-2:]:  # Only consider the last two lines
            if "time:" in line:  # Look for the specific keyword "time:"
                key, value = line.split(":")
                timers[key.strip()] = float(value.split()[0])  # Extract seconds
        return timers

    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()
    linker.print_timers()
    sys.stdout = old_stdout
    timer_output = captured_output.getvalue()

    timing_metrics = {}
    lines = timer_output.strip().split("\n")
    for line in lines[-2:]:  # Only process the last two lines
        if "time:" in line:  # Ensure the line contains timing information
            key, value = line.split(":")
            timing_metrics[key.strip()] = float(value.split()[0])  # Extract seconds

    # Log the timing metrics to wandb
    wandb.log(timing_metrics)
    wandb.finish()

    # remove the temp file
    os.remove(temp_path)

    # print all the results
    print(f"Accuracy with filter: {acc_with_filter}")
    print(f"Precision with filter: {prec_with_filter}")
    print(f"Column accuracy with filter: {prefix_acc_with_filter}")
    print(f"Accuracy without filter: {acc_without_filter}")
    print(f"Precision without filter: {prec_without_filter}")
    print(f"Column accuracy without filter: {prefix_acc_without_filter}")


if __name__ == "__main__":
    main()
