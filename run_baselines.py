import os
from cvr_extractor.cvr_extractor_llm import LLMExtractor, DictionaryExtractor
from cvr_extractor.cvr_extractor_ngrams import (
    NGramsExtractor,
)
from filtering.filtering_bridge import BridgeFilter
from value_index.value_index_abc import ValueLinker
from value_index.value_index_bm25_pyserini import BM25Index
from value_index.value_index_bridge import BRIDGEIndex
from value_index.value_index_chess import CHESSIndex
from value_index.value_index_dart import DartSQLIndex
from utils.value_linking_performance import ValueLinkingPerformance
import json
from utils.sqlite_db import DatabaseSqlite
import time
import logging
from tqdm import tqdm
from pathlib import Path
os.makedirs('logs', exist_ok=True)

def main():

    dictionary_of_trials = {
        "codes" : ["bm25", "ngrams", "lexical"],
        "chess" : ["chess", "dict", "none"],
        "bridge" : ["bridge", "none", "none"],
        "dart" : ["dart", "none", "none"]
    }

    
    for trial in dictionary_of_trials:
        index_type = dictionary_of_trials[trial][0]
        keywords_method = dictionary_of_trials[trial][1]
        filter_instance = dictionary_of_trials[trial][2]
        print(f"Running trial: {trial}")
        top_k = 10      #only used in codes for this scenario
        if index_type == "bm25":
            index = [BM25Index()]
        elif index_type == "chess":
            index_chess = CHESSIndex()
        elif index_type == "bridge":
            index = [BRIDGEIndex()]
        elif index_type == "dart":
            index = [DartSQLIndex()]

        if keywords_method == "ngrams":
            keyword_extractor = NGramsExtractor()
        elif keywords_method == "none":
            keyword_extractor = None
        elif keywords_method == "dict":
            keyword_extractor = DictionaryExtractor(
                "assets/bird_value_references_llm.json"
            )
        elif keywords_method == "llm":
            keyword_extractor = LLMExtractor(
                "assets/bird_value_references_llm.json"
            )
        if filter_instance == "lexical":
            filter = BridgeFilter()
        elif filter_instance == "none":
            filter = None
        databases_folder = (
            "dev_20240627/dev_databases"
        )
        base_path = ""
        if index_type == "bm25":
            base_path = "assets/bm25_indexes_bird"
        elif index_type == "chess":
            base_path = "assets/chess_indexes_bird"
        elif index_type == "bridge":
            base_path = "assets/bridge_indexes_bird"
        elif index_type == "dart":
            base_path = "assets/dart_indexes_bird"
            
        output_folder = base_path

        if index_type == "chess":
            for db in Path(databases_folder).iterdir():
                index_chess.create_index(db)
        else:
            linker = ValueLinker(index, keyword_extractor=keyword_extractor)
            if index_type != "bridge":
                for db_folder in os.listdir(databases_folder):
                    if db_folder.startswith("."):
                        continue
                    db_path = os.path.join(databases_folder, db_folder, f"{db_folder}.sqlite")
                    db_folder_path = os.path.join(databases_folder, db_folder)
                    output_folder_temp = os.path.join(output_folder, db_folder)
                    if not os.path.exists(output_folder_temp):
                        os.makedirs(output_folder_temp)
                        db = DatabaseSqlite(db_path)
                        linker.create_indexes(db, output_folder_temp)
        query_path = "dev_20240627/dev.json"
        input_folder = output_folder
        all_results = []
        previous_db_id = None
        db = None
        start = time.time()
        with open(query_path, "r") as f:
            json_data = json.load(f)
            for record in tqdm(json_data, desc="Processing queries"):
                db_id = record.get("db_id")
                query = record.get("question")
                if index_type == "chess":
                    keywords = keyword_extractor.extract_keywords(query)
                    db_path = os.path.join(databases_folder, db_id)
                    results = index_chess.query_index(keywords=keywords,index_path=db_path)
                    temp_results = set()
                    for table_name, columns in results.items():
                        for column_name, values in columns.items():
                            for value in values:
                                temp_results.add(f"{table_name}.{column_name}.{value}".lower())
                    all_results.append(temp_results)
                    continue
                if index_type != "bridge":
                    index_path = os.path.join(input_folder, db_id)
                else:
                    index_path = None
                if index_type == "bridge":
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
        if index_type != "chess":
            linker.print_timers()

        ground_truth_strings = (
            "assets/value_linking_dataset_list.json"
        )
        results_folder = (
            "assets/results_bird"
        )
        calculator = ValueLinkingPerformance(ground_truth_strings, results_folder)
        temp_path = f"assets/{trial}.json"
        all_results = [[x.lower() for x in result] for result in all_results]
        with open(temp_path, "w") as f:
            json.dump(all_results, f, indent=4)

        output_file = os.path.join(results_folder, "results.json")
        (
            perfect_recall_non_numerical,
            precision_non_numerical,
            column_recall_non_numerical,
            perfect_recall,
            precision,
            column_recall,
            recall_non_numerical,
            recall
        ) = calculator.calculate_accuracy_and_log(
            predicted_file=temp_path, output_file=output_file, remove_values=True
        )

        print(f"Perfect Recall Non-Numerical: {perfect_recall_non_numerical}")
        print(f"Precision Non-Numerical: {precision_non_numerical}")
        print(f"Column Recall Non-Numerical: {column_recall_non_numerical}")
        print(f"Perfect Recall: {perfect_recall}")
        print(f"Precision: {precision}")
        print(f"Column Recall: {column_recall}")
        print(f"Recall Non-Numerical: {recall_non_numerical}")
        print(f"Recall: {recall}")
        print(f"Query time: {query_time}")

        logging.info(f"Results for {trial}")
        logging.info(f"perfect_recall_non_numerical: {perfect_recall_non_numerical}")
        logging.info(f"precision_non_numerical: {precision_non_numerical}")
        logging.info(f"column_recall_non_numerical: {column_recall_non_numerical}")
        logging.info(f"perfect_recall: {perfect_recall}")
        logging.info(f"precision: {precision}")
        logging.info(f"column_recall: {column_recall}")
        logging.info(f"recall_non_numerical: {recall_non_numerical}")
        logging.info(f"recall: {recall}")
        logging.info(f"query_time: {query_time}")
        logging.info(f"\n\n############################################\n\n")
        
if __name__ == "__main__":
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename='logs/baseline.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    main()
