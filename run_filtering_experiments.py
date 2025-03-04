import os
from cvr_extractor.cvr_extractor_ngrams import (
    NGramsExtractor,
)
from value_index.value_index_abc import ValueLinker
from value_index.value_index_bm25_pyserini import BM25Index
from utils.value_linking_performance import ValueLinkingPerformance
from value_index.value_index_faiss_flat import FaissFlatIndex
from filtering.filtering_bridge import BridgeFilter
from filtering.filtering_cosine import CosineFilter
import json
from utils.sqlite_db import DatabaseSqlite
import time
import logging
from tqdm import tqdm

def main():
    dictionary_of_trials = {
        "faiss" : ["faiss", "ngrams", "lexical","0.50"],
        "faiss" : ["faiss", "ngrams", "lexical","0.85"],
        "faiss" : ["faiss", "ngrams", "lexical","0.99"],
        "faiss" : ["faiss", "ngrams", "semantic","0.70"],
        "faiss" : ["faiss", "ngrams", "semantic","0.75"],
        "faiss" : ["faiss", "ngrams", "semantic","0.80"],
        "bm25_faiss" : ["bm25_faiss", "ngrams", "lexical","0.50"],
        "bm25_faiss" : ["bm25_faiss", "ngrams", "lexical","0.85"],
        "bm25_faiss" : ["bm25_faiss", "ngrams", "lexical","0.99"],
        "bm25_faiss" : ["bm25_faiss", "ngrams", "semantic","0.70"],
        "bm25_faiss" : ["bm25_faiss", "ngrams", "semantic","0.75"],
        "bm25_faiss" : ["bm25_faiss", "ngrams", "semantic","0.80"],
    }

    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        filename='logs/filtering_experiments.log',  
        level=logging.INFO,     
        format='%(asctime)s - %(levelname)s - %(message)s'  
    )
    for trial in dictionary_of_trials:
        index_type = dictionary_of_trials[trial][0]
        keywords_method = dictionary_of_trials[trial][1]
        filter_instance = dictionary_of_trials[trial][2]
        #threshold is the fourth element in the list, convert it to float
        threshold = float(dictionary_of_trials[trial][3])
        print(f"Running trial: {trial}_{filter_instance}_{threshold}")
        top_k = 10      
        if index_type == "faiss":
            index = [FaissFlatIndex()]
        elif index_type == "bm25_faiss":
            index = [BM25Index(), FaissFlatIndex()]
            
        if keywords_method == "ngrams":
            keyword_extractor = NGramsExtractor()
        model_name = "WhereIsAI/UAE-Large-V1"
        if filter_instance == "lexical":
            filter = BridgeFilter(filter_threshold=threshold)
        elif filter_instance == "semantic":
            filter = CosineFilter(threshold=threshold,model_name=model_name)
        databases_folder = (
            "dev_20240627/dev_databases"
        )
        output_folder = "assets/mix_indexes_bird"


        linker = ValueLinker(index, keyword_extractor=keyword_extractor)
        for db_folder in os.listdir(databases_folder):
            if db_folder.startswith("."):
                continue
            db_path = os.path.join(databases_folder, db_folder, f"{db_folder}.sqlite")
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
                index_path = os.path.join(input_folder, db_id)
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
            "assets/value_linking_dataset_list.json"
        )
        results_folder = (
            "assets/results_bird"
        )
        calculator = ValueLinkingPerformance(ground_truth_strings, results_folder)

        temp_path = "assets/temp.json"
        # Lowercase all the results
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
    main()
