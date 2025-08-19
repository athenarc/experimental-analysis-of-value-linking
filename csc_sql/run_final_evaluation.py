import argparse
import os
from pathlib import Path
from cscsql.utils.file_utils import FileUtils
from cscsql.utils.infer_utils import major_voting2, calc_nl2sql_result

# This function is a simplified version of the one in the evaluation script
def print_data(score_lists, count_lists):
    levels = ['simple', 'moderate', 'challenging', 'total']
    print("\n" + "="*80)
    print("{:20} {:20} {:20} {:20} {:20}".format("", *levels))
    print("{:20} {:<20} {:<20} {:<20} {:<20}".format('count', *count_lists))
    print('='*25 + '    EXECUTION ACCURACY    ' + '='*26)
    print("{:20} {:<20.2f} {:<20.2f} {:<20.2f} {:<20.2f}".format('accuracy (%)', *[s * 100 for s in score_lists]))
    print("="*80)

def main(args):
    print(f"Loading predictions from: {args.pred_json_file}")
    pred_results = FileUtils.load_json(args.pred_json_file)

    print(f"Loading original data from: {args.original_data_json}")
    dev_data = FileUtils.load_json(args.original_data_json)

    if not pred_results:
        print("Prediction file is empty or could not be loaded. Exiting.")
        return

    # --- START OF FIX ---
    # Bypassing the buggy read_packed_sql function by loading gold data directly.
    has_gold = args.gold_sql_file and os.path.exists(args.gold_sql_file)
    ground_truth_sqls, gold_dbs = None, None
    if has_gold:
        print(f"Loading gold SQLs from: {args.gold_sql_file}")
        # Get the ordered list of db_ids from the original data file
        db_id_list = [item["db_id"] for item in dev_data]
        
        # Read the plain text gold.sql file
        with open(args.gold_sql_file, 'r') as f:
            gold_sql_lines = [line.strip() for line in f.readlines()]

        if len(db_id_list) != len(gold_sql_lines):
            print(f"Error: Mismatch between number of entries in {args.original_data_json} ({len(db_id_list)}) and {args.gold_sql_file} ({len(gold_sql_lines)})")
            return
            
        ground_truth_sqls = gold_sql_lines
        gold_dbs = [str(Path(args.db_path) / db_id / f"{db_id}.sqlite") for db_id in db_id_list]
        print(f"Successfully paired {len(ground_truth_sqls)} gold SQLs with their databases.")
    else:
        print("Gold SQL file not found. Will only generate predicted SQLs without scoring.")
    # --- END OF FIX ---

    pred_sql_key = "pred_sqls"
    sampling_num = len(pred_results[0][pred_sql_key])
    print(f"Detected sampling number (n): {sampling_num}")

    db_files = []
    pred_sqls = []
    for pred_data in pred_results:
        db_id = pred_data["db_id"]
        db_file_path = os.path.join(args.db_path, db_id, db_id + ".sqlite")
        db_files.extend([db_file_path] * sampling_num)
        pred_sqls.extend(pred_data[pred_sql_key])

    print("Running majority voting and SQL execution...")
    (mj_pred_correctness_list, _, _) = major_voting2(
        db_files,
        pred_sqls,
        sampling_num=sampling_num,
        ground_truth_sqls=ground_truth_sqls,
        gold_db_files=gold_dbs,
        num_cpus=args.num_cpus,
        timeout=args.timeout
    )

    # Save the final predicted SQL file
    output_sql_file = args.pred_json_file.replace(".json", "_pred_major_voting_sqls.sql")
    final_sqls = [item['sql'].replace("\n", " ") if item['sql'] else "Error SQL" for item in mj_pred_correctness_list]
    with open(output_sql_file, "w") as f:
        f.write("\n".join(final_sqls))
    print(f"Successfully generated final SQL file: {output_sql_file}")

    if not has_gold:
        print("Evaluation skipped as no gold SQL file was provided.")
        return

    # Calculate and print the final accuracy
    evaluation_scores = [res["correctness"] for res in mj_pred_correctness_list]
    metric = calc_nl2sql_result(evaluation_scores, dev_data)

    # Prepare data for printing
    score_lists = [metric.get("easy", 0), metric.get("moderate", 0), metric.get("challenging", 0), metric.get("all", 0)]
    count_lists = [metric.get("easy_total", 0), metric.get("moderate_total", 0), metric.get("challenging_total", 0), metric.get("all_total", 0)]

    print_data(score_lists, count_lists)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_json_file", required=True, help="Path to the JSON file with predictions (e.g., ..._merge.json)")
    parser.add_argument("--original_data_json", required=True, help="Path to the original data file (e.g., dev.json)")
    parser.add_argument("--gold_sql_file", help="Path to the ground truth SQL file (gold.sql)")
    parser.add_argument("--db_path", required=True, help="Path to the databases directory")
    parser.add_argument("--num_cpus", type=int, default=16, help="Number of CPUs for parallel execution")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout for SQL execution")
    args = parser.parse_args()
    main(args)