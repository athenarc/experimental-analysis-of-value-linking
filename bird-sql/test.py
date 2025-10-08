# File: validate_gt.py
import csv
import sys

gt_file = 'my_benchmark_precision_1/test_gold_sqls.txt'
error_found = False

print(f"Validating file: {gt_file}")

with open(gt_file, 'r', encoding='utf-8') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    for i, row in enumerate(reader):
        line_num = i + 1
        # Check 1: Is the row completely empty? (whitespace-only lines become this)
        if not row:
            print(f"ERROR on line {line_num}: Line is empty or contains only whitespace.")
            error_found = True
            continue

        # Check 2: Does the row have at least the SQL and db_id?
        if len(row) < 2:
            print(f"ERROR on line {line_num}: Line is missing the tab separator or the db_id.")
            print(f"   Content: {row}")
            error_found = True

if not error_found:
    print("Validation successful! No obvious errors found.")
else:
    print("\nValidation failed. Please fix the errors listed above.")
    sys.exit(1)