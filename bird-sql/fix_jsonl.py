# File: fix_jsonl.py
import json
from pathlib import Path

benchmark_dir = Path("my_benchmark")
input_json_path = benchmark_dir / "test_all.json"
output_jsonl_path = benchmark_dir / "test_all.jsonl"

print(f"Reading from: {input_json_path}")
with open(input_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Found {len(data)} entries. Writing to: {output_jsonl_path}")
with open(output_jsonl_path, "w", encoding="utf-8") as f:
    for i, entry in enumerate(data):
        # The original prep_data script adds these fields. Let's replicate that.
        entry.setdefault("question_id", str(i))
        entry["db_path"] = str(
            benchmark_dir / "test_databases" / entry["db_id"] / f"{entry['db_id']}.sqlite"
        )
        entry.setdefault("difficulty", "easy")
        entry.setdefault("SQL", entry.get("SQL", ""))

        json_string = json.dumps(entry, ensure_ascii=False)
        f.write(json_string + "\n")
print("Successfully created the JSONL file.")