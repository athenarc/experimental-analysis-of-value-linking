import json
import os
import shutil

json_path = 'assets/all_benchmarks_human/all_dump_good.json'
source_dir = 'assets/all_databases'
destination_dir = 'assets/retrievers/databases'

with open(json_path, 'r') as f:
    records = json.load(f)

distinct_db_ids = set()
for record in records:
    distinct_db_ids.add(record['db_id'])

os.makedirs(destination_dir, exist_ok=True)

for db_id in distinct_db_ids:
    source_db_path = os.path.join(source_dir, db_id)
    destination_db_path = os.path.join(destination_dir, db_id)
    
    if os.path.exists(source_db_path):
        if os.path.exists(destination_db_path):
            shutil.rmtree(destination_db_path)
        shutil.copytree(source_db_path, destination_db_path)