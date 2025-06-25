db_root_directory=value_linking #root directory
dev_json=dev.json
train_json=dev.json
dev_table=merged_tables.json
train_table=merged_tables.json
dev_database=databases #dev database directory
fewshot_llm=gpt-4o-0513
DAIL_SQL=Bird/bird_dev.json     #dailsql json file 
bert_model=BAAI/bge-m3

python -u src/database_process/data_preprocess.py \
    --db_root_directory "${db_root_directory}" \
    --dev_json "${dev_json}" \
    --train_json "${train_json}" \
    --dev_table "${dev_table}" \
    --train_table "${train_table}"


#python -u src/database_process/prepare_train_queries.py \
#    --db_root_directory "${db_root_directory}" \
#    --model "${fewshot_llm}" 


#python -u src/database_process/generate_question.py \
#    --db_root_directory "${db_root_directory}" \
#    --DAIL_SQL "${DAIL_SQL}" 


#python -u src/database_process/make_emb.py \
#    --db_root_directory "${db_root_directory}" \
#    --dev_database "${dev_database}" \
#    --bert_model "${bert_model}"
#