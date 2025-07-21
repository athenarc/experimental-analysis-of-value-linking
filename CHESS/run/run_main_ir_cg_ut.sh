source ./.env
data_mode=$DATA_MODE # Options: 'dev', 'train' 
data_path=$DATA_PATH # UPDATE THIS WITH THE PATH TO THE TARGET DATASET

config="./run/configs/CHESS_IR_CG_UT.yaml"

num_workers=1 # Number of workers to use for parallel processing, set to 1 for no parallel processing

python3 -u ./src/main.py --data_mode dev --data_path ./data/value_linking/human_annotated.json --config ./run/configs/CHESS_IR_CG_UT.yaml \
        --num_workers 1 --pick_final_sql true 

