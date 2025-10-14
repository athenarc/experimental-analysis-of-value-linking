# Define variables
data_mode='dev' 
db_root_path=value_linking_prec_05
start=0 
end=500
pipeline_nodes='generate_db_schema+simplified_info_gathering+candidate_generate+align_correct+vote+evaluation'

engine1='Qwen/Qwen2.5-Coder-32B-Instruct'
pipeline_setup='{
    "generate_db_schema": {
        "engine": "'"$engine1"'",
        "bert_model": "BAAI/bge-m3",  
        "device":"cpu"
    },
    "candidate_generate":{
        "engine": "'"$engine1"'",
        "temperature": 0.7,  
        "n":6,
        "return_question":"True",
        "single":"False"
    },
    "align_correct":{
        "engine": "'"$engine1"'",
        "n":6,
        "bert_model": "BAAI/bge-m3",  
        "device":"cpu",
        "align_methods":"style_align+function_align+agent_align"
    }
}' 

python3 -u ./src/main.py --data_mode ${data_mode} --db_root_path ${db_root_path}\
        --pipeline_nodes ${pipeline_nodes} --pipeline_setup "$pipeline_setup"\
        --start ${start} --end ${end} \
  
