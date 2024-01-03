BASE_PATH=${1}

MAX_LENGTH=256

PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_pretrain.py \
    --data-dir ${BASE_PATH}/data/roberta_en \
    --processed-data-dir ${BASE_PATH}/processed_data/roberta_both/llama/${MAX_LENGTH}/ \
    --model-path bigscience/bloom-3b  \
    --max-length ${MAX_LENGTH} \
    --train-num 20000000 \
    --data-process-workers 32 \
    --dev-num 10000 \