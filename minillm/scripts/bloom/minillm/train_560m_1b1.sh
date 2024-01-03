#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=${2-3401}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${3-1}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# model
BASE_PATH=${1-"/home/MiniLLM"}
CKPT_NAME="bloom-560m"
CKPT="bigscience/bloom-560m"
TEACHER_CKPT_NAME="bloom-1b1"
TEACHER_CKPT="bigscience/bloom-1b1"
MP_SIZE=4
# data
PROMPT_DATA_DIR="${BASE_PATH}/processed_data/dolly/prompt/bloom/"
LM_DATA_DIR="${BASE_PATH}/processed_data/roberta/bloom/512/20M/"
# runtime
SAVE_PATH="${BASE_PATH}/results/bloom/train/minillm/"
# hp
GRAD_ACC=16
BATCH_SIZE=8
CHUNK_SIZE=8


OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --n-nodes ${NNODES}"
OPTS+=" --model-type bloom"
OPTS+=" --teacher-model-fp16"
OPTS+=" --gradient-checkpointing"
#OPTS+=" --model-parallel"
#OPTS+=" --model-parallel-size ${MP_SIZE}"
# data
OPTS+=" --prompt-data-dir ${PROMPT_DATA_DIR}"
OPTS+=" --lm-data-dir ${LM_DATA_DIR}"
OPTS+=" --dev-num 1000"
OPTS+=" --num-workers 0"
# hp
OPTS+=" --epochs 10"
OPTS+=" --total-iters 500"
OPTS+=" --kd-ratio 0.5"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --lr 5e-6"
OPTS+=" --lr-min 5e-6"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --max-length 256"
OPTS+=" --max-prompt-length 128"
OPTS+=" --warmup-iters 100"
OPTS+=" --scheduler-name cosine_trm"
# runtime
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --seed 10"
OPTS+=" --seed-ppo 42"
OPTS+=" --seed-lm 7"
OPTS+=" --save-interval 500"
OPTS+=" --eval-interval 100"
OPTS+=" --log-interval 16"
OPTS+=" --mid-log-num 1"
# ppo
OPTS+=" --type minillm"
OPTS+=" --ppo-epochs 4"
OPTS+=" --num-rollouts 32"
OPTS+=" --chunk-size ${CHUNK_SIZE}"
# minillm
OPTS+=" --length-norm"
OPTS+=" --single-step-reg"
OPTS+=" --teacher-mixed-alpha 0.2"
# reward
OPTS+=" --reward-scaling 0.5"
OPTS+=" --cliprange-reward 100"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
CMD="python3 -m torch.distributed.run ${DISTRIBUTED_ARGS} ${BASE_PATH}/train_minillm.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
