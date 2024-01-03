
data_dir=data/splits/english
task_dir=data/tasks
output_dir=output/
max_num_instances_per_eval_task=100

echo "english instruction + 2 positive examples"

python3 src/run_bloom_spanish.py \
    --data_dir $data_dir \
    --task_dir $task_dir \
    --overwrite_cache \
    --max_num_instances_per_task 1 \
    --max_num_instances_per_eval_task ${max_num_instances_per_eval_task} \
    --add_task_definition True \
    --num_pos_examples 2 \
    --num_neg_examples 0 \
    --add_explanation False \
    --max_source_length 1024 \
    --max_target_length 128 \
    --output_dir ${output_dir}/english/bloom_spanish
python3 src/compute_metrics.py --predictions ${output_dir}/english/bloom_spanish/predicted_examples.jsonl --track english

echo "spanish instruction + 2 positive examples"

python3 src/run_bloom_spanish.py \
    --data_dir data/splits/spanish/ \
    --task_dir $task_dir \
    --overwrite_cache \
    --max_num_instances_per_task 1 \
    --max_num_instances_per_eval_task ${max_num_instances_per_eval_task} \
    --add_task_definition True \
    --num_pos_examples 2 \
    --num_neg_examples 0 \
    --add_explanation False \
    --max_source_length 1024 \
    --max_target_length 128 \
    --output_dir ${output_dir}/spanish/bloom_spanish
python3 src/compute_metrics.py --predictions ${output_dir}/spanish/bloom_spanish/predicted_examples.jsonl --track spanish


echo "chinese instruction + 2 positive examples"

python3 src/run_bloom_spanish.py \
    --data_dir data/splits/chinese/ \
    --task_dir $task_dir \
    --overwrite_cache \
    --max_num_instances_per_task 1 \
    --max_num_instances_per_eval_task ${max_num_instances_per_eval_task} \
    --add_task_definition True \
    --num_pos_examples 2 \
    --num_neg_examples 0 \
    --add_explanation False \
    --max_source_length 1024 \
    --max_target_length 128 \
    --output_dir ${output_dir}/chinese/bloom_spanish
python3 src/compute_metrics.py --predictions ${output_dir}/chinese/bloom_spanish/predicted_examples.jsonl --track chinese

