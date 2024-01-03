
data_dir=data/splits/english
task_dir=data/tasks
output_dir=output/
max_num_instances_per_eval_task=100

echo "english instruction + 2 positive examples"

python3 src/run_llama13b.py \
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
    --output_dir ${output_dir}/english/llama13b
python3 src/compute_metrics.py --predictions ${output_dir}/english/llama13b/predicted_examples.jsonl --track english
done

echo "spanish instruction + 2 positive examples"

python3 src/run_llama13b.py \
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
    --output_dir ${output_dir}/spanish/llama13b
python3 src/compute_metrics.py --predictions ${output_dir}/spanish/llama13b/predicted_examples.jsonl --track spanish
done

echo "chinese instruction + 2 positive examples"

python3 src/run_llama13b.py \
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
    --output_dir ${output_dir}/chinese/llama13b
python3 src/compute_metrics.py --predictions ${output_dir}/chinese/llama13b/predicted_examples.jsonl --track chinese
done
