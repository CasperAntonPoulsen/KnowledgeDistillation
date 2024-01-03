from transformers import AutoModelForCausalLM, AutoConfig
import os
import torch
from lm_eval.api.registry import get_model
import time

path = "/home/data_shares/mapillary/llama/train/minillm_init/llama-7B"
model_args = f"pretrained={path},cache_dir=./llm_weights"

if os.path.exists(path):
    print("path exists")
else:
    print("path does not exist")


batch_size = 1
max_batch_size = 64

start_time = time.time()
lm = get_model("hf").create_from_arg_string(
            model_args,
            {
                "batch_size": batch_size,
                "max_batch_size": max_batch_size,
                # "device": device,
            },
        )
print("time to load model:", time.time() - start_time)
print("-----")
print(lm)
print("-----")
print(lm.model)