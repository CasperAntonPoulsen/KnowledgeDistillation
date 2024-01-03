import glob
import json
import tqdm
import os
import random
from transformers import HfArgumentParser, GPT2TokenizerFast, AutoTokenizer, AutoModelForCausalLM
from run_s2s import DataTrainingArguments
from datasets import load_dataset
from ni_collator import DataCollatorForNI
from dataclasses import dataclass, field



@dataclass
class BloomArguments(DataTrainingArguments):
    data_dir: str = field(
        default="data/splits", metadata={"help": "The directory for saving the NaturalInstructions train/dev/test splits."}
    )
    output_dir: str = field(
        default="output/bloom/", metadata={"help": "The directory for saving the NaturalInstructions train/dev/test splits."}
    )


if __name__ == "__main__":
    random.seed(123)
    parser = HfArgumentParser((BloomArguments))
    args, = parser.parse_args_into_dataclasses()
    raw_datasets = load_dataset(
        "src/ni_dataset.py", 
        data_dir=args.data_dir, 
        task_dir=args.task_dir, 
        max_num_instances_per_task=args.max_num_instances_per_task,
        max_num_instances_per_eval_task=args.max_num_instances_per_eval_task
    )

    tokenizer = AutoTokenizer.from_pretrained("/dtu/p1/johlau/distilled_models/bloom_both_1000it")
    model = AutoModelForCausalLM.from_pretrained("/dtu/p1/johlau/distilled_models/bloom_both_1000it")

    data_collator = DataCollatorForNI(
        tokenizer,
        model=None,
        padding="max_length" if args.pad_to_max_length else "longest",
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        add_task_definition=args.add_task_definition,
        num_pos_examples=args.num_pos_examples,
        num_neg_examples=args.num_neg_examples,
        add_explanation=args.add_explanation,
        text_only=False
    )

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "bloom_run_config.json"), "w") as fout:
        json.dump(args.__dict__, fout)

    existing_requests = {}
    if os.path.exists(os.path.join(args.output_dir, "predicted_examples.jsonl")):
        with open(os.path.join(args.output_dir, "predicted_examples.jsonl")) as fin:
            for line in fin:
                request_info = json.loads(line)
                existing_requests[request_info["bloom_input"]] = request_info["bloom_response"]

    with open(os.path.join(args.output_dir, "predicted_examples.jsonl"), "w") as fout:
        for example in tqdm.tqdm(raw_datasets["test"]):
            encoded_example = data_collator([example])
            
            example["bloom_input"] = encoded_example["input_ids"]
            example["bloom_target"] = tokenizer.decode(encoded_example["labels"][0])
            #print(example["bloom_target"])
            if example["bloom_input"] in existing_requests:
                response = existing_requests[example["bloom_input"]]
            else:
                response = model.generate(
                    input_ids=example["bloom_input"],
                    max_length = args.max_target_length
                )

            example["bloom_response"] = tokenizer.decode(response[0])
            example["bloom_input"] = tokenizer.decode(example["bloom_input"][0])
            #print(example["bloom_response"])
            #print(type(example["bloom_response"]))
            #print(example)
            # Note: we cut the generated text at the first period, since the GPT3 language model sometimes generates more than one sentences.
            # Our results show that this won't affect the instruct-GPT3 model very much, but will significantly improve the original GPT3 LM.
            example["prediction"] = example["bloom_response"].strip().split(".")[0]
            fout.write(json.dumps(example) + "\n")

        