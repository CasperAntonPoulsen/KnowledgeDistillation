from lm_eval import evaluator
from lm_eval.api.registry import get_model
import sys
from lm_eval import tasks
import time
import utils
import json
from pathlib import Path

device = "cuda:0"
task_list=["headqa_en","headqa_es","paws_en","paws_es","paws_zh","xnli_en","xnli_es","xnli_zh", "xstorycloze_en", "xstorycloze_es", "xstorycloze_zh", "lambada_openai_mt_en", "lambada_openai_mt_es"]
# task_list=["headqa_en"]
batch_size = "auto" # Try using "auto" and see what happens
max_batch_size = 64
# tasks.initialize_tasks()


def eval_model(model_name, model_path):
    lm = get_model("hf")(
                pretrained=model_path,

            )

    results = evaluator.simple_evaluate(
                model=lm,
                model_args="",
                tasks=task_list,
                device=device,
                # pretrained_model=model,
                # tokenizer=tokenizer,
                use_cache=f"lm_cache/{model_name}",
                batch_size=batch_size,
                max_batch_size=max_batch_size,
                write_out=False,
                log_samples=False,)
    utils.save_results(model_name, results)
    print(evaluator.make_table(results))
    if "groups" in results:
        print(evaluator.make_table(results, "groups"))

    if "samples" in results.keys():
        samples = results.pop("samples")
    dumped = json.dumps(results, indent=2, default=utils._handle_non_serializable)
    
    output_path_file = "results/" + model_name + '.json'
    output_path_file = Path(output_path_file)
    output_path_file.open("w").write(dumped)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python eval_single.py <string> <string>")
        sys.exit(1)

    model_name = sys.argv[1]
    model_path = sys.argv[2]
    eval_model(model_name, model_path)
    
