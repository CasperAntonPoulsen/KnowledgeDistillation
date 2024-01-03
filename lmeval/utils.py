import pandas as pd
import io
import hashlib
import torch
import json


def extract_metrics(results_dict):
    # Extracting the 'results' dictionary
    metrics = results_dict.get('results', {})
    # Flattening the nested structure
    flat_metrics = {}
    for test, scores in metrics.items():
        for metric, value in scores.items():
            flat_metrics[f'{test}_{metric}'] = value
    return flat_metrics


def get_dataframe(results_dict):
    return pd.DataFrame([extract_metrics(results_dict)])

def save_results(name, results_dict):
    print(name, " ", results_dict)
    df = get_dataframe(results_dict)
    df.to_csv("results/" + name + '.csv')
   

def create_checksum(model):
    model_parameters = model.state_dict()
    buffer = io.BytesIO()

    param_bytes = torch.save(model_parameters, buffer)
    checksum = hashlib.sha256(param_bytes).hexdigest()

    return checksum

def verify_checksum(model, checksum):
    return checksum == create_checksum(model)
    

def count_prompts_in_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
        return
    except json.JSONDecodeError:
        print(f"The file {file_path} is not a valid JSON file.")
        return

    if not isinstance(data, list):
        print("The JSON file does not contain a list.")
        return

    total_count = {
        'prompt_2': 0,
        'prompt_3': 0,
        'prompt_4': 0
    }

    for item in data:
        if not isinstance(item, dict):
            continue

        for key in item:
            if key in total_count:
                total_count[key] += 1

    return total_count


def _handle_non_serializable(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)