import os
import json
import pandas as pd

results_folder = 'results'

keys_of_interest = ['acc,none', 'acc_norm,none', 'perplexity,none']  # Replace with actual keys
model_type = "bloom"
# Initialize a list to store the data
data = []

# Iterate over each subfolder in the 'results' folder
for subfolder in os.listdir(results_folder):
    subfolder_path = os.path.join(results_folder, subfolder)
    if model_type not in subfolder or "256" in subfolder:
        continue
    # Check if it's a directory
    if os.path.isdir(subfolder_path):
        json_file_path = os.path.join(subfolder_path, 'results.json')

        # Check if the JSON file exists
        if os.path.isfile(json_file_path):
            with open(json_file_path, 'r') as file:
                json_data = json.load(file)
                json_data = json_data['results']

            # Extract the required data
            tasks = list(json_data.keys())
            tasks.remove("know_dist")
            row_data = {}
            row_data['model'] = subfolder  # Add the model name
            for task in tasks:
                task_data = {task+"_"+key.removesuffix(",none"): json_data[task].get(key, None) for key in keys_of_interest}
                row_data.update(task_data)
            
            data.append(row_data)

# Create a DataFrame
df = pd.DataFrame(data)
df = df.dropna(axis=1, how='all')

# Display the DataFrame
print(df)
df.to_csv(model_type+'_results.csv', index=False)