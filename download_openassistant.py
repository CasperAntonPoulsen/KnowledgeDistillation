from datasets import load_dataset
import datasets
import pandas as pd

dataset = load_dataset("OpenAssistant/oasst1")

df = dataset["train"].to_pandas()

df_assistant = df[df["role"] == "assistant"][["message_id","parent_id","text","lang"]]
df_prompter = df[(df["role"] == "prompter") & (df["parent_id"].isna()) ][["message_id","parent_id","text","lang"]]

df_combined= df_assistant.merge( df_prompter, left_on="parent_id", right_on="message_id", suffixes= ["_response", "_prompt"])
df_combined = df_combined.drop_duplicates(["parent_id_response"])
df_combined["input"] = ""
df_combined = df_combined[['text_prompt', "input",'text_response','lang_prompt']]
df_combined.columns = ['instruction', "input", 'output', 'lang']

print(len(df_combined))
with open("/dtu/p1/johlau/LMOps/minillm/data/dolly/raw.jsonl", "w+") as file:
    df_combined[df_combined["lang"] == "es"].to_json(file, orient="records", lines=True)