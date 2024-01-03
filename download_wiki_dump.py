from datasets import load_dataset
import re

dataset = load_dataset("graelo/wikipedia", "20230901.es", split='train')


num = 0
with open("/dtu/p1/johlau/LMOps/minillm/data/roberta/data.txt", "w") as f:
    for data in dataset:
        f.write(re.sub(r"\n+", "<@x(x!>", data['text']) + "\n")
        num += 1

print("Number of lines:", num)

