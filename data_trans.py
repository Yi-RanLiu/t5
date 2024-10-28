"""
-*- coding: utf-8 -*-

@Author  : houcg
@Time    : 2024/10/27 12:21
"""
import json

import pandas
from sklearn.model_selection import train_test_split

total_data = {"questions": [], "answers": []}

input_len = []
output_len = []

with open("./data/alpaca_zh_demo.json", encoding="utf-8") as f:
    data = json.load(f)
    for i in data:
        total_data["questions"].append(i['instruction'])
        total_data["answers"].append(i['output'])
        
        input_len.append(len(i['instruction']))
        output_len.append(len(i['output']))
        
print(sum(input_len) / len(input_len))
print(sum(output_len) / len(output_len))

total_data = pandas.DataFrame(total_data)

train_data, test_val_data = train_test_split(total_data, test_size=0.2, random_state=5)
test_data, val_data = train_test_split(test_val_data, test_size=0.5, random_state=5)


train_data = total_data
test_data = test_data.reset_index(drop=True)
val_data = val_data.reset_index(drop=True)

print(len(train_data))
print(len(test_data))
print(len(val_data))


with open("./data/train.tsv", "w", encoding="utf-8") as f:
    for i in range(len(train_data)):
        f.write(train_data["answers"][i].strip().replace("\n" , "") + "\t" + train_data["questions"][i].strip().replace("\n" , "") + "\n")

with open("./data/val.tsv", "w", encoding="utf-8") as f:
    for i in range(len(val_data)):
        f.write(val_data["answers"][i].strip().replace("\n" , "") + "\t" + val_data["questions"][i].strip().replace("\n" , "") + "\n")

with open("./data/test.tsv", "w", encoding="utf-8") as f:
    for i in range(len(test_data)):
        f.write(test_data["answers"][i].strip().replace("\n" , "") + "\t" + test_data["questions"][i].strip().replace("\n" , "") + "\n")
