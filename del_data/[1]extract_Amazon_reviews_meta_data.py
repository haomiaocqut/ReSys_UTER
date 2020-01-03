#!/usr/bin/python3
# 2019.11.04
# Author Zhang Yihao @NUS

from collections import defaultdict, OrderedDict
import numpy as np
import json

dataName = "Video"

num_ui_link = 20  # the number of each user link to items
num_iu_link = 0  # the number of each item link to user

# user -> item
ui_dict = defaultdict(list)
iu_dict = defaultdict(list)

reviews_data = []
meta_data = []
with open('G:/Datasets/Amazon datasets/' + dataName + '/reviews_' + dataName + '.json') as f:
    for line in f:
        reviews_data.append(json.loads(line))
    f.close()
with open('G:/Datasets/Amazon datasets/' + dataName + '/meta_' + dataName + '.json') as f:
    for line in f:
        line_dict = json.dumps(eval(line))
        meta_data.append(json.loads(line_dict))
    f.close()

for line_data in reviews_data:
    user_id = line_data["reviewerID"]
    item_id = line_data["asin"]
    ui_dict[user_id].append(item_id)
    iu_dict[item_id].append(user_id)
print("len(ui_dict)=====", len(ui_dict))

ex_data = open('../data/' + dataName + '/reviews_' + dataName + '.txt', "w")
for line_dict in reviews_data:
    user_id = line_dict["reviewerID"]
    item_id = line_dict["asin"]
    ui_num = len(ui_dict[user_id])
    iu_num = len(iu_dict[item_id])
    if ui_num >= num_ui_link and iu_num >= num_iu_link:
        ex_data.writelines(str(line_dict) + "\n")
        iu_dict[item_id].append(user_id)
print("len(iu_dict)=====", len(iu_dict))
ex_data.close()

ex_data = open('../data/' + dataName + '/meta_' + dataName + '.txt', "w")
for line_dict in meta_data:
    item_id = line_dict["asin"]
    if item_id in iu_dict.keys():
        iu_num = len(iu_dict[item_id])
        if iu_num >= num_iu_link:
            ex_data.writelines(str(line_dict) + "\n")
ex_data.close()
