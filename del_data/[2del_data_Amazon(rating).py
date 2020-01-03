#!/usr/bin/python3
# 2019.12.24
# Author Zhang Yihao @NUS

import json
import numpy as np
from collections import defaultdict

dataName = "Video"

split_ratio = 4  # train_data与test_data的比例
iu_dict = defaultdict(list)
ui_dict = defaultdict(list)
ui_neg_dict = defaultdict(list)
reviews_data = []
reviews_train_data = []
reviews_test_data = []
train_list = []
test_list = []
len_ui_dict = {}
global max_num_item  # 最大items编号

with open('../data/' + dataName + '/reviews_' + dataName + '.txt') as f:
    for line in f:
        line_dict = json.dumps(eval(line))
        reviews_data.append(json.loads(line_dict))
    f.close()

'''建立ID和Num的对应关系 '''
asin2itemNum = {}
reviewerID2userNum = {}
num = 0
for ui in reviews_data:
    if ui["asin"] not in asin2itemNum:
        asin2itemNum[ui["asin"]] = num
        num += 1
num = 0
for uu in reviews_data:
    if uu["reviewerID"] not in reviewerID2userNum:
        reviewerID2userNum[uu["reviewerID"]] = num
        num += 1

# loading_user_items
max_num_item = 1
max_num_user = 1
for d in reviews_data:
    user_id = reviewerID2userNum.get(d["reviewerID"])
    item_id = asin2itemNum.get(d["asin"])
    if user_id not in ui_dict:
        ui_dict[user_id] = list()
        if int(user_id) > max_num_user:
            max_num_user = int(user_id)
    ui_dict[user_id].append(item_id)
    if item_id not in iu_dict:
        iu_dict[item_id] = list()
        if int(item_id) > max_num_item:
            max_num_item = int(item_id)
    iu_dict[item_id].append(user_id)
for l_d in reviews_data:
    u_id = reviewerID2userNum.get(l_d["reviewerID"])
    len_ui_dict[u_id] = len(ui_dict.get(u_id))
print("max_num_user = %d, max_num_item = %d" % (max_num_user, max_num_item))

# 划分reviews_train_data和reviews_test_data
num = 0
for l_d in reviews_data:
    if num % (split_ratio + 1) < split_ratio:
        reviews_train_data.append(l_d)
        num += 1
    else:
        reviews_test_data.append(l_d)
        num += 1


def construct_all_data():
    all_data = open('../data/' + dataName + '/' + dataName + ".rating", "w")
    for l_data in reviews_data:
        user_id = reviewerID2userNum.get(l_data["reviewerID"])
        item_id = asin2itemNum.get(l_data["asin"])
        rating_score = int(l_data["overall"])
        str_line = str(user_id) + "\t" + str(item_id) + "\t" + str(rating_score)
        all_data.writelines(str_line + "\n")
    all_data.close()


def construct_train_test_data():
    train_data = open('../data/' + dataName + '/' + dataName + ".train.rating", "w")
    test_data = open('../data/' + dataName + '/' + dataName + ".test.rating", "w")
    for l_train in reviews_train_data:
        user_id = reviewerID2userNum.get(l_train["reviewerID"])
        item_id = asin2itemNum.get(l_train["asin"])
        rating_score = int(l_train["overall"])
        str_line = str(user_id) + "\t" + str(item_id) + "\t" + str(rating_score)
        train_data.writelines(str_line + "\n")
    train_data.close()
    # 写入test_data
    for l_test in reviews_test_data:
        user_id = reviewerID2userNum.get(l_test["reviewerID"])
        item_id = asin2itemNum.get(l_test["asin"])
        rating_score = int(l_test["overall"])
        str_line = str(user_id) + "\t" + str(item_id) + "\t" + str(rating_score)
        test_data.writelines(str_line + "\n")
        test_list.append(str_line)
    test_data.close()


def construct_test_negative(num_negative, max_num_item):
    test_neg = open('../data/' + dataName + '/' + dataName + ".test.negative", "w")
    for t_line in test_list:
        user_id = t_line.split("\t")[0]
        item_id = t_line.split("\t")[1]
        str_pos_list = ui_dict[int(user_id)]
        str_pos_neg = "(" + user_id + "," + item_id + ")"
        for _ in range(num_negative):
            neg_item = np.random.randint(max_num_item)
            while str(neg_item) in str_pos_list:
                neg_item = np.random.randint(max_num_item)
            str_pos_neg = str_pos_neg + "\t" + str(neg_item)
        test_neg.writelines(str_pos_neg + "\n")
    test_neg.close()
    return "test_negative successful!"


if __name__ == '__main__':
    #train_neg_num = 4
    #num_negatives = 100  # 构建num_negatives的个数
    #construct_all_data()
    construct_train_test_data()
    #construct_test_negative(num_negatives, max_num_item)
