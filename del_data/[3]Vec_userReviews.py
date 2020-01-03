#!/usr/bin/python3
import json
from del_data.base_hm.GensimUvec import *

dataName = "Video"
vector_size = 5
epoch_num = 100

reviews_data = []
with open('../data/' + dataName + '/reviews_' + dataName + '.txt') as f:
    for line in f:
        line_dict = json.dumps(eval(line))
        reviews_data.append(json.loads(line_dict))
    f.close()

'''建立ID和Num的对应关系 '''
asin2itemNum = {}
reviewerID2userNum = {}
num = 1
for ui in reviews_data:
    if ui["asin"] not in asin2itemNum:
        asin2itemNum[ui["asin"]] = num
        num += 1
num = 1
for uu in reviews_data:
    if uu["reviewerID"] not in reviewerID2userNum:
        reviewerID2userNum[uu["reviewerID"]] = num
        num += 1


def loading_data():
    # loading the dictionary of UserId_ItemID_Num
    data = []
    dict_d = {}
    with open('../data/' + dataName + '/reviews_' + dataName + '.txt') as f:
        for line in f:
            line_dict = json.dumps(eval(line))
            data.append(json.loads(line_dict))
        for d_item in data:
            k = reviewerID2userNum.get(d_item["reviewerID"])
            print(k)
            if "reviewText" in d_item.keys():
                dict_d[k] = d_item["reviewText"].replace("\n", "")
            else:
                dict_d[k] = ""
            print(dict_d[k])
        return dict_d


if __name__ == '__main__':
    dict_d = loading_data()
    x_train, count = DoctoVec.get_data(dict_d)
    model_dm = DoctoVec.train(dataName, x_train, vector_size, epoch_num)
    DoctoVec.saveVector(dataName, model_dm, vector_size, count)
