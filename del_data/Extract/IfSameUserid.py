#!/usr/bin/python3
# 2019.8.12
#Author Zhang Yihao @NUS

import json

data = []
with open('../inputdata/reviews_Baby_10000.json') as f:
    for line in f:
        #linedict = json.dumps(line)
        data.append(json.loads(line))
    f.close()

userdict = {}
unum = 1

for ui in data:
    if (ui["reviewerID"] not in userdict):
        userdict[ui["reviewerID"]] = unum
    else:
        userdict[ui["reviewerID"]] +=1

print (sorted(userdict.items(), reverse= True, key=lambda d: d[1])[0:1000])
"""
#来一个根据value排序的，先把item的key和value交换位置放入一个list中，再根据list每个元素的第一个值，即原来的value值，排序：
backitems = ""
items = userdict.items()
backitems = [[v[1],v[0]] for v in items]
backitems.sort(reverse=True)

print(backitems)
"""


