#!/usr/bin/python3
import json

num = 100626  # uiRating_num中数据条数
usermax = 531889
itemmax = 71316
datatrain = open("../../ReSys_HM/data/Baby/Baby.train.txt","w")
datatest = open("../../ReSys_HM/data/Baby/Baby.test.txt","w")

with open('../outputdata/Baby/[3]uiRating_num_filtering.txt') as f:
    for i in range(int(num/2)):
        line1 = f.readline()
        split1 = line1.split('\t')
        if (int(split1[0])<= usermax and int(split1[1])<= itemmax):
            datatrain.writelines(str(line1))
        else:
            print('==========',split1[0])
        line2 = f.readline()
        split2 = line2.split('\t')
        if (int(split2[0]) <= usermax and int(split2[1]) <= itemmax):
            datatest.writelines(str(line2))
        else:
            print('##########',split1[0])
    datatrain.close()
    datatest.close()