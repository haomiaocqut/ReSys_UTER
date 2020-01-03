#!/usr/bin/python3
import json

num = 10000
data = open("../outputdata/Sports_and_Outdoors/10000.txt","w")

with open('../outputdata/Sports_and_Outdoors/[5]user_vector.txt') as f:
    for i in range(num):
        line = f.readline()
        data.writelines(str(line))
    data.close()