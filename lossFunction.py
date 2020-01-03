# !/usr/bin/python3
# 2019.8.12
# Author Zhang Yihao @NUS

from keras import backend as K

def hm_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))