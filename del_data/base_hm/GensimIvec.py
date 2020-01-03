#!/usr/bin/python3
# 2019.8.15
# Author Zhang Yihao @NUS
import warnings
import gensim
from gensim.models.doc2vec import Doc2Vec

TaggededDocument = gensim.models.doc2vec.TaggedDocument
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')


class DoctoVec:
    def get_data(dict_d):
        x_train = []
        count = 0
        for (i, text) in dict_d.items():
            word_list = text.split(' ')
            l = len(word_list)
            word_list[l - 1] = word_list[l - 1].strip()
            document = TaggededDocument(word_list, tags=[i])
            x_train.append(document)
            count += 1
        return x_train, count

    def train(dataName, x_train, vector_size, epoch_num):
        model_dm = Doc2Vec(x_train, min_count=1, window=3, vector_size=vector_size, sample=1e-3, negative=5, workers=4)
        model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=epoch_num)
        model_dm.save('../data/' + dataName + '/Model_item_' + str(vector_size))
        return model_dm

    # save vectors
    def saveVector(dataName, model_dm, v_size, count):
        out = open('../data/' + dataName + '/' + dataName + '.' + str(v_size) + '.item', "w", encoding='utf-8')
        for num in range(0, count):
            doc_vec = model_dm.docvecs[num]
            vec_list = str(num) + ","
            for i_doc in doc_vec:
                vec_list = vec_list + str(i_doc) + ","
            out.writelines(vec_list[:-1] + "\n")
        out.close()
