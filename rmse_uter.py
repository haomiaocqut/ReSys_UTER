#!/usr/bin/python3
# 2019.8.12
# Author Zhang Yihao @NUS

import numpy as np
import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Embedding, Input, Dense, Flatten, Concatenate, Multiply, Lambda, Reshape
from keras.layers.core import Dropout
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from keras.regularizers import l2
from evaluate import eval_mae_rmse
from time import time
import argparse
from dataset import Dataset
from lossFunction import hm_mean_squared_error
from lookahead import Lookahead

# ============Arguments ==================
# --k 和--num_factors 两个参数必须相等
factor = 50
dataName = 'Pet'
lr_rate = 0.0001
Num_epochs = 300


def parse_args():
    parser = argparse.ArgumentParser(description="Run ReSys_UTER.")
    parser.add_argument('--path', nargs='?', default='data/' + dataName + '/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default=dataName,
                        help='Choose a dataset.')
    parser.add_argument('--k', type=int, default=factor,
                        help='Number of latent topics in represnetation')
    parser.add_argument('--activation_function', nargs='?', default='relu',
                        help='activation functions')
    parser.add_argument('--epochs', type=int, default=Num_epochs,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=factor,
                        help='Embedding size.')
    parser.add_argument('--regs', nargs='?', default='[0,0]',
                        help="Regularization for user and item embeddings.")
    parser.add_argument('--lr', type=float, default=lr_rate,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    return parser.parse_args()


def init_normal(shape):
    return K.random_normal(shape, mean=0, stddev=0.01, seed=None)


def user_Sentiment(user_latent, user_sent):
    latent_size = user_latent.shape[1].value

    inputs = user_sent
    layer = Dense(latent_size,
                  activation='relu',
                  kernel_initializer='glorot_normal',
                  kernel_regularizer=l2(0.001),
                  name='user_attention_layer')(inputs)
    sent = Lambda(lambda x: K.softmax(x), name='user_Sentiment_softmax')(layer)
    output = Multiply()([user_latent, sent])
    return output


def item_Content(item_latent, item_cont):
    latent_size = item_latent.shape[1].value

    inputs = item_cont
    layer = Dense(latent_size,
                  activation='relu',
                  kernel_initializer='glorot_normal',
                  kernel_regularizer=l2(0.001),
                  name='item_attention_layer')(inputs)
    cont = Lambda(lambda x: K.softmax(x), name='item_Content_softmax')(layer)
    output = Multiply()([item_latent, cont])
    return output


def get_model(num_users, num_items, k, latent_dim, regs=[0, 0]):
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    user_sent = Input(shape=(k,), dtype='float32', name='user_sentiment')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')
    item_cont = Input(shape=(k,), dtype='float32', name='item_content')

    Embedding_User = Embedding(input_dim=num_users,
                               input_length=1,
                               output_dim=latent_dim,
                               embeddings_initializer=init_normal,
                               embeddings_regularizer=l2(regs[0]),
                               name='user_embedding')

    Embedding_Item = Embedding(input_dim=num_items,
                               input_length=1,
                               output_dim=latent_dim,
                               embeddings_initializer=init_normal,
                               embeddings_regularizer=l2(regs[0]),
                               name='item_embedding')

    # Crucial to flatten an embedding vector!
    user_latent = Reshape((latent_dim,))(Flatten()(Embedding_User(user_input)))
    item_latent = Reshape((latent_dim,))(Flatten()(Embedding_Item(item_input)))
    user_latent_atten = user_Sentiment(user_latent, user_sent)
    item_latent_atten = item_Content(item_latent, item_cont)

    user_latent = Dense(latent_dim, kernel_initializer='glorot_normal', activation='relu')(user_latent_atten)
    item_latent = Dense(latent_dim, kernel_initializer='glorot_normal', activation='relu')(item_latent_atten)
    # review-based attention calculation
    vec = Multiply()([user_latent, item_latent])
    user_item_concat = Concatenate()([user_sent, item_cont, user_latent, item_latent])
    att = Dense(latent_dim, kernel_initializer='random_uniform', activation='softmax')(user_item_concat)

    # Element-wise product of user and item embeddings 
    predict_vec = Multiply()([vec, att])

    # Final prediction layer
    prediction = Dense(latent_dim, kernel_initializer='glorot_normal', activation='relu')(predict_vec)
    prediction = Dropout(0.5)(prediction)
    prediction = Dense(1, kernel_initializer='glorot_normal', name='prediction')(prediction)

    uter_model = Model(inputs=[user_input, user_sent, item_input, item_cont], outputs=prediction)
    return uter_model


def get_train_instances(train, user_review_fea, item_review_fea):
    user_input, user_fea, item_input, item_fea, labels = [], [], [], [], []
    num_users = train.shape[0]
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        user_fea.append(user_review_fea[u])
        item_input.append(i)
        item_fea.append(item_review_fea[i])
        label = train[u, i]
        labels.append(label)
    # one_hot_labels = keras.utils.to_categorical(labels, num_classes=5)
    return np.array(user_input), np.array(user_fea, dtype='float32'), np.array(item_input),\
           np.array(item_fea, dtype='float32'), np.array(labels)


if __name__ == '__main__':
    args = parse_args()
    num_factors = args.num_factors
    k = args.k
    regs = eval(args.regs)
    learner = args.learner
    learning_rate = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    verbose = args.verbose
    activation_function = args.activation_function

    evaluation_threads = 1  # mp.cpu_count()
    print("ReSys_HM arguments: %s" % args)
    # model_out_file = 'Pretrain/%sNumofTopic_%d_GMF_%d_%d.h5' %(args.dataset, k, num_factors, time())

    # Loading data
    t1 = time()

    dataset = Dataset(args.path + args.dataset, k)
    train, user_review_fea, item_review_fea, testRatings = dataset.trainMatrix, dataset.user_review_fea, \
                                                           dataset.item_review_fea, dataset.testRatings
    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
          % (time() - t1, num_users, num_items, train.nnz, len(testRatings)))

    # Build model
    model = get_model(num_users, num_items, k, num_factors, regs)
    if learner.lower() == "adagrad":
        model.compile(optimizer=Adagrad(lr=learning_rate), loss="mean_squared_error")
        lookahead = Lookahead(k=5, alpha=0.5)  # Initialize Lookahead
        lookahead.inject(model)  # add into model
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss="mean_squared_error")
        lookahead = Lookahead(k=5, alpha=0.5)
        lookahead.inject(model)
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss="mean_squared_error")
        lookahead = Lookahead(k=5, alpha=0.5)
        lookahead.inject(model)
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss="mean_squared_error")
        lookahead = Lookahead(k=5, alpha=0.5)
        lookahead.inject(model)
    # print(model.summary())

    # Init performance
    t1 = time()
    (mae, rmse) = eval_mae_rmse(model, testRatings, user_review_fea, item_review_fea)
    print('dataName = %s , factor = %d, lr_rate = %.5f' %(dataName, factor, lr_rate))
    print('Init: MAE = %.4f, RMSE = %.4f\t [%.1f s]' % (mae, rmse, time() - t1))

    # Train model
    best_mae, best_rmse, best_iter = mae, rmse, -1

    for epoch in range(epochs):
        t1 = time()
        # Generate training instances
        user_input, user_fea, item_input, item_fea, labels = get_train_instances(train, user_review_fea,
                                                                                 item_review_fea)
        # Training the model
        hist = model.fit([user_input, user_fea, item_input, item_fea],  # input
                         labels,  # labels
                         batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
        t2 = time()

        # Evaluation the model
        if epoch % verbose == 0:
            (mae, rmse) = eval_mae_rmse(model, testRatings, user_review_fea, item_review_fea)
            loss = hist.history['loss'][0]
            print('Iteration %d [%.1f s]: mae = %.3f, rmse = %.3f, loss = %.3f [%.1f s]'
                  % (epoch, t2 - t1, mae, rmse, loss, time() - t2))
            if rmse < best_rmse:
                best_mae, best_rmse, best_iter = mae, rmse, epoch

    print("End. Best Iteration %d:  mae = %.4f, rmse = %.4f. " % (best_iter, best_mae, best_rmse))
    outFile = 'results/ancf' + '.result'
    f = open(outFile, 'a')
    f.write(args.dataset + '\t' + activation_function + "\t" + str(num_factors) + '\t' + str(best_mae) + '\t' + str(
        best_rmse) + '\n')
    f.close()
