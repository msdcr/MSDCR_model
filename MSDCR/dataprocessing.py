import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init as init
import pickle
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from tqdm import tqdm
import math
from torch.utils.data import Dataset,DataLoader
from random import choice
from random import sample
import concurrent.futures
import torch.nn.functional as F

#  encode all interactive data as vectors
def user_item_interaction():
    # build the Movie-User interaction matrix
    df_mov = pd.read_csv('../data_graph/item/movie.csv')
    users_mov = df_mov['UserID'].max() + 1
    items_mov = df_mov['itemID'].max() + 1
    #print(users_mov, items_mov)
    occurences_mov = csr_matrix((users_mov, items_mov), dtype='int8')
    def set_occurences_mov(user, item):
        occurences_mov[user, item] = 1
    df_mov.apply(lambda row: set_occurences_mov(row['UserID'], row['itemID']), axis=1)

    # build the Book-User interaction matrix
    df_bok = pd.read_csv('../data_graph/item/book.csv')
    users_bok = df_bok['UserID'].max() + 1
    items_bok = df_bok['itemID'].max() + 1
    occurences_bok = csr_matrix((users_bok, items_bok), dtype='int8')
    def set_occurences_bok(user, item):
        occurences_bok[user, item] = 1
    df_bok.apply(lambda row: set_occurences_bok(row['UserID'], row['itemID']), axis=1)

    return occurences_mov, occurences_bok

# code user as a one-hot vector
def user_to_onehot():
    df = pd.read_csv('../data_graph/bok_msc/user.csv', sep='\n')
    df_user = df['UserID']
    data_user = df_user.values.astype(int)
    with open('../data_graph/bok_msc/' + 'user_index.pkl', 'wb') as f_t:
        pickle.dump(data_user, f_t, pickle.HIGHEST_PROTOCOL)
    return data_user

def update(t1, t2, dropna=False):
    return t1.map(t2).dropna() if dropna else t1.map(t2).fillna(t1)


# code the Movie domain item as a multi-hot vector
def mov_fea_value():
    df = pd.read_csv('../data/mov_bok/movie.csv', sep='|')
    df_mov = pd.read_csv('../data/mov_bok/movie_feature_id.csv', sep=',')
    df1 = df.drop(['itemID'], axis=1)
    df1['director'] = df1['director'].str.strip()
    update_d = update(df1.director, df_mov.set_index('itemID').itemIdx)
    df1['director'] = update_d
    df1['writer1'] = df1['writer1'].str.strip()
    update_w1 = update(df1.writer1, df_mov.set_index('itemID').itemIdx)
    df1['writer1'] = update_w1
    df1['writer2'] = df1['writer2'].str.strip()
    update_w2 = update(df1.writer2, df_mov.set_index('itemID').itemIdx)
    df1['writer2'] = update_w2
    df1['actor1'].astype(str)
    df1['actor1'] = df1['actor1'].str.strip()
    update_a1 = update(df1.actor1, df_mov.set_index('itemID').itemIdx)
    df1['actor1'] = update_a1
    df1['actor2'] = df1['actor2'].values.astype(str)
    update_a2 = update(df1.actor2, df_mov.set_index('itemID').itemIdx)
    df1['actor2'] = update_a2
    df1['actor3'] = df1['actor3'].values.astype(str)
    update_a3 = update(df1.actor3, df_mov.set_index('itemID').itemIdx)
    df1['actor3'] = update_a3
    df1['actor4'] = df1['actor4'].values.astype(str)
    update_a4 = update(df1.actor4, df_mov.set_index('itemID').itemIdx)
    df1['actor4'] = update_a4
    df1['actor5'] = df1['actor5'].values.astype(str)
    update_a5 = update(df1.actor5, df_mov.set_index('itemID').itemIdx)
    df1['actor5'] = update_a5
    df1['actor6'] = df1['actor6'].values.astype(str)
    update_a6 = update(df1.actor6, df_mov.set_index('itemID').itemIdx)
    df1['actor6'] = update_a6
    df1['actor7'] = df1['actor7'].values.astype(str)
    update_a7 = update(df1.actor7, df_mov.set_index('itemID').itemIdx)
    df1['actor7'] = update_a7
    df1['actor8'] = df1['actor8'].values.astype(str)
    update_a8 = update(df1.actor8, df_mov.set_index('itemID').itemIdx)
    df1['actor8'] = update_a8
    df1['type1'] = df1['type1'].values.astype(str)
    update_t1 = update(df1.type1, df_mov.set_index('itemID').itemIdx)
    df1['type1'] = update_t1
    df1['type2'] = df1['type2'].values.astype(str)
    update_t2 = update(df1.type2, df_mov.set_index('itemID').itemIdx)
    df1['type2'] = update_t2
    df1['country'] = df1['country'].values.astype(str)
    update_c = update(df1.country, df_mov.set_index('itemID').itemIdx)
    df1['country'] = update_c
    df1['language'] = df1['language'].values.astype(str)
    update_l = update(df1.language, df_mov.set_index('itemID').itemIdx)
    df1['language'] = update_l


    data = df1.values.astype(int)
    data=np.maximum(data, 0)
    with open('../data/mov_bok/' + 'movie_index.pkl', 'wb') as f_t:
        pickle.dump(data, f_t, pickle.HIGHEST_PROTOCOL)

    return data

# Code the Book domain item as a multi-hot vector
def bok_fea_value():
    df = pd.read_csv('../data_graph/bok_msc/book.csv', sep='|')
    df_bok = pd.read_csv('../data_graph/bok_msc/book_feature_id.csv', sep=',')
    df1 = df.drop(['itemID'], axis=1)
    df1['writer'] = df1['writer'].str.strip()
    update_w = update(df1.writer, df_bok.set_index('itemID').itemIdx)
    df1['writer'] = update_w
    df1['publish'] = df1['publish'].str.strip()
    update_p = update(df1.publish, df_bok.set_index('itemID').itemIdx)
    df1['publish'] = update_p
    df1['translator'] = df1['translator'].str.strip()
    update_t = update(df1.translator, df_bok.set_index('itemID').itemIdx)
    df1['translator'] = update_t
    df1['score'] = df1['score'].str.strip()
    update_s = update(df1.score, df_bok.set_index('itemID').itemIdx)
    df1['score'] = update_s
    data = df1.values.astype(int)
    # data_graph = df1.values.astype(int)
    with open('../data/bok_msc/' + 'book_index.pkl', 'wb') as f_t:
        pickle.dump(data, f_t, pickle.HIGHEST_PROTOCOL)

    return data


# divide the training set, validation set, and test set
def train_validate_test_split():
    user_all = pd.read_csv('../data_graph/item/user.csv', sep='\n')['UserID'].tolist()
    user_mov, user_bok = user_item_interaction()
    item_all_mov = pd.read_csv('../data_graph/item/mov_item.csv', sep='|')['itemID'].tolist()  # 得到所有item的ID集合
    item_all_bok = pd.read_csv('../data_graph/item/bok_item.csv', sep='|')['itemID'].tolist()

    item_train_sample = []
    item_validate_sample = []
    item_test_sample = []

    for u_id in user_all:
        u_mov, u_bok = [], []
        u_mov_sample, u_bok_sample = [], []
        u_sample = []

        indices_u_mov = np.nonzero(user_mov[u_id])
        u_mov.append(indices_u_mov[1:])
        item_all_n_mov = list(set(item_all_mov) - set(indices_u_mov[1]))
        for i in u_mov[0][0]:
            u_hist = np.delete(u_mov[0][0], np.where(u_mov[0][0] == i))
            u_mov_sample.append([i, u_hist])

        indices_u_bok = np.nonzero(user_bok[u_id])
        u_bok.append(indices_u_bok[1:])
        item_all_n_bok = list(set(item_all_bok) - set(indices_u_bok[1]))
        for i in u_bok[0][0]:
            u_hist = np.delete(u_bok[0][0], np.where(u_bok[0][0] == i))
            u_bok_sample.append([i, u_hist])

        # organize the complete training data sample
        for i in u_mov_sample:
            u_sample_i = []
            u_b = choice(u_bok_sample)
            u_sample_i.append([u_id, i, u_b])
            if u_sample_i not in u_sample:
                u_sample.append([u_sample_i])

        for i in u_bok_sample:
            u_sample_i = []
            u_m = choice(u_mov_sample)
            u_sample_i.append([u_id, u_m, i])
            if u_sample_i not in u_sample:
                u_sample.append([u_sample_i])

        u_train_validate = sample(u_sample, math.floor(0.8 * len(u_sample)))
        u_test = [i for i in u_sample if i not in u_train_validate]
        u_train = sample(u_train_validate, math.floor(0.9 * len(u_train_validate)))
        u_validate = [i for i in u_train_validate if i not in u_train]

        for u_p in u_train:
            j_m = choice(item_all_n_mov)
            j_b = choice(item_all_n_bok)
            u_n = [u_id, j_m, u_p[0][0][1][1], j_b, u_p[0][0][2][1]]
            item_train_sample.append([u_p, u_n])


        for u_p in u_validate:
            u_n = []
            for i in range(99):  # we're going to take 99 negative samples for every positive sample
                j_m = choice(item_all_n_mov)
                j_b = choice(item_all_n_bok)
                u_n.append([u_id, j_m, u_p[0][0][1][1], j_b, u_p[0][0][2][1]])
            item_validate_sample.append([u_p, u_n])

        for u_p in u_test:
            u_n = []
            for i in range(99):
                j_m = choice(item_all_n_mov)
                j_b = choice(item_all_n_bok)
                u_n.append([u_id, j_m, u_p[0][0][1][1], j_b, u_p[0][0][2][1]])
            item_test_sample.append([u_p, u_n])

    # Store the divided training set, validation set, and test set as files
    with open('../data_graph/item/' + 'train.pkl', 'wb') as f_t:
        pickle.dump(item_train_sample, f_t, pickle.HIGHEST_PROTOCOL)
    with open('../data_graph/item/' + 'validate.pkl', 'wb') as f_v:
        pickle.dump(item_validate_sample, f_v, pickle.HIGHEST_PROTOCOL)
    with open('../data_graph/item/' + 'test.pkl', 'wb') as f_tt:
        pickle.dump(item_test_sample, f_tt, pickle.HIGHEST_PROTOCOL)

# The item index of each user interaction is indexed through the user ID,
# and the item multi-hot vector crossed by the user is retrieved through the item index
def item_Feature(data):
    with open('../data/mov_bok/user_index.pkl', 'rb') as f:
        users = pickle.load(f)
    with open('../data/mov_bok/movie_index.pkl', 'rb') as f:
        data_mov = pickle.load(f)
    with open('../data/mov_bok/book_index.pkl', 'rb') as f:
        data_bok = pickle.load(f)
    u_i_sample_feature = []
    for j in range(len(data)):
        # The data features of each part of the sample were retrieved
        u_feature = users[data[j][0]]
        #  Take the Movie item sample feature in training set
        mov_i = data_mov[data[j][1]]
        mov_hist = data_mov.take(data[j][2], axis=0)

        #  Take the Book item sample feature in training set
        bok_i = data_bok[data[j][3]]
        bok_hist = data_bok.take(data[j][4], axis=0)
        mov_n = data_mov[data[j][6]]
        bok_n = data_bok[data[j][8]]

        u_i_sample_feature.append([u_feature, mov_i, mov_hist, bok_i, bok_hist , u_feature, mov_n, mov_hist, bok_n, bok_hist])


    return u_i_sample_feature

def train_sample(j):
    # Reorganize the multi_hot encoding of the validation set and test set data input format
    sample = []
    # for i in range(len(train)):
    for i in j:
        u_positive = [i[0][0][0][0], i[0][0][0][1][0], i[0][0][0][1][1], i[0][0][0][2][0],
                      i[0][0][0][2][1]]

        u_negtive = i[1]

        sample.append([u_positive, u_negtive])

    return sample


def validate_test_sample(m):
    with open('../data/mov_bok/user_index.pkl', 'rb') as f:
        users = pickle.load(f)
    with open('../data/mov_bok/movie_index.pkl', 'rb') as f:
        data_mov = pickle.load(f)
    with open('../data/mov_bok/book_index.pkl', 'rb') as f:
        data_bok = pickle.load(f)
    # Reorganize the multi_hot encoding of the validation set and test set data input format
    sample = []

    u_feature = users[m[0][0][0][0]]
    #  Take the Movie's training samples and divided into some parts
    mov_i = data_mov[m[0][0][0][1][0]]
    mov_hist = data_mov.take(m[0][0][0][1][1], axis=0)

    #  Take the Book's training samples and divided into some parts
    bok_i = data_bok[m[0][0][0][2][0]]  # the positive item to prediction
    bok_hist = data_bok.take(m[0][0][0][2][1], axis=0)  # users interaction history
    u_pos = [u_feature, mov_i, mov_hist, bok_i, bok_hist]   # a whole positive training sample consist of positive item of two domains
    u_neg = []
    for j in m[1]:
        u_feature = users[j[0]]
        #  Take the Movie item sample feature in validate/test set
        mov_i = data_mov[j[1]]
        mov_hist = data_mov.take(j[2], axis=0)

        #  Take the Book item sample feature in validate/test set
        bok_i = data_bok[j[3]]
        bok_hist = data_bok.take(j[4], axis=0)
        u_neg.append([u_feature, mov_i, mov_hist, bok_i, bok_hist])
    sample.append([u_pos, u_neg])    # the training sample consist of a positive one and a negative one
    return sample

# Padding data
def processing_data(train):

    data_train = train_sample(train)
    length_mov, length_bok = 10, 10
    # d_m = len(data_train[0][0][1])
    # d_b = len(data_train[0][0][3])
    X_train =[]
    for j in range(len(data_train)):
        U_P = data_train[j][0][0]
        I_mov_P = data_train[j][0][1]
        if len(data_train[j][0][2]) < length_mov:
            mov = data_train[j][0][2].tolist()
            mov.extend([10] * (length_mov - len(data_train[j][0][2])))
            data_train[j][0][2] = np.array(mov)
        M_mov_P = data_train[j][0][2]
        I_bok_P = data_train[j][0][3]
        if len(data_train[j][0][4]) < length_bok:
            bok = data_train[j][0][4].tolist()
            bok.extend([10] * (length_mov - len(data_train[j][0][4])))
            data_train[j][0][4] = np.array(bok)

        B_bok_P = data_train[j][0][4]

        # Negative Sample：
        I_mov_N = data_train[j][1][1]
        I_bok_N = data_train[j][1][3]
        X_train.append([U_P, I_mov_P, M_mov_P, I_bok_P, B_bok_P, U_P, I_mov_N, M_mov_P, I_bok_N, B_bok_P])

    # Store the divided training set, validation set, and test set as files

    with open('../data_graph/item/' + 'train11.pkl', 'wb') as f_t:
        pickle.dump(X_train, f_t, pickle.HIGHEST_PROTOCOL)
    print('This process has done！！！')











