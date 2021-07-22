import torch
import numpy as np
import math
from torch.autograd import Variable
import torch.nn as nn
from tqdm import tqdm
from MSDCR.Book_Attention import Attention_book
from MSDCR.Movie_Attention import Attention_movie
from MSDCR.domain_classfier import Domain_classfier, Domain_classfier_specific
from MSDCR.Feature_extraction import Feature_extraction,Domain_enhancement,Feature_extra_specific
from MSDCR.NCF import Movie_NCF,Book_NCF
from MSDCR.dataprocessing import item_Feature,validate_test_sample

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class MTBA(torch.nn.Module):
    def __init__(self, input_mov_col, input_bok_col, latent_dim, num_aspects, num_aspects_specific, attn_dropout, input_dim_user,input_mov_item,input_bok_item, latent_item_dim, batch_size):
        super(MTBA, self).__init__()
        self.batch_size = batch_size
        self.input_mov_item = input_mov_item   # item 初始各个特征取值个数
        self.input_bok_item = input_bok_item
        self.latent_item_dim = latent_item_dim  # item 初始各个特征取值one-hot嵌入维度
        self.attn_dropout = attn_dropout
        self.latent_dim = latent_dim
        self.input_dim_user = input_dim_user
        self.num_aspects = num_aspects
        self.num_aspects_specific = num_aspects_specific
        self.Attention_movie = Attention_movie(self.latent_dim)
        self.Attention_book = Attention_book(self.latent_dim)
        self.dropout = torch.nn.Dropout(self.attn_dropout)
        self.sigmoid = torch.nn.Sigmoid()
        self.Embedding_user = torch.nn.Embedding(self.input_dim_user, self.latent_dim)
        self.Embedding_mov = torch.nn.Embedding(self.input_mov_item, self.latent_item_dim)
        self.Embedding_bok = torch.nn.Embedding(self.input_bok_item, self.latent_item_dim)
        self.input_mov_col = input_mov_col
        self.input_bok_col = input_bok_col
        self.Linear_mov = torch.nn.Linear(self.input_mov_col * self.latent_item_dim,self.latent_dim)
        self.Linear_bok = torch.nn.Linear(self.input_bok_col * self.latent_item_dim, self.latent_dim)
        self.Feature_extraction = Feature_extraction(self.latent_dim, self.num_aspects, self.attn_dropout)
        self.Feature_extra_specific = Feature_extra_specific(self.latent_dim, self.num_aspects_specific, self.attn_dropout)
        self.Domain_classfier = Domain_classfier(self.latent_dim, self.num_aspects)
        self.Domain_classfier_specific = Domain_classfier_specific(self.latent_dim, self.num_aspects_specific)
        self.Domain_enhancement = Domain_enhancement(self.latent_dim, self.num_aspects_specific)
        self.Movie_NCF = Movie_NCF(self.latent_dim, self.num_aspects,self.num_aspects_specific)
        self.Book_NCF = Book_NCF(self.latent_dim, self.num_aspects,self.num_aspects_specific)


    def forward(self,batch_size, u, i_mov, X_movie, i_bok, X_book):
        X_movie = torch.from_numpy(np.array(X_movie)).type(torch.LongTensor).to(device=device)
        X_book = torch.from_numpy(np.array(X_book)).type(torch.LongTensor).to(device=device)
        Z_mov = self.Embedding_mov(X_movie)
        Z_bok = self.Embedding_bok(X_book)
        Z_mov = torch.reshape(Z_mov, (batch_size, -1, self.input_mov_col*self.latent_item_dim))
        Z_bok = torch.reshape(Z_bok, (batch_size, -1, self.input_bok_col * self.latent_item_dim))
        Z_mov = self.Linear_mov(Z_mov)
        Z_bok = self.Linear_bok(Z_bok)
        H_mov = self.Attention_movie(Z_mov, Z_mov, Z_mov)
        H_bok = self.Attention_book(Z_bok, Z_bok, Z_bok)
        u = torch.from_numpy(np.array(u)).type(torch.LongTensor).to(device=device)
        user_embedding = self.Embedding_user(u)
        C_mov = self.Feature_extraction(H_mov, user_embedding)
        C_bok = self.Feature_extraction(H_bok, user_embedding)
        C_mov_specific = self.Feature_extra_specific(H_mov, user_embedding)
        C_bok_specific = self.Feature_extra_specific(H_bok, user_embedding)
        i_mov = torch.from_numpy(i_mov).type(torch.LongTensor).to(device=device)
        i_bok = torch.from_numpy(i_bok).type(torch.LongTensor).to(device=device)
        e_mov_i = self.Embedding_mov(i_mov)
        e_bok_i = self.Embedding_bok(i_bok)
        e_mov_i = torch.reshape(e_mov_i, (batch_size, self.input_mov_col* self.latent_item_dim))
        e_bok_i = torch.reshape(e_bok_i, (batch_size, self.input_bok_col * self.latent_item_dim))
        e_mov_i = self.Linear_mov(e_mov_i)
        e_bok_i = self.Linear_bok(e_bok_i)

        # domains common features extraction
        c_mov_predict, c_bok_predict = self.Domain_classfier(C_mov, C_bok)
        c_mov_specific_predict, c_bok_specific_predict = self.Domain_classfier_specific(C_mov_specific, C_bok_specific)
        A_mov, A_bok = self.Domain_enhancement(C_mov_specific, C_bok_specific)
        # the rating of every domains
        rating_mov_i = self.Movie_NCF(A_mov, C_mov, e_mov_i)
        rating_bok_i = self.Book_NCF(A_bok, C_bok, e_bok_i)

        return c_mov_predict, c_bok_predict, c_mov_specific_predict, c_bok_specific_predict, rating_mov_i, rating_bok_i

def NDCG(y_true, y_score, k):
    if y_true in y_score[:k]:
        y_score = y_score[:k].tolist()
        gain = 2 ** 1 - 1
        discounts = np.log2(y_score.index(y_true) + 2)
        ndcg = (gain / discounts)
        return ndcg
    else:
        ndcg = 0.0
        return ndcg

def Precision(y_true, y_score, k):
    if y_true in y_score[:k]:
        i = 1
        return i
    else:
        i = 0
        return i

class MTBA_runner(object):
    def __init__(self,input_mov_col, input_bok_col,input_dim_user,input_mov_item,input_bok_item,latent_item_dim,config):
        self.config = config
        self.sigmoid = torch.nn.Sigmoid()
        self.MTBA = MTBA(input_mov_col=input_mov_col, input_bok_col=input_bok_col,latent_dim=config['latent_dim'], num_aspects=config['num_aspects'], num_aspects_specific=config['num_aspects_specific'], attn_dropout=config['attn_dropout']
                         ,input_dim_user=input_dim_user,input_mov_item = input_mov_item,input_bok_item = input_bok_item,latent_item_dim=latent_item_dim, batch_size=self.config['batch_size'])
        self.MTBA.to(device)

        self.optPrediction = torch.optim.Adam([{'params': self.MTBA.Movie_NCF.parameters()},
                                       {'params': self.MTBA.Book_NCF.parameters()},
                                       {'params': self.MTBA.Domain_enhancement.parameters()},
                                       {'params': self.MTBA.Attention_movie.parameters()},
                                       {'params': self.MTBA.Attention_book.parameters()},
                                       {'params': self.MTBA.Linear_mov.parameters()},
                                       {'params': self.MTBA.Linear_bok.parameters()},
                                       {'params': self.MTBA.Embedding_user.parameters()},
                                       {'params': self.MTBA.Embedding_mov.parameters()},
                                       {'params': self.MTBA.Embedding_bok.parameters()}], lr=config['lr'],  weight_decay=0.001)

        self.optInvariant_classfier = torch.optim.Adam(self.MTBA.Domain_classfier.parameters(), lr=config['lr'],  weight_decay=0.001)
        self.optInvariant = torch.optim.Adam([{'params': self.MTBA.Feature_extraction.parameters()},
                                       {'params': self.MTBA.Attention_movie.parameters()},
                                       {'params': self.MTBA.Attention_book.parameters()},
                                       {'params': self.MTBA.Linear_mov.parameters()},
                                       {'params': self.MTBA.Linear_bok.parameters()},
                                       {'params': self.MTBA.Embedding_user.parameters()},
                                       {'params': self.MTBA.Embedding_mov.parameters()},
                                       {'params': self.MTBA.Embedding_bok.parameters()}], lr=config['lr'], weight_decay=0.001)

        self.optSpecific_classfier = torch.optim.Adam(self.MTBA.Domain_classfier_specific.parameters(), lr=config['lr'], weight_decay=0.001)
        self.optSpecific = torch.optim.Adam([{'params': self.MTBA.Feature_extra_specific.parameters()},
                                            {'params': self.MTBA.Attention_movie.parameters()},
                                            {'params': self.MTBA.Attention_book.parameters()},
                                             {'params': self.MTBA.Linear_mov.parameters()},
                                             {'params': self.MTBA.Linear_bok.parameters()},
                                            {'params': self.MTBA.Embedding_user.parameters()},
                                            {'params': self.MTBA.Embedding_mov.parameters()},
                                            {'params': self.MTBA.Embedding_bok.parameters()}], lr=config['lr'], weight_decay=0.001)

        self.crit = torch.nn.CrossEntropyLoss()

    def train_single_batch(self, batch_train):
        self.optPrediction.zero_grad()
        self.optInvariant_classfier.zero_grad()
        self.optInvariant.zero_grad()
        self.optSpecific_classfier.zero_grad()
        self.optSpecific.zero_grad()

        batch_train = item_Feature(batch_train)
        batch_size = len(batch_train)
        d_m = len(batch_train[0][1])
        d_b = len(batch_train[0][3])

        U_P, I_mov_P, M_mov_P, I_bok_P, B_bok_P = np.empty([0, 1], dtype=int), np.empty([0, d_m], dtype=int), np.empty([0, d_m], dtype=int), np.empty([0, d_b], dtype=int), np.empty([0, d_b], dtype=int)
        I_mov_N, I_bok_N = np.empty([0, d_m], dtype=int), np.empty([0, d_b], dtype=int)

        for j in range(batch_size):
            U_P = np.concatenate([U_P, batch_train[j][0].reshape(1, -1)], axis=0)
            I_mov_P = np.concatenate([I_mov_P, batch_train[j][1].reshape(1, -1)], axis=0)
            M_mov_P = np.concatenate([M_mov_P, batch_train[j][2]], axis=0)
            I_bok_P = np.concatenate([I_bok_P, batch_train[j][3].reshape(1, -1)], axis=0)
            B_bok_P = np.concatenate([B_bok_P, batch_train[j][4]], axis=0)

            # Negative sample j[1]：
            I_mov_N = np.concatenate([I_mov_N, batch_train[j][6].reshape(1, -1)], axis=0)
            I_bok_N = np.concatenate([I_bok_N, batch_train[j][8].reshape(1, -1)], axis=0)

        # # Positive sample j[0]：
        M_mov_P = np.reshape(M_mov_P, (batch_size, -1, d_m))
        B_bok_P = np.reshape(B_bok_P, (batch_size, -1, d_b))
        c_mov_predict_i, c_bok_predict_i, c_mov_specific_predict_i, c_bok_specific_predict_i, \
        rating_mov_i, rating_bok_i = self.MTBA(batch_size, U_P, I_mov_P, M_mov_P, I_bok_P, B_bok_P)
        # the domain-invariant loss calculation
        loss_mov_i = self.crit(c_mov_predict_i, torch.tensor(np.array([0] * batch_size), dtype=torch.long, device=device))
        loss_bok_i = self.crit(c_bok_predict_i, torch.tensor(np.array([1] * batch_size), dtype=torch.long, device=device))
        # the domain-specific loss calculation
        loss_mov_specific_i = self.crit(c_mov_specific_predict_i, torch.tensor(np.array([0] * batch_size), dtype=torch.long, device=device))
        loss_bok_specific_i = self.crit(c_bok_specific_predict_i, torch.tensor(np.array([1] * batch_size), dtype=torch.long, device=device))

        # # Negative sample j[0]：
        c_mov_predict_j, c_bok_predict_j, c_mov_specific_predict_j, c_bok_specific_predict_j, \
        rating_mov_j, rating_bok_j = self.MTBA(batch_size, U_P, I_mov_N, M_mov_P, I_bok_N, B_bok_P)

        # the domain-invariant loss calculation
        loss_mov_j = self.crit(c_mov_predict_j, torch.tensor(np.array([0] * batch_size), dtype=torch.long, device=device))
        loss_bok_j = self.crit(c_bok_predict_j, torch.tensor(np.array([1] * batch_size), dtype=torch.long, device=device))
        loss_classfier = loss_mov_i + loss_bok_i + loss_mov_j + loss_bok_j

        # the domain-specific loss calculation
        loss_mov_specific_j = self.crit(c_mov_specific_predict_j, torch.tensor(np.array([0] * batch_size), dtype=torch.long, device=device))
        loss_bok_specific_j = self.crit(c_bok_specific_predict_j, torch.tensor(np.array([1] * batch_size), dtype=torch.long, device=device))
        loss_classfier_specific = loss_mov_specific_i + loss_bok_specific_i + loss_mov_specific_j + loss_bok_specific_j

        # BPR
        loss_predict = -torch.sum(torch.log(self.sigmoid(rating_mov_i - rating_mov_j) + 1e-24) +
                         torch.log(torch.sigmoid(rating_bok_i - rating_bok_j) + 1e-24))

        loss_classfier_batch = loss_classfier
        loss_classfier_specific_batch = loss_classfier_specific
        loss_predict_batch = loss_predict

        loss_predict_batch.backward(retain_graph=True)

        loss_classfier_batch.backward(retain_graph=True)

        loss_classfier_batch_max = -loss_classfier_batch
        loss_classfier_batch_max.backward(retain_graph=True)

        loss_classfier_specific_batch.backward(retain_graph=True)

        loss_classfier_specific_batch_max = -loss_classfier_specific_batch
        loss_classfier_specific_batch_max.backward()
        self.optPrediction.step()
        self.optInvariant_classfier.step()
        self.optInvariant.step()
        self.optSpecific.step()
        self.optSpecific_classfier.step()



        loss_predict_batch = loss_predict_batch.data.cpu().numpy()
        loss_classfier_batch = loss_classfier_batch.data.cpu().numpy()
        loss_classfier_specific_batch = loss_classfier_specific_batch.data.cpu().numpy()


        return loss_predict_batch, loss_classfier_batch, loss_classfier_specific_batch

    def train_single_epoch(self, train, config, epoch_id):
        self.MTBA.train()
        np.random.shuffle(train)
        num_examples = len(train)
        loss_epoch = 0
        batch_size = config['batch_size']
        for i in tqdm(range(0, num_examples, batch_size)):
            batch_id = (i // batch_size) + 1
            batch_train = train[i: min(i + config['batch_size'], num_examples)]
            # batch_train = item_Feature(batch_train)
            loss_predict_batch, loss_classfier_batch, loss_classfier_specific_batch = self.train_single_batch(batch_train)

            print('\nBatch number:{}  Prediction loss：{}     Domain-invariant loss：{}    Domain-specific loss：{}'.format(batch_id, loss_predict_batch, loss_classfier_batch, loss_classfier_specific_batch))

            loss_epoch = loss_predict_batch + loss_epoch

        loss_epoch_mean = loss_epoch / (num_examples // batch_size + 1)
        print("the {} epoch loss:{}".format(epoch_id+1, loss_epoch_mean))
        with open('../parameters/MTBA_epoch_loss.txt', 'a+') as f:
            f.write("epoch {}    epoch loss:{}".format(epoch_id+1, loss_epoch_mean) +'\n')
        return loss_epoch_mean

    def evaluate(self, data):
        self.MTBA.eval()
        np.random.shuffle(data)
        m_ndcg_all_5, m_ndcg_all_10 = [], []
        b_ndcg_all_5, b_ndcg_all_10 = [], []
        m_prec_all_5, m_prec_all_10 = [], []
        b_prec_all_5, b_prec_all_10 = [], []
        for i in tqdm(range(len(data))):
            u = validate_test_sample(data[i])
            batch_size = 10
            d_m = len(u[0][0][1])
            d_b = len(u[0][0][3])
            # Initializes the empty lists
            U_P, I_mov_P, M_mov_P, I_bok_P, B_bok_P = np.empty([0, 1], dtype=int), np.empty([0, d_m],dtype=int), np.empty(
                [0, d_m], dtype=int), np.empty([0, d_b], dtype=int), np.empty([0, d_b], dtype=int)
            U_P = np.concatenate([U_P, np.array(u[0][0][0]).reshape(1, -1)], axis=0)
            I_mov_P = np.concatenate([I_mov_P, u[0][0][1].reshape(1, -1)], axis=0)
            M_mov_P = np.concatenate([M_mov_P, u[0][0][2]], axis=0)
            I_bok_P = np.concatenate([I_bok_P, u[0][0][3].reshape(1, -1)], axis=0)
            B_bok_P = np.concatenate([B_bok_P, u[0][0][4]], axis=0)
            for k in u[0][1]:
                U_P = np.concatenate([U_P, np.array(k[0]).reshape(1, -1)], axis=0)
                I_mov_P = np.concatenate([I_mov_P, k[1].reshape(1, -1)], axis=0)
                M_mov_P = np.concatenate([M_mov_P, k[2]], axis=0)
                I_bok_P = np.concatenate([I_bok_P, k[3].reshape(1, -1)], axis=0)
                B_bok_P = np.concatenate([B_bok_P, k[4]], axis=0)

            M_mov_P = np.reshape(M_mov_P, (batch_size, -1, d_m))
            B_bok_P = np.reshape(B_bok_P, (batch_size, -1, d_b))

            c_mov_predict_i, c_bok_predict_i, c_mov_specific_predict_i, c_bok_specific_predict_i, \
            rating_mov_i, rating_bok_i = self.MTBA(batch_size, U_P, I_mov_P, M_mov_P, I_bok_P, B_bok_P)
            rating_mov_i = rating_mov_i.squeeze(dim=1).detach().cpu().numpy()
            rating_bok_i = rating_bok_i.squeeze(dim=1).detach().cpu().numpy()

            m_, m_Rank = torch.topk(torch.tensor(rating_mov_i), len(rating_mov_i), dim=0, largest=True, sorted=True)
            b_, b_Rank = torch.topk(torch.tensor(rating_bok_i), len(rating_bok_i), dim=0, largest=True, sorted=True)
            # NDCG@5 、@10


            m_ndcg_5 = NDCG(0, m_Rank, 5)
            b_ndcg_5 = NDCG(0, b_Rank, 5)

            m_ndcg_all_5.append(m_ndcg_5)
            b_ndcg_all_5.append(b_ndcg_5)

            m_ndcg_10 = NDCG(0, m_Rank, 10)
            b_ndcg_10 = NDCG(0, b_Rank, 10)

            m_ndcg_all_10.append(m_ndcg_10)
            b_ndcg_all_10.append(b_ndcg_10)

            # precision @5、 @10
            m_prec_5 = Precision(0, m_Rank, 5)
            b_prec_5 = Precision(0, b_Rank, 5)
            m_prec_all_5.append(m_prec_5)
            b_prec_all_5.append(b_prec_5)

            m_prec_10 = Precision(0, m_Rank, 10)
            b_prec_10 = Precision(0, b_Rank, 10)

            m_prec_all_10.append(m_prec_10)
            b_prec_all_10.append(b_prec_10)
        # Take the average of each metrics

        m_average_ndcg_5 = np.mean(m_ndcg_all_5)
        b_average_ndcg_5 = np.mean(b_ndcg_all_5)

        m_average_ndcg_10 = np.mean(m_ndcg_all_10)
        b_average_ndcg_10 = np.mean(b_ndcg_all_10)

        m_average_prec_5 = np.mean(m_prec_all_5)
        b_average_prec_5 = np.mean(b_prec_all_5)

        m_average_prec_10 = np.mean(m_prec_all_10)
        b_average_prec_10 = np.mean(b_prec_all_10)

        print("[Movie evaluate :]  NDCG@5={} , NDCG@10={} , Precision@5={} , Precision@10={}".format(m_average_ndcg_5, m_average_ndcg_10, m_average_prec_5, m_average_prec_10))
        print("[Book evaluate :] NDCG@5={} , NDCG@10={} , Precision@5={} , Precision@10={}".format(b_average_ndcg_5, b_average_ndcg_10, b_average_prec_5, b_average_prec_10))





