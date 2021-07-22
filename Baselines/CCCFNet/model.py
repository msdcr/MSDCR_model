import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from CCCFNet.NCF import CC_NCF_mov,CC_NCF_bok
from CCCFNet.dataprocessing import item_Feature,validate_test_sample

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class CCCFNet(torch.nn.Module):
    def __init__(self, input_mov_col, input_bok_col, latent_dim, attn_dropout, input_mov, input_bok,  input_dim_user, input_mov_item, input_bok_item, batch_size):
        super(CCCFNet, self).__init__()
        self.batch_size = batch_size
        self.input_mov_item = input_mov_item
        self.input_bok_item = input_bok_item
        self.attn_dropout = attn_dropout
        self.latent_dim = latent_dim
        self.input_mov = input_mov
        self.input_bok = input_bok
        self.input_dim_user = input_dim_user
        self.input_mov_col = input_mov_col
        self.input_bok_col = input_bok_col
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()
        self.Embedding_user = torch.nn.Embedding(self.input_dim_user, 2 * self.latent_dim)
        self.Embedding_mov_item = torch.nn.Embedding(self.input_mov_item, self.latent_dim)
        self.Embedding_bok_item = torch.nn.Embedding(self.input_bok_item, self.latent_dim)
        self.Embedding_mov = torch.nn.Embedding(self.input_mov, self.latent_dim)
        self.Embedding_bok = torch.nn.Embedding(self.input_bok, self.latent_dim)

        self.Linear_mov = torch.nn.Linear(self.input_mov_col * self.latent_dim, self.latent_dim)
        self.Linear_bok = torch.nn.Linear(self.input_bok_col * self.latent_dim, self.latent_dim)
        self.CC_NCF_mov = CC_NCF_mov(self.latent_dim)
        self.CC_NCF_bok = CC_NCF_bok(self.latent_dim)

    def forward(self, batch_size, u, mov_id, i_mov, X_movie, bok_id, i_bok, X_book):

        u = torch.from_numpy(np.array(u)).type(torch.LongTensor).to(device=device)
        user_embedding = self.Embedding_user(u)
        mov_id = torch.from_numpy(np.array(mov_id)).type(torch.LongTensor).to(device=device)
        mov_id_embedding = self.Embedding_mov_item(mov_id)
        bok_id = torch.from_numpy(np.array(bok_id)).type(torch.LongTensor).to(device=device)
        bok_id_embedding = self.Embedding_bok_item(bok_id)
        i_mov = torch.from_numpy(i_mov).type(torch.LongTensor).to(device=device)
        i_bok = torch.from_numpy(i_bok).type(torch.LongTensor).to(device=device)
        e_mov_i = self.Embedding_mov_item(i_mov)
        e_bok_i = self.Embedding_bok_item(i_bok)
        e_mov_i = torch.reshape(e_mov_i, (batch_size, -1, self.input_mov_col* self.latent_dim))
        e_bok_i = torch.reshape(e_bok_i, (batch_size, -1, self.input_bok_col * self.latent_dim))
        e_mov_i = self.Linear_mov(e_mov_i)
        e_bok_i = self.Linear_bok(e_bok_i)

        mov_i= torch.cat([mov_id_embedding, e_mov_i], dim=2)
        bok_i = torch.cat([bok_id_embedding, e_bok_i], dim=2)
        # the rating of every domains
        rating_mov_i = self.CC_NCF_mov(user_embedding, mov_i)
        rating_bok_i = self.CC_NCF_bok(user_embedding, bok_i)

        return rating_mov_i, rating_bok_i

def NDCG(y_true, y_score, k):
    if y_true in y_score[:k]:
        y_score = y_score[:k].tolist()
        gain = 2 ** 1 - 1
        discounts = np.log2(y_score.index([y_true]) + 2)
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


class CCCFNet_runner(object):
    def __init__(self,input_mov_col, input_bok_col, input_mov, input_bok, input_dim_user,input_mov_item,input_bok_item,config):
        self.config = config
        self.sigmoid = torch.nn.Sigmoid()
        self.CCCFNet = CCCFNet(input_mov_col=input_mov_col, input_bok_col=input_bok_col,latent_dim=config['latent_dim'], attn_dropout=config['attn_dropout'], input_mov = input_mov, input_bok = input_bok
                         ,input_dim_user=input_dim_user,input_mov_item = input_mov_item,input_bok_item = input_bok_item, batch_size=self.config['batch_size'])
        self.CCCFNet.to(device)

        self.opt_mov = torch.optim.Adam([
                                       {'params': self.CCCFNet.CC_NCF_mov.parameters()},
                                       {'params': self.CCCFNet.Linear_mov.parameters()},
                                       {'params': self.CCCFNet.Embedding_mov.parameters()},
                                       {'params': self.CCCFNet.Embedding_mov_item.parameters()},
                                       {'params': self.CCCFNet.Embedding_user.parameters()}], lr=config['lr'],  weight_decay=0.001)

        self.opt_bok = torch.optim.Adam([{'params': self.CCCFNet.CC_NCF_bok.parameters()},
                                       {'params': self.CCCFNet.Linear_bok.parameters()},
                                       {'params': self.CCCFNet.Embedding_bok.parameters()},
                                       {'params': self.CCCFNet.Embedding_bok_item.parameters()},
                                       {'params': self.CCCFNet.Embedding_user.parameters()}], lr=config['lr'], weight_decay=0.001)

        self.crit = torch.nn.CrossEntropyLoss()

    def train_single_batch(self, batch_train):
        self.opt_mov.zero_grad()
        self.opt_bok.zero_grad()
        batch_train = item_Feature(batch_train)
        batch_size = len(batch_train)
        d_m = len(batch_train[0][2])
        d_b = len(batch_train[0][5])

        U_P, mov_ID_P, I_mov_P, M_mov_P, bok_ID_P, I_bok_P, B_bok_P = np.empty([0, 1], dtype=int), np.empty([0, 1], dtype=int), np.empty([0, d_m], dtype=int), np.empty([0, d_m], dtype=int), np.empty([0, 1], dtype=int), np.empty([0, d_b], dtype=int), np.empty([0, d_b], dtype=int)
        mov_ID_N, I_mov_N, bok_ID_N, I_bok_N = np.empty([0, 1], dtype=int), np.empty([0, d_m], dtype=int), np.empty([0, 1], dtype=int), np.empty([0, d_b], dtype=int)

        for j in range(batch_size):
            U_P = np.concatenate([U_P, batch_train[j][0].reshape(1, -1)], axis=0)
            mov_ID_P = np.concatenate([mov_ID_P, batch_train[j][1].reshape(1, -1)], axis=0)
            I_mov_P = np.concatenate([I_mov_P, batch_train[j][2].reshape(1, -1)], axis=0)
            M_mov_P = np.concatenate([M_mov_P, batch_train[j][3]], axis=0)
            bok_ID_P = np.concatenate([bok_ID_P, batch_train[j][4].reshape(1, -1)], axis=0)
            I_bok_P = np.concatenate([I_bok_P, batch_train[j][5].reshape(1, -1)], axis=0)
            B_bok_P = np.concatenate([B_bok_P, batch_train[j][6]], axis=0)


            mov_ID_N = np.concatenate([mov_ID_N, np.array(batch_train[j][8]).reshape(1, -1)], axis=0)
            I_mov_N = np.concatenate([I_mov_N, batch_train[j][9].reshape(1, -1)], axis=0)
            bok_ID_N = np.concatenate([bok_ID_N, np.array(batch_train[j][11]).reshape(1, -1)], axis=0)
            I_bok_N = np.concatenate([I_bok_N, batch_train[j][12].reshape(1, -1)], axis=0)


        M_mov_P = np.reshape(M_mov_P, (batch_size, -1, d_m))
        B_bok_P = np.reshape(B_bok_P, (batch_size, -1, d_b))

        rating_mov_i, rating_bok_i = self.CCCFNet(batch_size, U_P, mov_ID_P, I_mov_P, M_mov_P, bok_ID_P, I_bok_P, B_bok_P)


        rating_mov_j, rating_bok_j = self.CCCFNet(batch_size, U_P, mov_ID_N, I_mov_N, M_mov_P, bok_ID_N, I_bok_N, B_bok_P)

        # BPR
        loss_predict = -torch.sum(torch.log(self.sigmoid(rating_mov_i - rating_mov_j) + 1e-24) +
                         torch.log(torch.sigmoid(rating_bok_i - rating_bok_j) + 1e-24))

        loss_predict_batch = loss_predict

        loss_predict_batch.backward()

        self.opt_mov.step()
        self.opt_bok.step()

        loss_predict_batch = loss_predict_batch.data.cpu().numpy()


        return loss_predict_batch

    def train_single_epoch(self, train, config, epoch_id):
        self.CCCFNet.train()
        np.random.shuffle(train)
        num_examples = len(train)
        loss_epoch = 0
        batch_size = config['batch_size']
        for i in tqdm(range(0, num_examples, batch_size)):
            batch_id = (i // batch_size) + 1
            batch_train = train[i: min(i + config['batch_size'], num_examples)]

            loss_predict_batch = self.train_single_batch(batch_train)

            print('\nBatch number:{}  Prediction loss：{} '.format(batch_id, loss_predict_batch))

            loss_epoch = loss_predict_batch + loss_epoch

        loss_epoch_mean = loss_epoch / (num_examples // batch_size + 1)
        print("the {} epoch loss:{}".format(epoch_id+1, loss_epoch_mean))
        with open('../parameters/CCCFNet_epoch_loss.txt', 'a+') as f:
            f.write("epoch {}    epoch loss:{}".format(epoch_id+1, loss_epoch_mean) +'\n')
        return loss_epoch_mean

    def evaluate(self, data):
        self.CCCFNet.eval()
        np.random.shuffle(data)
        m_ndcg_all_5, m_ndcg_all_10 = [], []
        b_ndcg_all_5, b_ndcg_all_10 = [], []
        m_prec_all_5, m_prec_all_10 = [], []
        b_prec_all_5, b_prec_all_10 = [], []
        for i in tqdm(range(len(data))):
            u = validate_test_sample(data[i])
            batch_size = 10
            d_m = len(u[0][0][2])
            d_b = len(u[0][0][5])

            U_P, mov_ID_P, I_mov_P, M_mov_P, bok_ID_P, I_bok_P, B_bok_P = np.empty([0, 1], dtype=int), np.empty([0, 1], dtype=int), np.empty([0, d_m],dtype=int), np.empty(
                [0, d_m], dtype=int), np.empty([0, 1], dtype=int), np.empty([0, d_b], dtype=int), np.empty([0, d_b], dtype=int)
            U_P = np.concatenate([U_P, np.array(u[0][0][0]).reshape(1, -1)], axis=0)
            mov_ID_P = np.concatenate([mov_ID_P, np.array(u[0][0][1]).reshape(1, -1)], axis=0)
            I_mov_P = np.concatenate([I_mov_P, u[0][0][2].reshape(1, -1)], axis=0)
            M_mov_P = np.concatenate([M_mov_P, u[0][0][3]], axis=0)
            bok_ID_P = np.concatenate([bok_ID_P, np.array(u[0][0][4]).reshape(1, -1)], axis=0)
            I_bok_P = np.concatenate([I_bok_P, u[0][0][5].reshape(1, -1)], axis=0)
            B_bok_P = np.concatenate([B_bok_P, u[0][0][6]], axis=0)
            for k in u[0][1]:
                U_P = np.concatenate([U_P, np.array(k[0]).reshape(1, -1)], axis=0)
                mov_ID_P = np.concatenate([mov_ID_P, np.array(k[1]).reshape(1, -1)], axis=0)
                I_mov_P = np.concatenate([I_mov_P, k[2].reshape(1, -1)], axis=0)
                M_mov_P = np.concatenate([M_mov_P, k[3]], axis=0)
                bok_ID_P = np.concatenate([bok_ID_P, np.array(k[4]).reshape(1, -1)], axis=0)
                I_bok_P = np.concatenate([I_bok_P, k[5].reshape(1, -1)], axis=0)
                B_bok_P = np.concatenate([B_bok_P, k[6]], axis=0)

            M_mov_P = np.reshape(M_mov_P, (batch_size, -1, d_m))
            B_bok_P = np.reshape(B_bok_P, (batch_size, -1, d_b))

            rating_mov_i, rating_bok_i = self.CCCFNet(batch_size, U_P, mov_ID_P, I_mov_P, M_mov_P, bok_ID_P, I_bok_P, B_bok_P)
            rating_mov_i = rating_mov_i.squeeze(dim=1).detach().cpu().numpy()
            rating_bok_i = rating_bok_i.squeeze(dim=1).detach().cpu().numpy()

            m_, m_Rank = torch.topk(torch.tensor(rating_mov_i), len(rating_mov_i), dim=0, largest=True, sorted=True)
            b_, b_Rank = torch.topk(torch.tensor(rating_bok_i), len(rating_bok_i), dim=0, largest=True, sorted=True)
            # NDCG @5 、@10

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

        m_average_ndcg_5 = np.mean(m_ndcg_all_5)
        b_average_ndcg_5 = np.mean(b_ndcg_all_5)

        m_average_ndcg_10 = np.mean(m_ndcg_all_10)
        b_average_ndcg_10 = np.mean(b_ndcg_all_10)

        m_average_prec_5 = np.mean(m_prec_all_5)
        b_average_prec_5 = np.mean(b_prec_all_5)

        m_average_prec_10 = np.mean(m_prec_all_10)
        b_average_prec_10 = np.mean(b_prec_all_10)

        print("[Movie evaluate :]NDCG@5={} , NDCG@10={} , Precision@5={} , Precision@10={}".format(m_average_ndcg_5, m_average_ndcg_10, m_average_prec_5, m_average_prec_10))
        print("[Book evaluate :] NDCG@5={} , NDCG@10={} , Precision@5={} , Precision@10={}".format(b_average_ndcg_5, b_average_ndcg_10, b_average_prec_5, b_average_prec_10))





