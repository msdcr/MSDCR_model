import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import torch.nn.functional as F


class Attention_movie(nn.Module):

    def __init__(self, latent_dim, d_k=32, d_v=32, n_heads=1, is_layer_norm=True, attn_dropout=0.1):
        super(Attention_movie, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k if d_k is not None else latent_dim
        self.d_v = d_v if d_v is not None else latent_dim

        self.is_layer_norm = is_layer_norm
        if is_layer_norm:
            self.layer_morm = nn.LayerNorm(normalized_shape=latent_dim)

        self.W_q = nn.Parameter(torch.Tensor(latent_dim, n_heads * d_k))
        self.W_k = nn.Parameter(torch.Tensor(latent_dim, n_heads * d_k))
        self.W_v = nn.Parameter(torch.Tensor(latent_dim, n_heads * d_v))

        self.W_o = nn.Parameter(torch.Tensor(d_v*n_heads, latent_dim))
        self.linear1 = nn.Linear(latent_dim, latent_dim)
        self.linear2 = nn.Linear(latent_dim, latent_dim)

        self.dropout = nn.Dropout(attn_dropout)
        self.__init_weights__()

    def __init_weights__(self):
        init.xavier_normal_(self.W_q)
        init.xavier_normal_(self.W_k)
        init.xavier_normal_(self.W_v)
        init.xavier_normal_(self.W_o)

        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    # def FFN(self, X):
    #     output = self.linear2(F.relu(self.linear1(X)))
    #     output = self.dropout(output)
    #     return output

    def scaled_dot_product_attention(self, Q, K, V):
        '''
        :param Q: (*, max_q_words, n_heads, input_size)
        :param K: (*, max_k_words, n_heads, input_size)
        :param V: (*, max_v_words, n_heads, input_size)
        :param episilon:
        :return:
        '''
        temperature = self.d_k ** 0.5
        Q_K = torch.einsum("bqd,bkd->bqk", Q, K) / (temperature)
        Q_K_score = F.softmax(Q_K, dim=-1)  # (batch_size, max_q_words, max_k_words)
        Q_K_score = self.dropout(Q_K_score)

        V_att = Q_K_score.bmm(V)  # (*, max_q_words, input_size)
        return V_att


    def multi_head_attention(self, Q, K, V):
        batch_size, q_len, _ = Q.size()
        batch_size, k_len, _ = K.size()
        batch_size, v_len, _ = V.size()

        Q_ = Q.matmul(self.W_q).view(batch_size, q_len, self.n_heads, self.d_k)
        K_ = K.matmul(self.W_k).view(batch_size, k_len, self.n_heads, self.d_k)
        V_ = V.matmul(self.W_v).view(batch_size, v_len, self.n_heads, self.d_v)

        Q_ = Q_.permute(0, 2, 1, 3).contiguous().view(batch_size*self.n_heads, q_len, self.d_k)
        K_ = K_.permute(0, 2, 1, 3).contiguous().view(batch_size*self.n_heads, q_len, self.d_k)
        V_ = V_.permute(0, 2, 1, 3).contiguous().view(batch_size*self.n_heads, q_len, self.d_v)

        V_att = self.scaled_dot_product_attention(Q_, K_, V_)
        V_att = V_att.view(batch_size, self.n_heads, q_len, self.d_v)
        V_att = V_att.permute(0, 2, 1, 3).contiguous().view(batch_size, q_len, self.n_heads*self.d_v)

        output = self.dropout(V_att.matmul(self.W_o)) # (batch_size, max_q_words, latent_dim)
        return output


    def forward(self, Q, K, V):
        '''
        :param Q: (batch_size, max_q_words, input_size)
        :param K: (batch_size, max_k_words, input_size)
        :param V: (batch_size, max_v_words, input_size)
        :return:  output: (batch_size, max_q_words, input_size)  same size as Q
        '''
        V_att = self.multi_head_attention(Q, K, V)

        if self.is_layer_norm:
            X = self.layer_morm(Q + V_att)  # (batch_size, max_r_words, embedding_dim)
            output = self.layer_morm(X)
            # output = self.layer_morm(self.FFN(X) + X)
        else:
            X = Q + V_att
            output=X
            # output = self.FFN(X) + X
        return output

# if __name__=='__main__':
#     x=torch.randn(1,32)
#     input_dim_bok=32
#     latent_dim=32
#     net1= Embedding_movie(input_dim_bok,latent_dim )
#     z1=net1(x)
#     print(z1)
#     net=Attention_movie(latent_dim)
#     z=net(x,x,x)
#     print(z)