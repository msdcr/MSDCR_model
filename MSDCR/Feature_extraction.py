import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

# Aspect Mapping layer and Aspect Attention layer
class Feature_extraction(torch.nn.Module):
    def __init__(self,latent_dim, num_aspects,attn_dropout):
        super(Feature_extraction, self).__init__()
        self.latent_dim = latent_dim
        self.num_aspects = num_aspects
        self.attn_dropout = attn_dropout
        self.dropout = nn.Dropout(self.attn_dropout)
        self.asp_vs_d = nn.Parameter(torch.Tensor(self.num_aspects, self.latent_dim))
        self.W_map = nn.ParameterList(
            [nn.Parameter(torch.Tensor(2 * self.latent_dim, self.latent_dim)) for i in range(self.num_aspects)])
        self.b_map = nn.ParameterList(
            [nn.Parameter(torch.Tensor(1, self.latent_dim)) for i in range(self.num_aspects)])
        self.W_proj = nn.ParameterList(
            [nn.Parameter(torch.Tensor(2 * self.latent_dim, self.latent_dim)) for i in range(self.num_aspects)])
        self.b_proj = nn.ParameterList(
            [nn.Parameter(torch.Tensor(1, self.latent_dim)) for i in range(self.num_aspects)])
        self.__init__weight()

    def __init__weight(self):
        init.xavier_normal_(self.asp_vs_d)
        for i in range(self.num_aspects):
            init.xavier_normal_(self.W_map[i])
            init.xavier_normal_(self.b_map[i])
            init.xavier_normal_(self.W_proj[i])
            init.xavier_normal_(self.b_proj[i])

    def forward(self, H, user_embedding):


        batch_size = len(H)
        length = len(H[0])
        User = []
        for i in range(len(user_embedding)):
            User.append(user_embedding[i].repeat(length, 1))
        User = torch.reshape((torch.cat(User, dim=0)), (batch_size, length, -1))
        Batch = torch.cat([H, User], dim=2)

        map_asp_embed_concat = []
        for i in range(self.num_aspects):
            map_item_embed = torch.sigmoid(
                (torch.matmul(Batch, self.W_map[i]) + self.b_map[i]))
            proj_item_embed = torch.matmul(Batch, self.W_proj[i]) + self.b_proj[i]
            map_asp_embed = map_item_embed * proj_item_embed

            map_asp_embed_concat.append(map_asp_embed)
        # item's in H representation on M aspects
        map_asp_embed_concat = torch.cat(map_asp_embed_concat, dim=0).view(self.num_aspects, batch_size,
                                                                           length, self.latent_dim)
        item_asp_concat = map_asp_embed_concat.reshape(batch_size, length, self.num_aspects, self.latent_dim)

        # Attention to n items on every aspects
        attn = torch.softmax((torch.sum(item_asp_concat * self.asp_vs_d, dim=3) + 1e-24), 1)
        attn = attn.unsqueeze(3).expand(batch_size, length, self.num_aspects, self.latent_dim)
        asp_repr = self.dropout(torch.sum(item_asp_concat * attn, dim=1))

        return asp_repr

# Domain specific Preference Enhancement
class Feature_extra_specific(torch.nn.Module):
    def __init__(self,latent_dim, num_aspects_specific,attn_dropout):
        super(Feature_extra_specific, self).__init__()
        self.latent_dim = latent_dim
        self.num_aspects_specific = num_aspects_specific
        self.dropout = nn.Dropout(attn_dropout)
        self.asp_vs_d = nn.Parameter(torch.Tensor(self.num_aspects_specific, self.latent_dim))
        self.W_map = nn.ParameterList(
            [nn.Parameter(torch.Tensor(2 * self.latent_dim, self.latent_dim)) for i in range(self.num_aspects_specific)])
        self.b_map = nn.ParameterList(
            [nn.Parameter(torch.Tensor(1, self.latent_dim)) for i in range(self.num_aspects_specific)])
        self.W_proj = nn.ParameterList(
            [nn.Parameter(torch.Tensor(2 * self.latent_dim, self.latent_dim)) for i in range(self.num_aspects_specific)])
        self.b_proj = nn.ParameterList(
            [nn.Parameter(torch.Tensor(1, self.latent_dim)) for i in range(self.num_aspects_specific)])
        self.__init__weight()

    def __init__weight(self):
        init.xavier_normal_(self.asp_vs_d)
        for i in range(self.num_aspects_specific):
            init.xavier_normal_(self.W_map[i])
            init.xavier_normal_(self.b_map[i])
            init.xavier_normal_(self.W_proj[i])
            init.xavier_normal_(self.b_proj[i])

    def forward(self, H, user_embedding):


        batch_size = len(H)
        length = len(H[0])
        User = []
        for i in range(len(user_embedding)):
            User.append(user_embedding[i].repeat(length, 1))
        User = torch.reshape((torch.cat(User, dim=0)), (batch_size, length, -1))
        Batch = torch.cat([H, User], dim=2)

        map_asp_embed_concat = []
        for i in range(self.num_aspects_specific):
            map_item_embed = torch.sigmoid(
                (torch.matmul(Batch, self.W_map[i]) + self.b_map[i]))
            proj_item_embed = torch.matmul(Batch, self.W_proj[i]) + self.b_proj[i]
            map_asp_embed = map_item_embed * proj_item_embed

            map_asp_embed_concat.append(map_asp_embed)
        # item's in H representation on M aspects
        map_asp_embed_concat = torch.cat(map_asp_embed_concat, dim=0).view(self.num_aspects_specific, batch_size, length, self.latent_dim)
        item_asp_concat = map_asp_embed_concat.reshape(batch_size, length, self.num_aspects_specific, self.latent_dim)

        # Attention to n items on every aspects
        attn = torch.softmax((torch.sum(item_asp_concat * self.asp_vs_d, dim=3) + 1e-24), 1)
        attn = attn.unsqueeze(3).expand(batch_size, length, self.num_aspects_specific, self.latent_dim)
        asp_repr_specific = self.dropout(torch.sum(item_asp_concat * attn, dim=1))

        return asp_repr_specific

# Domain Preference Enhancement
class Domain_enhancement(torch.nn.Module):
    def __init__(self, latent_dim, num_aspects_specific, domain_num=2):
        super(Domain_enhancement,self).__init__()
        self.latent_dim = latent_dim
        self.num_aspects = num_aspects_specific
        self.domain_num = domain_num
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.W_map_b_mo = nn.Parameter(torch.Tensor(2 * self.latent_dim, self.latent_dim))
        self.b_map_b_mo = nn.Parameter(torch.Tensor(1, self.latent_dim))

        self.W_merge_movie = nn.Parameter(torch.Tensor(self.domain_num * self.latent_dim, self.latent_dim))
        self.b_merge_movie = nn.Parameter(torch.Tensor(1, self.latent_dim))

        self.W_map_mo_b = nn.Parameter(torch.Tensor(2 * self.latent_dim, self.latent_dim))
        self.b_map_mo_b = nn.Parameter(torch.Tensor(1, self.latent_dim))

        self.W_merge_book = nn.Parameter(torch.Tensor(self.domain_num * self.latent_dim, self.latent_dim))
        self.b_merge_book = nn.Parameter(torch.Tensor(1, self.latent_dim))

        self.__init__weight()

    def __init__weight(self):
        init.xavier_normal_(self.W_map_b_mo)
        init.xavier_normal_(self.b_map_b_mo)
        init.xavier_normal_(self.W_merge_movie)
        init.xavier_normal_(self.b_merge_movie)
        init.xavier_normal_(self.W_map_mo_b)
        init.xavier_normal_(self.b_map_mo_b)
        init.xavier_normal_(self.W_merge_book)
        init.xavier_normal_(self.b_merge_book)

    def forward(self, C_mov, C_bok):
        bok_mov_merge=torch.cat([C_mov - C_bok, C_mov * C_bok],dim=2)
        mov_bok_merge = torch.cat([C_bok - C_mov, C_bok * C_mov], dim=2)

        bok_mov_gate=self.sigmoid(torch.matmul(
            bok_mov_merge, self.W_map_b_mo)+self.b_map_b_mo)
        mov_bok_gate = self.sigmoid(torch.matmul(
            mov_bok_merge, self.W_map_mo_b) + self.b_map_mo_b)

        movie_repr_ehanc=self.tanh(torch.matmul(
            torch.cat([bok_mov_gate * C_mov,  C_mov], dim=2), self.W_merge_movie) + self.b_merge_movie)
        book_repr_ehanc=self.tanh(torch.matmul(
            torch.cat([mov_bok_gate * C_bok, C_bok], dim=2), self.W_merge_book) + self.b_merge_book)

        return movie_repr_ehanc, book_repr_ehanc



