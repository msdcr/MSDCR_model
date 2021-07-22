import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn

# Calculate the cosine similarity between User and each domain item, which is shared by all domains in this layer

class CC_NCF_mov(torch.nn.Module):
    def __init__(self, latent_dim, attn_dropout=0.1):
        super(CC_NCF_mov, self).__init__()
        self.latent_dim = latent_dim
        self.movie_mlp = nn.Sequential(
            nn.Linear((self.latent_dim * 4), 256),
            nn.Dropout(attn_dropout),
            nn.Linear(256, 128),
            nn.Dropout(attn_dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def init_weight(self):
        nn.init.xavier_normal_(self.movie_mlp.weight)

    def forward(self, a_movie, e_movie):
        vector_mov = torch.cat([a_movie, e_movie], dim=2)
        rating_mov = self.movie_mlp(vector_mov)
        return rating_mov


class CC_NCF_bok(torch.nn.Module):
    def __init__(self, latent_dim, attn_dropout=0.1):
        super(CC_NCF_bok, self).__init__()
        self.latent_dim = latent_dim
        self.book_mlp = nn.Sequential(
            nn.Linear((self.latent_dim * 4), 256),
            nn.Dropout(attn_dropout),
            nn.Linear(256, 128),
            nn.Dropout(attn_dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def init_weight(self):
        nn.init.xavier_normal_(self.movie_mlp.weight)

    def forward(self, a_book, e_book):
        vector_bok = torch.cat([a_book, e_book], dim=2)
        rating_bok = self.book_mlp(vector_bok)
        return rating_bok