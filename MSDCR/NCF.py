import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn

## movie rating predicition thought NCF layer

class Movie_NCF(torch.nn.Module):
    def __init__(self, latent_dim, num_aspects, num_aspects_specific, attn_dropout=0.1):
        super(Movie_NCF, self).__init__()
        self.latent_dim = latent_dim
        self.num_aspects = num_aspects
        self.num_aspects_specific = num_aspects_specific
        self.movie_mlp = nn.Sequential(
            nn.Linear((self.latent_dim * (self.num_aspects + self.num_aspects_specific + 1)), 256),
            nn.Dropout(attn_dropout),
            nn.Linear(256, 128),
            nn.Dropout(attn_dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def init_weight(self):
        nn.init.xavier_normal_(self.movie_mlp.weight)

    def forward(self, A_movie, C_movie, e_movie):
        batch_size, num_aspects, latent_dim = A_movie.size()
        a_movie = torch.reshape(A_movie, (batch_size, self.num_aspects_specific * self.latent_dim))
        c_movie = torch.reshape(C_movie, (batch_size, self.num_aspects * self.latent_dim))
        vector_mov = torch.cat([a_movie, c_movie, e_movie], dim=1)
        rating_mov = self.movie_mlp(vector_mov)
        return rating_mov

## book rating predicition thought NCF layer

class Book_NCF(torch.nn.Module):
    def __init__(self,latent_dim, num_aspects, num_aspects_specific ,attn_dropout=0.1):
        super(Book_NCF,self).__init__()
        self.latent_dim = latent_dim
        self.num_aspects = num_aspects
        self.num_aspects_specific = num_aspects_specific
        self.book_mlp = nn.Sequential(
            nn.Linear((self.latent_dim * (self.num_aspects + self.num_aspects_specific + 1)), 256),
            nn.Dropout(attn_dropout),
            nn.Linear(256, 128),
            nn.Dropout(attn_dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def init_weight(self):
        nn.init.xavier_normal_(self.book_mlp.weight)

    def forward(self, A_book, C_book,e_book):
        batch_size, num_aspects, latent_dim = A_book.size()
        a_book = torch.reshape(A_book, (batch_size, self.num_aspects_specific * self.latent_dim))
        c_book = torch.reshape(C_book, (batch_size, self.num_aspects * self.latent_dim))
        vector_bok = torch.cat([a_book, c_book, e_book], dim=1)
        rating_bok = self.book_mlp(vector_bok)
        return rating_bok



