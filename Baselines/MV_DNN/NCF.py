import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn

class Cosine_Score(torch.nn.Module):
    def __init__(self, latent_media_dim, latent_dim):
        super(Cosine_Score, self).__init__()
        self.latent_media_dim = latent_media_dim
        self.latent_dim = latent_dim
        self.mlp = torch.nn.Linear(self.latent_media_dim, self.latent_dim)
        self.out = torch.nn.Linear(self.latent_dim, 1)
        self.score = torch.nn.Tanh()
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_normal_(self.mlp.weight)
        nn.init.xavier_normal_(self.out.weight)

    def forward(self, user, item):
        user_latent = self.mlp(user)
        item_latent = self.mlp(item)
        cos_rating = torch.cosine_similarity(user_latent, item_latent, dim=0)
        out = self.out(cos_rating)
        score=self.score(out)
        return score

