import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn

## movie rating predicition thought NCF layer

class Domain_NCF(torch.nn.Module):
    def __init__(self, latent_dim, attn_dropout=0.1):
        super(Domain_NCF, self).__init__()
        self.latent_dim = latent_dim
        self.mlp = nn.Sequential(
            nn.Linear(2 * self.latent_dim, 256),
            nn.Dropout(attn_dropout),
            nn.Linear(256, 128),
            nn.Dropout(attn_dropout),
            nn.Linear(128, 64),
            nn.Dropout(attn_dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def init_weight(self):
        nn.init.xavier_normal_(self.mlp.weight)

    def forward(self, user, item):
        batch_size, _, _ = user.size()
        user = torch.reshape(user,(batch_size, self.latent_dim))
        vector = torch.cat([user, item], dim=1)
        rating = self.mlp(vector)
        return rating

