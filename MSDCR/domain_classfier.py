import torch
import torch.nn as nn

# Domain invariant adversarial adaptation
class Domain_classfier(torch.nn.Module):
    def __init__(self, latent_dim, num_aspects, domain_num=2):
        super(Domain_classfier, self).__init__()
        self.latent_dim = latent_dim
        self.num_aspects = num_aspects
        self.domain_num = domain_num
        self.dis = nn.Sequential(
            nn.Linear(self.latent_dim * self.num_aspects, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, self.domain_num)

        )

    def init_weight(self):
        nn.init.xavier_normal_(self.dis.weight)

    def forward(self, C_mov, C_bok):
        batch_size, num_aspects, latent_dim = C_mov.size()
        c_m = C_mov.reshape(batch_size, num_aspects * latent_dim)
        c_b = C_bok.reshape(batch_size, num_aspects * latent_dim)
        c_mov_predict = self.dis(c_m)
        c_bok_predict = self.dis(c_b)

        return c_mov_predict, c_bok_predict


# Domain sepcific adversarial adaptation
class Domain_classfier_specific(torch.nn.Module):
    def __init__(self, latent_dim, num_aspects_specific, domain_num=2):
        super(Domain_classfier_specific, self).__init__()
        self.latent_dim = latent_dim
        self.num_aspects = num_aspects_specific
        self.domain_num = domain_num
        self.dis = nn.Sequential(
            nn.Linear(self.latent_dim * self.num_aspects, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, self.domain_num)

        )

    def init_weight(self):
        nn.init.xavier_normal_(self.dis.weight)

    def forward(self, C_mov_specific, C_bok_specific):
        batch_size, num_aspects, latent_dim = C_mov_specific.size()
        c_m = C_mov_specific.reshape(batch_size, num_aspects * latent_dim)
        c_b = C_bok_specific.reshape(batch_size, num_aspects * latent_dim)
        c_mov_specific_predict = self.dis(c_m)
        c_bok_specific_predict = self.dis(c_b)

        return c_mov_specific_predict, c_bok_specific_predict



