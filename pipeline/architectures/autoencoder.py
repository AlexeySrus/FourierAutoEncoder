import torch
from torch import nn


class FFTAE(nn.Module):
    def __init__(self, fft_shape=None, latent_dim=10):
        super(FFTAE, self).__init__()

        self.fft_shape = fft_shape
        self.latent_dim = latent_dim

        self.dense1 = nn.Linear(fft_shape[0]*fft_shape[1]*2, self.latent_dim*2)
        self.dense2 = nn.Linear(self.latent_dim*2, self.latent_dim)
        self.dense3 = nn.Linear(self.latent_dim, self.latent_dim*2)
        self.dense4 = nn.Linear(self.latent_dim*2, fft_shape[0]*fft_shape[1]*2)

    def forward(self, x):
        bs = x.size(0)
        x = x.view(bs, self.fft_shape[0]*self.fft_shape[1]*2)
        x = nn.ReLU()(self.dense1(x))
        x = nn.ReLU()(self.dense2(x))
        x = nn.ReLU()(self.dense3(x))
        x = self.dense4(x)
        x = x.view(bs, 2, self.fft_shape[0], self.fft_shape[1])
        return x
