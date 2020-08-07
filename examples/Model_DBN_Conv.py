import site
site.addsitedir('..')

import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_probgraph import BernoulliLayer, DiracDeltaLayer, CategoricalLayer
from pytorch_probgraph import GaussianLayer
from pytorch_probgraph import InteractionLinear, InteractionModule, InteractionPoolMapIn2D, InteractionPoolMapOut2D
from pytorch_probgraph import InteractionSequential
from pytorch_probgraph import RestrictedBoltzmannMachinePCD
from pytorch_probgraph import DeepBeliefNetwork
from itertools import chain
from tqdm import tqdm



class Model_DBN_Conv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        layer0 = BernoulliLayer(torch.zeros([1, 1, 28, 28], requires_grad=True))
        layer1 = BernoulliLayer(torch.zeros([1, 40, 17, 17], requires_grad=True))

        interaction0 = InteractionModule(torch.nn.Conv2d(1,40,12))

        rbm1 = RestrictedBoltzmannMachinePCD(layer0, layer1, interaction0, fantasy_particles=10)
        opt = torch.optim.Adam(rbm1.parameters(), lr=1e-3)
        self.model = DeepBeliefNetwork([rbm1], opt)
        #self.model = self.model.to(device)
        #print(interaction.weight.shape)

    def train(self, data, epochs=1, device=None):
        self.model.train(data, epochs=epochs, device=device)

    def loglikelihood(self, data):
        return -self.model.free_energy_estimate(data)

    def generate(self, N=1):
        return self.model.sample(N=N, gibbs_steps=100).cpu()
