import site
site.addsitedir('..')

import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_probgraph import BernoulliLayer, DiracDeltaLayer, CategoricalLayer
from pytorch_probgraph import GaussianLayer
from pytorch_probgraph import InteractionLinear, InteractionModule, InteractionPoolMapIn2D, InteractionPoolMapOut2D
from pytorch_probgraph import InteractionPoolMapIn1D, InteractionPoolMapOut1D
from pytorch_probgraph import RestrictedBoltzmannMachinePCD
from pytorch_probgraph import DeepBeliefNetwork
from itertools import chain
from tqdm import tqdm



class Model_DBN_IntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        layer0 = BernoulliLayer(torch.zeros([1, 784], requires_grad=True))
        layer1 = BernoulliLayer(torch.zeros([1, 200], requires_grad=True))
        layer2 = BernoulliLayer(torch.zeros([1, 200], requires_grad=True))
        interaction0 = InteractionModule(torch.nn.Linear(layer0.bias.shape[1], layer1.bias.shape[1]), inputShape=layer0.bias.shape)
        interaction1 = InteractionModule(torch.nn.Linear(layer1.bias.shape[1], layer2.bias.shape[1]), inputShape=layer1.bias.shape)
        rbm1 = RestrictedBoltzmannMachinePCD(layer0, layer1, interaction0, fantasy_particles=10)
        rbm2 = RestrictedBoltzmannMachinePCD(layer1, layer2, interaction1, fantasy_particles=10)
        opt = torch.optim.Adam(chain(rbm1.parameters(), rbm2.parameters()), lr=1e-3)
        self.model = DeepBeliefNetwork([rbm1, rbm2], opt)
        #self.model = self.model.to(device)
        #print(interaction.weight.shape)

    def train(self, data, epochs=1, device=None):
        datnew = [dat.reshape(-1, 784) for dat in data]
        if isinstance(data, tqdm):
              datnew = tqdm(datnew)
        self.model.train(datnew, epochs=epochs, device=device)

    def loglikelihood(self, data):
        if data.shape[0] == 1:
            dataresh = data.reshape(-1, 784)
        else:
            dataresh = data.reshape(-1, 784)
        return -self.model.free_energy_estimate(dataresh)

    def generate(self, N=1):
        return self.model.sample(N=N, gibbs_steps=100).cpu().reshape(-1,28,28)
