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
from typing import Iterable
from tqdm import tqdm



class Model_DBN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        layer0 = BernoulliLayer(torch.zeros([1, 1, 28, 28], requires_grad=True))
        layer1 = BernoulliLayer(torch.zeros([1, 200], requires_grad=True))
        layer2 = BernoulliLayer(torch.zeros([1, 200], requires_grad=True))
        interaction0 = InteractionLinear(layer0.bias.shape[1:], layer1.bias.shape[1:])
        interaction1 = InteractionLinear(layer1.bias.shape[1:], layer2.bias.shape[1:])
        rbm1 = RestrictedBoltzmannMachinePCD(layer0, layer1, interaction0, fantasy_particles=10)
        rbm2 = RestrictedBoltzmannMachinePCD(layer1, layer2, interaction1, fantasy_particles=10)
        opt = torch.optim.Adam(chain(rbm1.parameters(), rbm2.parameters()), lr=1e-3)
        self.model = DeepBeliefNetwork([rbm1, rbm2], opt)
        #self.model = self.model.to(device)
        #print(interaction.weight.shape)

    def train(self,
              data: Iterable[torch.tensor],
              epochs: int=1,
              device: torch.device=None
              ) -> None:
        self.model.train(data, epochs=epochs, device=device)

    def loglikelihood(self,
                      data: torch.Tensor
                      ) -> torch.Tensor:
        return -self.model.free_energy_estimate(data)

    def generate(self, N=1):
        return self.model.sample(N=N, gibbs_steps=100).cpu()
