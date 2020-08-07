
import site
site.addsitedir('..')

import torch
from pytorch_probgraph import BernoulliLayer
from pytorch_probgraph import InteractionLinear
from pytorch_probgraph import HelmholtzMachine
from itertools import chain
from tqdm import tqdm

class Model_HM_WS(torch.nn.Module):
    def __init__(self):
        super().__init__()
        layer0 = BernoulliLayer(torch.nn.Parameter(torch.zeros([1, 1, 28, 28]), requires_grad=True))
        layer1 = BernoulliLayer(torch.nn.Parameter(torch.zeros([1, 200]), requires_grad=True))
        layer2 = BernoulliLayer(torch.nn.Parameter(torch.zeros([1, 200]), requires_grad=True))

        interactionUp1 = InteractionLinear(layer0.bias.shape[1:], layer1.bias.shape[1:])
        interactionDown1 = InteractionLinear(layer1.bias.shape[1:], layer0.bias.shape[1:])
        interactionUp2 = InteractionLinear(layer1.bias.shape[1:], layer2.bias.shape[1:])
        interactionDown2 = InteractionLinear(layer2.bias.shape[1:], layer1.bias.shape[1:])

        parameters = chain(*[m.parameters() for m in [layer0, layer1, layer2, interactionUp1, interactionUp2, interactionDown1, interactionDown2]])
        opt = torch.optim.Adam(parameters)

        self.model = HelmholtzMachine([layer0, layer1, layer2],
                                      [interactionUp1, interactionUp2],
                                      [interactionDown1, interactionDown2],
                                      optimizer=opt)
        #print(interaction.weight.shape)

    def train(self, data, epochs=1, device=None):
        for epoch in range(epochs):
            for dat in data:
                self.model.trainWS(dat.to(device))
            if isinstance(data, tqdm):
                data = tqdm(data)
            #print(torch.sum(self.model.interaction.weight))

    def loglikelihood(self, data):
        return self.model.loglikelihood(data, ksamples=100).cpu().detach()

    def generate(self, N=1):
        return self.model.sampleAll(N=N)[0][0].cpu()
