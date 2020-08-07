import sys
sys.path = [".."] + sys.path

from pytorch_probgraph import BernoulliLayer
from pytorch_probgraph import InteractionLinear
from pytorch_probgraph import RestrictedBoltzmannMachineCD_Smooth
from pytorch_probgraph import DeepBoltzmannMachineLS


import torch

class Model_DBM_CD(torch.nn.Module): #


    def __init__(self):
        '''
        Builds the DBM with hidden dimension = 200, 200. Uses PCD for learning.
        '''
        super().__init__()

        # initialize the bias with zeros and lock them.
        # This is because the DBM approximates the bias distribution with higher layers.
        l0bias = torch.zeros([1, 1, 28, 28])
        l0bias.requires_grad = False
        l1bias = torch.zeros([1, 200])
        l1bias.requires_grad = False
        l2bias = torch.zeros([1, 200])
        l2bias.requires_grad = False

        # initialize the bernoulli layers
        l0 = BernoulliLayer(l0bias)
        l1 = BernoulliLayer(l1bias)
        l2 = BernoulliLayer(l2bias)

        # initialize the interaction layers
        i0 = InteractionLinear(l0.bias.shape[1:], l1.bias.shape[1:])
        i1 = InteractionLinear(l1.bias.shape[1:], l2.bias.shape[1:])

        # build two RBMs
        rbm0 = RestrictedBoltzmannMachineCD_Smooth(l0, l1, i0, ksteps=1)
        rbm1 = RestrictedBoltzmannMachineCD_Smooth(l1, l2, i1, ksteps=1)

        # get all parameters of the RBM for the optimizer
        params = list(rbm0.parameters()) + list(rbm1.parameters())

        # set up the optimizer and the scheduler
        opt = torch.optim.SGD(params, lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=700, gamma=0.94)

        # build the DBM
        self.model = DeepBoltzmannMachineLS(rbms=[rbm0, rbm1], optimizer=opt, scheduler=scheduler, learning='CD',
                                          nFantasy=100, ksteps=1)

    def train(self, data, epochs=1, device=None):
        '''
        Function to train the model
        :param data: tqdm object
        :param epochs: [int], number of epochs to train
        :return: None
        '''
        self.model.train_model(data, epochs=epochs, device=device)

    def loglikelihood(self, data, log_Z = None):
        '''
        Calculates the log likelihood
        :param data: tqdm object
        :param log_Z: [float], log of the partitioning sum
        :return: [torch.tensor(batch size, float)], log likelihood per image
        '''
        return self.model.loglikelihood(data, log_Z = log_Z)

    def generate(self, N=1):
        '''
        Generates new images according to the model distribution
        :param N: number of images to be generated
        :return: [torch.tensor(N,28,28)], generated images
        '''

        return self.model.generate(N=N, gibbs_steps=10).cpu()

    def get_log_Z(self, steps, samples):
        '''
        Calculates the partitioning sum.
        :param steps: [int], number of steps for the AIS algorithm
        :param samples: [int], number of samples to compute the mean
        :return: [torch.tensor(1, float)] log of the partitioning sum
        '''

        return self.model.ais(steps, samples)
