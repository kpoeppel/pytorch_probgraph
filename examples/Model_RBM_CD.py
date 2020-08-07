import sys
sys.path = [".."] + sys.path

import torch
from pytorch_probgraph import BernoulliLayer, GaussianLayer
from pytorch_probgraph import InteractionLinear, InteractionModule
from pytorch_probgraph import RestrictedBoltzmannMachineCD_Smooth
from pytorch_probgraph import DeepBoltzmannMachineLS


class Model_RBM_CD(torch.nn.Module):
    '''
    Constructs a RBM.
    '''

    def __init__(self):
        '''
        Builds the RBM with hidden dimension = 200.
        '''

        super().__init__()
        # initialize the bias with zeros
        l0bias = torch.zeros([1, 1, 28, 28])
        l0bias.requires_grad = True
        l1bias = torch.zeros([1, 200])
        l1bias.requires_grad = True

        # initialize the bernoulli layers
        l0 = BernoulliLayer(l0bias)
        l1 = BernoulliLayer(l1bias)

        # initialize the interaction layer
        i0 = InteractionLinear(l0.bias.shape[1:], l1.bias.shape[1:])

        # build the RBM
        rbm0 = RestrictedBoltzmannMachineCD_Smooth(l0, l1, i0, ksteps=1)

        # get all parameters of the RBM for the optimizer
        params = list(rbm0.parameters())

        # set up the optimizer and the scheduler
        self.opt = torch.optim.SGD(params, lr=1e-2, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=2000, gamma=0.94)
        self.model = rbm0

        #The DBM has a general AIS implementation, which can be used to calculate the log likelihood of the RBM.
        self.dbm = DeepBoltzmannMachineLS(rbms=[rbm0], optimizer=self.opt, scheduler=self.scheduler, learning='CD',
                                        nFantasy=100)

    def train(self, data, epochs=1, device=None):
        '''
        Function to train the model
        :param data: tqdm object
        :param epochs: [int], number of epochs to train
        :return: None
        '''

        self.model.train(data, epochs=epochs, optimizer=self.opt, scheduler=self.scheduler, device=device)

    def loglikelihood(self, data, log_Z = None):
        '''
        Calculates the log likelihood
        :param data: tqdm object
        :param log_Z: [float], log of the partitioning sum
        :return: [torch.tensor(batch size, float)], log likelihood per image
        '''

        return self.dbm.loglikelihood(data, log_Z = log_Z)

    def generate(self, N=1):
        '''
        Generates new images according to the model distribution
        :param N: number of images to be generated
        :return: [torch.tensor(N,28,28)], generated images
        '''

        return self.model.reconstruct(N=N, gibbs_steps=10, mean=True).cpu()

    def get_log_Z(self, steps, samples):
        '''
        Calculates the partitioning sum.
        :param steps: [int], number of steps for the AIS algorithm
        :param samples: [int], number of samples to compute the mean
        :return: [torch.tensor(1, float)] log of the partitioning sum
        '''

        return self.dbm.ais(steps, samples)
