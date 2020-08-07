'''
A library implementing a generic sigmoid belief network aka Helmholtz Machine.

'''

from typing import List, Tuple, Union
from .interaction import Interaction
from .unitlayer import UnitLayer
from itertools import chain
import torch
import numpy as np
from .utils import ListModule

def logsumexp(x, dim=0, keepdim=False):
    maxval = torch.max(x, dim=dim, keepdim=True).values
    return torch.log(torch.sum(torch.exp(x - maxval), dim=dim, keepdim=keepdim))\
           + torch.sum(maxval, dim, keepdim=keepdim)

class HelmholtzMachine(torch.nn.Module):
    '''
    A multilayer sigmoid belief network with (reweighted) wake-sleep learning.
    Using asymmetric conditional probabilities (interaction weights).
    From:

    [1] G.Hinton et al. "The wake-sleep algorithm for unsupervised
    neural networks"

    [2] Peter Dayan: Helmholtz Machines and Wake-Sleep Learning
    http://www.gatsby.ucl.ac.uk/~dayan/papers/d2000a.pdf
    Note that this implementation uses tied biases for generative and
    reconstructed probabilities.

    [3] https://arxiv.org/pdf/1406.2751.pdf

    [4] https://github.com/jbornschein/reweighted-ws

    '''

    def __init__(self,
                 layers: List[UnitLayer],
                 interactionsUp: List[Interaction],
                 interactionsDown: List[Interaction],
                 optimizer: torch.optim.Optimizer):
        '''
        :param layers: UnitLayers of Random Units
        :param interactionsUp: List of Interactions upwards
        :param interactionsDown: List of Interactions downwards
        :param optimizer: Optimizer for training
        '''
        super().__init__()
        if len(interactionsUp) != len(interactionsDown) or \
           len(layers)-1 != len(interactionsUp):
            raise ValueError('Non fitting layers')
        self.layers = ListModule(*layers)
        self.intsUp = ListModule(*interactionsUp)
        self.intsDown = ListModule(*interactionsDown)

        self.optim = optimizer

    def sampleQ(self,
                data: torch.Tensor
                ) -> Tuple[List[torch.Tensor],
                           List[torch.Tensor],
                           List[torch.Tensor],
                           torch.Tensor]:
        '''
        :param data: Data to sample Q (reconstruction model) from.
        :return: Samples/Means/Logprobs from reconstruction distribution (for all layers) + Total LogProb
        '''
        samplesUp = [data]
        meansUp = [None]
        logprobsUp = [0.]
        logprobsUp_total = 0.
        nlayers = len(self.layers)
        for i in range(nlayers-1):
            intterm = self.intsUp[i].gradOutput(self.layers[i].transform(samplesUp[i]))
            mean = self.layers[i+1].mean_cond(interaction=intterm)
            samp = self.layers[i+1].sample_cond(interaction=intterm)
            logprob = self.layers[i+1].logprob_cond(samp, intterm)
            samplesUp.append(samp)
            meansUp.append(mean)
            logprobsUp.append(logprob)
            logprobsUp_total += logprob
        return samplesUp, meansUp, logprobsUp, logprobsUp_total

    def logprobP(self,
                 total_samples: List[torch.Tensor]
                 ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        '''
        :param total_samples: Samples from all layers
        :return: logprob P of generative model of these samples
        '''
        logprob = [self.layers[-1].logprob_cond(total_samples[-1], interaction=0.)]
        logprob_total = logprob[0]
        for n in reversed(range(len(self.layers)-1)):
            interaction = self.intsDown[n].gradOutput(self.layers[n+1].transform(total_samples[n+1]))
            logprobn = self.layers[n].logprob_cond(total_samples[n], interaction=interaction)
            logprob = [logprobn] + logprob
            logprob_total += logprobn
        return logprob, logprob_total

    def wakePhaseReweighted(self,
                            data: torch.Tensor,
                            ksamples: int=1,
                            kmax_parallel: int=1000,
                            train: bool=True,
                            wakePhaseQ: bool=True
                            ) -> torch.Tensor:
        '''
        According to https://github.com/jbornschein/reweighted-ws/blob/master/learning/models/rws.py
        So k samples are drawn with each data point in batch.

        :param data: training batch
        :param ksamples: number of samples for reweighting
        :param kmax_parallel: max number of samples to run in parallel (for lower memory footprint)
        :param train: actually modifiy weights / apply gradients (as this function also returns likelihood)
        :param wakePhaseQ: use also wake phase for learning reconstruction model Q
        :return: log likelihood of data in the generative model
        '''

        nthrun = 0
        left = ksamples
        logprobP_total = None
        logprobQ_total = None
        while left > 0:
            take = min(kmax_parallel, left)
            left -= take
            shape = list(data.shape)
            shape_exp = [take] + shape
            shape[0] *= take  # data is expanded to ksamples*batchsize in dim 0
            # print("Nth Run {}, Take: {}".format(nthrun, take))
            nthrun+=1
            # sample upward pass q(h | x)
            dataExp = data.expand(shape_exp).transpose(0,1).reshape(shape)
            samplesUp, meansUp, logprobQ, logprobQ_total_take = self.sampleQ(dataExp)
            #
            logprobP, logprobP_total_take = self.logprobP(samplesUp)

            logprobP_total_take = logprobP_total_take.reshape((-1, take))
            logprobQ_total_take = logprobQ_total_take.reshape((-1, take))
            if logprobP_total is None:
                logprobP_total = logprobP_total_take.detach()
                logprobQ_total = logprobQ_total_take.detach()
            else:
                logprobP_total = torch.cat([logprobP_total, logprobP_total_take.detach()], dim=1)
                logprobQ_total = torch.cat([logprobQ_total, logprobQ_total_take.detach()], dim=1)
            # loglikelihood

            # calculate sampling weights
            if train:
                nlayers = len(self.layers)-1
                logPQ = (logprobP_total_take - logprobQ_total_take - np.log(take))
                wnorm = logsumexp(logPQ, dim=1)
                logw = logPQ - wnorm.reshape(-1, 1)
                w = torch.exp(logw).flatten().reshape(-1,1)
                # downward pass, taking same batch size
                samplesDown = [None]*nlayers + [self.layers[nlayers].sample_cond(N=data.shape[0])]
                meansDown = [None]*nlayers + [self.layers[nlayers].mean_cond(N=data.shape[0])]
                for i in reversed(range(nlayers)):
                    intterm = self.intsDown[i].gradOutput(self.layers[i].transform(samplesUp[i+1]))
                    mean = self.layers[i].mean_cond(interaction=intterm)
                    samp = self.layers[i].sample_cond(interaction=intterm)
                    samplesDown[i] = samp
                    meansDown[i] = mean
                # add stochastic batch gradients, ksamples needed because of internal normalziation
                for i in range(len(self.layers)-1):
                    self.layers[i].backward(samplesUp[i] - meansDown[i], factor=-w.view(-1, *([1]*(len(meansDown[i].shape)-1)))*take)
                for i in range(len(self.layers)-1):
                    self.intsDown[i].backward(self.layers[i+1].transform(samplesUp[i+1]),
                                              self.layers[i].transform(samplesUp[i]),
                                              factor=-w*take)
                    self.intsDown[i].backward(self.layers[i+1].transform(samplesUp[i+1]),
                                              self.layers[i].transform(meansDown[i]),
                                              factor=w*take)

        logPX = logsumexp(logprobP_total - logprobQ_total, dim=1) - np.log(ksamples)

        return logPX

    def sleepPhase(self,
                   N: int=1,
                   train: bool=False
                   ) -> torch.Tensor:
        '''
        Learning Q in the sleep phase, generating samples.

        :param N: number of samples to generate
        :param train: actually train weights
        :return: (samples, means) N samples and their means generating downwards
        '''
        nlayers = len(self.layers)-1
        samplesDown = [None]*nlayers + [self.layers[nlayers].sample_cond(N=N)]
        meansDown = [None]*nlayers + [self.layers[nlayers].mean_cond(N=N)]
        # downward pass
        for i in reversed(range(nlayers)):
            intterm = self.intsDown[i].gradOutput(self.layers[i+1].transform(samplesDown[i+1]))
            mean = self.layers[i].mean_cond(interaction=intterm)
            samp = self.layers[i].sample_cond(interaction=intterm)
            samplesDown[i] = samp
            meansDown[i] = mean

        # upward pass
        samplesUp = [None]*(nlayers+1)
        meansUp = [None]*(nlayers+1)
        for i in range(nlayers):
            intterm = self.intsUp[i].gradOutput(self.layers[i].transform(samplesDown[i]))
            mean = self.layers[i+1].mean_cond(interaction=intterm)
            samp = self.layers[i+1].sample_cond(interaction=intterm)
            samplesUp[i+1] = samp
            meansUp[i+1] = mean
        # add stochastic batch gradients
        if train:
            for i in range(1, len(self.layers)):
                self.layers[i].backward(samplesDown[i] - meansUp[i], factor=-1)
            for i in range(len(self.layers)-1):
                self.intsUp[i].backward(self.layers[i].transform(samplesDown[i]),
                                        self.layers[i+1].transform(samplesDown[i+1]),
                                        factor=-1)
                self.intsUp[i].backward(self.layers[i].transform(samplesDown[i]),
                                        self.layers[i+1].transform(meansUp[i+1]),
                                        factor=1)
        # self.interaction.backward()

        return samplesDown, meansDown

    def trainReweightedWS(self,
                          data: torch.Tensor,
                          ksamples: int = 1,
                          sleepPhaseQ: bool = True,
                          wakePhaseQ: bool = False
                          ) -> torch.Tensor:
        '''
        Reweighted Wake-Sleep following https://arxiv.org/pdf/1406.2751.pdf

        :param data: training batch
        :param ksamples: number of samples for reweighting
        :param sleepPhaseQ: use sleep phase for learning Q
        :param wakePhaseQ: use wake phase for learning Q
        :return: (estimated) loglikelihood of data
        '''

        self.zero_grad()
        loglik = self.wakePhaseReweighted(data, ksamples=ksamples, train=True, wakePhaseQ=wakePhaseQ)
        if sleepPhaseQ:
            self.sleepPhase(N=data.shape[0], train=True)
        self.optim.step()
        return loglik

    def trainWS(self,
                data: torch.Tensor
                ) -> torch.Tensor:
        '''
        Traditional wake sleep-algorithm, using only one sample (no reweighting)
        and no wake phase Q learning.
        :param data: training data batch
        '''
        return self.trainReweightedWS(data, ksamples=1, sleepPhaseQ=True, wakePhaseQ=False)

    def loglikelihood(self,
                      data: torch.Tensor,
                      ksamples: int=1,
                      kmax_parallel: int=1000
                      ) -> torch.Tensor:
        '''
        Estimate log likelihood as a byproduct of reweighting.

        :param data: data batch
        :param ksamples: number of reweighting samples
        :param kmax_parallel: maximal number of parallel samples (memory footprint)
        :return: loglikelihood of each batch sample
        '''
        return self.wakePhaseReweighted(data, ksamples=ksamples, kmax_parallel=kmax_parallel, train=False)

    def sampleAll(self,
                  N: int=1
                  )-> Tuple[List[torch.Tensor],
                            List[torch.Tensor],
                            List[torch.Tensor],
                            torch.Tensor]:
        '''
        Sample all layers from generative P, (list of samples).

        :param N: number of samples
        :return: batch of N generated data samples and their means for each layer
        '''
        return self.sleepPhase(N=N, train=False)

    def sample(self,
               N: int = 1
               ) -> torch.Tensor:
        '''
        Sample only visible layer from generative P.

        :param N: number of samples
        :return: batch of N generated data samples
        '''
        return self.sleepPhase(N=N, train=False)[0][0]
