import torch
import numpy as np
from tqdm import tqdm
from typing import List, Iterable, Optional
from .rbm import RestrictedBoltzmannMachine
from .utils import ListModule


class DeepBeliefNetwork(torch.nn.Module):
    '''
    From: "On the Quantitative Analysis of Deep Belief Networks" (Salakhutdinov, Murray)
    Using greedy learning on RBMs
    '''
    def __init__(self,
                 rbms: List[RestrictedBoltzmannMachine],
                 optimizer: torch.optim.Optimizer):
        '''
        Pass a list of rbms with fitting input and output sizes
        The optimizer should be an optimizer on all the rbm's parameters.
        :param rbms: [List[RestrictedBoltzmannMachine]] List of RBMs
        :param optimizer: Optimizer to train the RBMs
        '''
        super().__init__()
        self.rbms = ListModule(*rbms)
        self.optimizer = optimizer

    def train_layer(self,
                    rbm_num: int,
                    data: Iterable[torch.Tensor],
                    epochs: int,
                    device: torch.device=None
                    ) -> None:
        '''
        Train an RBM using some data (usually sampled from below)
        '''
        for epoch in range(epochs):
            # set all gradients zero first!
            for rbm in self.rbms:
                rbm.zero_grad()

            for bdat in data:
                self.rbms[rbm_num].zero_grad()
                self.rbms[rbm_num].step(bdat.to(device))
                self.optimizer.step()

            if isinstance(data, tqdm):
                data = tqdm(data)
                data.set_description("Epoch {}, RBM {}".format(epoch, rbm_num))

    def sample_layer_hidden(self,
                            rbm_num: int,
                            visible: Optional[torch.Tensor]=None,
                            N: int=1
                            ) -> torch.Tensor:
        if visible is not None:
            return self.rbms[rbm_num].sample_hidden(visible=visible)
        else:
            return self.rbms[rbm_num].sample_hidden(N=N)

    def train(self,
              data : Iterable[torch.Tensor],
              epochs: int=1,
              skip_rbm: List[int]=list(),
              device: torch.device=None
              ) -> None:
        '''
        Train the Deep Belief Network, data should be an iterator on training
        batches.

        :param data: Iterator of Training Batches
        :param epochs: Number of epochs to learn each Restricted Boltzmann Machine
        :param skip_rbm: Skip Learning for some RBMs (indexes)
        :param device: The torch device to move the data to before training
        '''
        for rbm_num in range(len(self.rbms)):
            if rbm_num not in skip_rbm:
                self.train_layer(rbm_num, data, epochs, device=device)
            newdat = []
            for bdat in data:
                #newdat.append(self.sample_layer_hidden(rbm_num, visible=bdat.to(device)).detach().cpu())
                next_data = self.sample_layer_hidden(rbm_num, visible=bdat.to(device)).detach().cpu()
                next_data.requires_grad = True
                newdat.append(next_data)
            if isinstance(data, tqdm):
                data = tqdm(newdat)
            else:
                data = newdat

    def free_energy_estimate(self,
                             data: torch.Tensor,
                             skip_rbm: List[int]=list(),
                             ) -> torch.Tensor:
        '''
        Calculate the sum of the free energies of the RBMs. Upward RBMs are \
        fed with samples from below.

        :param data: Data batch
        :param skip_rbm: Skip some RBMs (indices)
        '''
        free_energy = 0.
        for rbm_num in range(len(self.rbms)):
            if rbm_num not in skip_rbm:
                free_energy = free_energy + self.rbms[rbm_num].free_energy(data)
            data = self.sample_layer_hidden(rbm_num, visible=data).detach()
        return free_energy


    def sample(self,
               N: int,
               gibbs_steps: int=1
               ) -> torch.Tensor:
        '''
        Sample first from deepest rbm, then sample all the conditionals to visible layer

        :param N: Number of samples (batch size)
        :param gibbs_steps: Number of Gibbs steps to use in each RBM
        '''
        visible_sample = self.rbms[-1].reconstruct(N=N, gibbs_steps=gibbs_steps)
        for i in reversed(range(len(self.rbms)-1)):
            visible_sample = self.rbms[i].sample_visible(visible_sample)
        return visible_sample
