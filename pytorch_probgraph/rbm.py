import torch
import numpy as np
from itertools import chain
import matplotlib.pyplot as plt
import torch.nn as nn
from .unitlayer import UnitLayer
from .interaction import Interaction
from typing import Iterable, Union, Optional, List
from tqdm import tqdm

class RestrictedBoltzmannMachine(nn.Module):
    '''
    A two layer undirected hierarchical probabilistic model

    Sources:

    [1] R. Salakhutdinov "LEARNING DEEP GENERATIVE MODELS",
    https://tspace.library.utoronto.ca/bitstream/1807/19226/3/Salakhutdinov_Ruslan_R_200910_PhD_thesis.pdf

    [2] G. Hinton "A Practical Guide to Training Restricted Boltzmann Machines", 2010
    https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
    '''

    def __init__(self,
                 visible: UnitLayer,
                 hidden: UnitLayer,
                 interaction: Interaction):
        '''
        :param visible: [UnitLayer], visible layer
        :param hidden: [UnitLayer], hidden layer
        :param interaction: [Interaction], the interaction unit
        '''

        super(RestrictedBoltzmannMachine, self).__init__()
        self.add_module('visible', visible)
        self.add_module('hidden', hidden)
        self.add_module('interaction', interaction)

    def get_visible_layer(self):
        '''
        :return: visible layer
        '''
        return self.visible

    def get_hidden_layer(self):
        '''
        :return: hidden layer
        '''
        return self.hidden

    def get_interaction(self):
        '''
        :return: interaction
        '''
        return self.interaction

    def logprob_joint(self,
                      visible: torch.Tensor,
                      hidden: torch.Tensor
                      ) -> torch.Tensor:
        '''
        Computes the unnormalized log probability of the RBM state

        :param visible: visible unit activations
        :param hidden: hidden unit activations
        :return: unnormalized log probability
        '''

        # transform activations according the the layer (can be the identity for linear layer)
        intvisible = self.visible.transform(visible)
        inthidden = self.hidden.transform(hidden)

        # get the joint energy from the interaction term
        joint_energy = self.interaction.negenergy(intvisible, inthidden)
        # get the joint energy from the layer term (unnormalized log probability contribution of the layers)
        lpv = self.visible.logprob_joint(visible)
        lph = self.hidden.logprob_joint(hidden)
        return joint_energy + lpv + lph

    def free_energy(self,
                    visible: torch.Tensor
                    ) -> torch.Tensor:
        '''
        The free energy is computed as:
        $$ F = - ln Z = - ln ( \sum_{h} \exp^{ - E(h, v) } ) $$
        (see, Hinton: A Practical Guide to Training Restricted Boltzmann Machines)
        It can be used to track the learning process and overfitting.
        Note that here an averaged free energy is used.

        :param visible: visible units activation
        :return: unnormalized log probability
        '''
        free_en = -self.visible.logprob_joint(visible)
        hidden_int = self.interaction.gradOutput(self.visible.transform(visible))
        free_en += self.hidden.free_energy(hidden_int)
        return free_en

    def mean_visible(self,
                     hidden: torch.Tensor,
                     interaction_factor: Union[torch.Tensor, float]=1.
                     ) -> torch.Tensor:
        '''
        Get the mean of the visible conditional distribution
        If the factor is set to 0, the RBM follows a factorized distribution.

        :param hidden: hidden activation
        :param interaction_factor: a factor to weight the interaction energy.
        :return: mean of the visible activation
        '''
        hidden_int = self.hidden.transform(hidden)
        visible_int = interaction_factor*self.interaction.gradInput(hidden_int)
        return self.visible.mean_cond(visible_int)

    def mean_hidden(self,
                    visible: torch.Tensor,
                    interaction_factor: Union[torch.Tensor, float]=1.
                    ) -> torch.Tensor:
        '''
        Get the mean of the hidden conditional distribution
        If the factor is set to 0, the RBM follows a factorizing distribution.

        :param visible: visible activation
        :param interaction_factor: a factor to weight the interaction energy.
        :return: mean of hidden activation
        '''
        visible_int = self.visible.transform(visible)
        hidden_int = interaction_factor*self.interaction.gradOutput(visible_int)
        return self.hidden.mean_cond(hidden_int)

    def sample_visible(self,
                       hidden: Union[torch.Tensor, None]=None,
                       N: int=1,
                       interaction_factor: Union[torch.Tensor, float]=1.
                       ) -> torch.Tensor:
        '''
        Sample from the visible conditional distribution.
        If the factor is set to 0, the RBM follows a factorizing distribution.

        :param hidden: hidden activation
        :param N: batch size
        :param interaction_factor: a factor to weight the interaction energy.
        :return: visible activation according to the probability distribution
        '''

        # if hidden is none, the visible state is randomly sampled
        if hidden is None:
            return self.visible.sample_cond(N=N)
        else:
            hidden_int = self.hidden.transform(hidden)
            interaction_hidden = interaction_factor*self.interaction.gradInput(hidden_int)
            return self.visible.sample_cond(interaction_hidden)

    def sample_hidden(self,
                      visible: Union[torch.Tensor, None]=None,
                      N: int=1,
                      interaction_factor: Union[torch.Tensor, float]=1.
                      ) -> torch.Tensor:
        '''
        Sample from the hidden conditional distribution
        If the factor is set to 0, the RBM follows a factorizing distribution.

        :param visible: visible activation
        :param N: batch size
        :param interaction_factor: a factor to weight the interaction energy.
        :return: hidden activation according to the probability distribution
        '''
        if visible is None:
            return self.hidden.sample_cond(N=N)
        else:
            visible_int = self.visible.transform(visible)
            interaction_visible = interaction_factor*self.interaction.gradOutput(visible_int)
            return self.hidden.sample_cond(interaction_visible)

    def reconstruct(self,
                    N: int=1,
                    visible_input: Union[torch.Tensor, None]=None,
                    gibbs_steps: int=1,
                    visible_interaction_factor: Union[torch.Tensor, float]=1.,
                    hidden_interaction_factor: Union[torch.Tensor, float]=1.,
                    mean: bool=False,
                    all_states: bool=False
                    ) -> torch.Tensor:
        '''
        Take N visible samples of an RBM, either preconditioned or
        by Gibbs sampling with random start

        :param N: batch size
        :param visible_input: visible activation
        :param gibbs_steps: number of Gibbs steps
        :param visible_interaction_factor: a factor to weight the visible layer energy.
        :param hidden_interaction_factor: a factor to weight the hidden layer energy.
        :param mean: if true, returns the mean activation of the visible units, \
        else returns a sample
        :param all_states: if true, returns a sample for both the hidden and the \
        visible layer, else returns a sample of the visible layer
        :return: sample according to the RBM probability distribution
        '''
        if visible_input is not None:
            hidden_mean = self.mean_hidden(visible_input, interaction_factor=hidden_interaction_factor)
            hidden_sample = self.sample_hidden(visible_input, interaction_factor=hidden_interaction_factor)
        else:
            hidden_sample = self.sample_hidden(N=N)
        for _ in range(gibbs_steps):
            visible_mean = self.mean_visible(hidden_sample, interaction_factor=visible_interaction_factor)
            visible_sample = self.sample_visible(hidden_sample, interaction_factor=visible_interaction_factor)
            hidden_mean = self.mean_hidden(visible_sample, interaction_factor=hidden_interaction_factor)
            hidden_sample = self.sample_hidden(visible_sample, interaction_factor=hidden_interaction_factor)
        if all_states:
            if mean:
              return visible_mean, hidden_mean
            else:
              return visible_sample, hidden_sample
        else:
            if mean:
                return visible_mean
            else:
                return visible_sample

    def train(self,
              data: Iterable[torch.Tensor],
              epochs: int,
              optimizer: torch.optim.Optimizer,
              scheduler = None,
              visible_interaction_factor: Union[torch.Tensor, float]=1.,
              hidden_interaction_factor: Union[torch.Tensor, float]=1.,
              device: Union[torch.device, None]=None):
        '''
        Training method for the RBM

        :param data: training data
        :param epochs: number of epochs
        :param optimizer: torch optimizer
        :param scheduler: torch scheduler
        :param visible_interaction_factor: factor for the visible activation units
        :param hidden_interaction_factor: factor for the hidden activation units
        :return: None
        '''
        for epoch in range(epochs):
            for j, bdat in enumerate(data):
                self.zero_grad()
                self.step(bdat.to(device),
                          visible_interaction_factor=visible_interaction_factor,
                          hidden_interaction_factor=hidden_interaction_factor)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            if isinstance(data, tqdm):
                data = tqdm(data)
                data.set_description('Epoch {}'.format(epoch))

    def plot_reconstruction(self,
                            X_rec : torch.Tensor,
                            X_train : torch.Tensor):
        '''
        Plots the reconstruction

        :param X_rec: reconstructed image
        :param X_train: original image
        :return: None
        '''

        mesh_size = int(np.sqrt(X_rec.size()[1]))
        plt.subplot(2, 1, 1)
        plt.imshow(X_rec.detach().cpu().numpy().reshape([mesh_size, mesh_size]), cmap='gray', interpolation='none')
        plt.subplot(2, 1, 2)
        plt.imshow(X_train[0].detach().cpu().numpy().reshape([mesh_size, mesh_size]), cmap='gray', interpolation='none')
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def plot_reconstruction_list(self,
                                 pictures: List[torch.Tensor]):
        '''
        Plots a list of reconstructed images

        :param pictures: list of images
        :return: None
        '''

        mesh_size = int(np.sqrt(pictures[0].size()[1]))
        fig, axs = plt.subplots(4, int(len(pictures)/4) + 1)
        for i in range(len(pictures)):
            j = i % 4
            k = int(i/4)
            axs[j, k].imshow(pictures[i].detach().cpu().numpy().reshape([mesh_size, mesh_size]), cmap='gray', interpolation='none')
        plt.show()

class RestrictedBoltzmannMachineCD_Smooth(RestrictedBoltzmannMachine):
    '''
    This class handles the training of the RBM with CD. Also the last sampled
    hidden states for the negative and positive phases are mean activations.
    '''
    def __init__(self,
                 visible: UnitLayer,
                 hidden: UnitLayer,
                 interaction: Interaction,
                 ksteps: int=1):
        '''
        :param visible: visible UnitLayer
        :param hidden: hidden UnitLayer
        :param interaction: Interaction
        :param ksteps: number of Gibbs steps
        '''
        super().__init__(visible, hidden, interaction)
        self.ksteps = ksteps

    def step(self,
             data : torch.Tensor,
             visible_interaction_factor: Union[torch.Tensor, float]=1.,
             hidden_interaction_factor: Union[torch.Tensor, float]=1.
             ) -> None:
        '''
        One update step for a batch

        :param data: batch data
        :param visible_interaction_factor: factor for the visible activation units
        :param hidden_interaction_factor: factor for the hidden activation units
        :return:
        '''
        visible_sample = data.clone()
        hidden_mean = self.mean_hidden(visible_sample, interaction_factor=hidden_interaction_factor)
        hidden_sample = self.sample_hidden(visible_sample, interaction_factor=visible_interaction_factor)
        hidden_mean_positive = hidden_mean.clone()
        # Do gibbs sampling for ksteps
        for _ in range(self.ksteps):
            visible_mean = self.mean_visible(hidden_sample, interaction_factor=visible_interaction_factor)
            visible_sample = self.sample_visible(hidden_sample, interaction_factor=visible_interaction_factor)
            hidden_mean = self.mean_hidden(visible_sample, interaction_factor=hidden_interaction_factor)
            hidden_sample = self.sample_hidden(visible_sample, interaction_factor=hidden_interaction_factor)

        # pos phase, biases first, then interaction weights
        # we are using negative gradients here because of gradient descent
        # replaced by ascent
        self.visible.backward(data, factor=-1)
        # the mean activation for the hidden units are used
        self.hidden.backward(hidden_mean_positive, factor=-1)
        self.interaction.backward(self.visible.transform(data),
                                  self.hidden.transform(hidden_mean_positive),
                                  factor=-1)

        # neg phase, biases first, then interaction weights
        self.visible.backward(visible_mean, factor=1)
        self.hidden.backward(hidden_mean, factor=1)
        self.interaction.backward(self.visible.transform(visible_mean),
                                  self.visible.transform(hidden_mean),
                                  factor=1)

class RestrictedBoltzmannMachineCD(RestrictedBoltzmannMachine):
    '''
    This class handles the training of the RBM with CD. Also the last sampled hidden states for the negative and
    positive phases are mean activations.
    '''
    def __init__(self,
                 visible: UnitLayer,
                 hidden: UnitLayer,
                 interaction: Interaction,
                 ksteps: int=1):
        '''
        :param visible: visible layer
        :param hidden: hidden layer
        :param interaction: interaction
        :param ksteps: number of Gibbs steps
        '''
        super().__init__(visible, hidden, interaction)
        self.ksteps = ksteps

    def step(self,
             data: torch.Tensor,
             visible_interaction_factor: Union[torch.Tensor, float]=1.,
             hidden_interaction_factor: Union[torch.Tensor, float]=1.
             ) -> None:
        '''
        One update step for a batch

        :param data: batch data
        :param visible_interaction_factor: factor for the visible activation units
        :param hidden_interaction_factor: factor for the hidden activation units
        :return:
        '''
        visible_sample = data
        hidden_mean = self.mean_hidden(visible_sample, interaction_factor=hidden_interaction_factor)
        hidden_sample = self.sample_hidden(visible_sample, interaction_factor=hidden_interaction_factor)
        hidden_sample_positive = hidden_sample

        # Do gibbs sampling for ksteps
        for _ in range(self.ksteps):
            visible_mean = self.mean_visible(hidden_sample, interaction_factor=visible_interaction_factor)
            visible_sample = self.sample_visible(hidden_sample, interaction_factor=visible_interaction_factor)

            hidden_mean = self.mean_hidden(visible_sample, interaction_factor=hidden_interaction_factor)
            hidden_sample = self.sample_hidden(visible_sample, interaction_factor=hidden_interaction_factor)

        # pos phase, biases first, then interaction weights
        # we are using negative gradients here because of gradient descent
        # replaced by ascent
        trans_dat = self.visible.transform(data)
        trans_dat.require_grad = True
        trans_hidden = self.hidden.transform(hidden_sample_positive)
        trans_hidden.require_grad = True

        self.visible.backward(data, factor=-1)
        self.hidden.backward(hidden_sample_positive, factor=-1)
        self.interaction.backward(trans_dat,
                                  trans_hidden,
                                  factor=-1)

        # neg phase, biases first, then interaction weights
        self.visible.backward(visible_sample, factor=1)
        self.hidden.backward(hidden_sample, factor=1)
        self.interaction.backward(self.visible.transform(visible_sample),
                                  self.hidden.transform(hidden_sample),
                                  factor=1)

class RestrictedBoltzmannMachinePCD(RestrictedBoltzmannMachine):
    '''
    This class handles the training of the RBM with PCD.
    '''
    def __init__(self,
                 visible: UnitLayer,
                 hidden: UnitLayer,
                 interaction: Interaction,
                 fantasy_particles: int=1):
        super().__init__(visible, hidden, interaction)
        '''
        :param visible: visible layer
        :param hidden: hidden layer
        :param interaction: interaction
        :param ksteps: number of Gibbs steps
        '''
        self.register_parameter("visible_fantasy", torch.nn.Parameter(self.sample_visible(N=fantasy_particles), requires_grad = False))

    def step(self,
             data: torch.Tensor,
             visible_interaction_factor: Union[torch.Tensor, float]=1.,
             hidden_interaction_factor: Union[torch.Tensor, float]=1.
             ) -> None:
        '''
        One update step for a batch

        :param data: batch data
        :param visible_interaction_factor: factor for the visible activation units
        :param hidden_interaction_factor: factor for the hidden activation units
        :return:
        '''
        with torch.no_grad():
            # Sample hidden from visible data
            visible_sample = data
            hidden_mean = self.mean_hidden(visible_sample, interaction_factor=hidden_interaction_factor)
            # hidden_sample = self.rbm.sample_hidden(hidden_mean)
            hidden_mean_positive = hidden_mean
            # Sample hidden from visible "fantasy particles"
            visible_sample_neg = self.visible_fantasy
            hidden_mean_neg = self.mean_hidden(self.visible_fantasy, interaction_factor=hidden_interaction_factor)
            hidden_sample_neg = self.sample_hidden(self.visible_fantasy, interaction_factor=hidden_interaction_factor)
            visible_mean_neg = self.mean_visible(hidden_sample_neg, interaction_factor=visible_interaction_factor)
            self.visible_fantasy = torch.nn.Parameter(self.sample_visible(hidden_sample_neg, interaction_factor=visible_interaction_factor), requires_grad = False)
            #self.visible_fantasy = self.visible_fantasy.detach() # don't propagate autograd through everything

        # pos phase, biases first, then interaction weights
        # we are using negative gradients here because of gradient descent
        # replaced by ascent
        self.visible.backward(data, factor=-1)
        self.hidden.backward(hidden_mean_positive, factor=-1)
        self.interaction.backward(self.visible.transform(data),
                                  self.hidden.transform(hidden_mean_positive),
                                  factor=-1)

        # negative phase using fantasy particles
        self.visible.backward(visible_sample_neg, factor=1)
        self.hidden.backward(hidden_mean_neg, factor=1)
        self.interaction.backward(self.visible.transform(visible_sample_neg),
                                  self.hidden.transform(hidden_mean_neg),
                                  factor=1)
