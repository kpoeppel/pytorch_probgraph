import torch.nn as nn
import torch
from typing import List, Union, Optional, Iterable
from tqdm import tqdm
from .rbm import RestrictedBoltzmannMachine

class DeepBoltzmannMachine(nn.Module):
    '''
    | A deep undirected hierarchical probabilistic model
    | From:

    [1] R. Salakhutdinov "LEARNING DEEP GENERATIVE MODELS",
    https://tspace.library.utoronto.ca/bitstream/1807/19226/3/Salakhutdinov_Ruslan_R_200910_PhD_thesis.pdf
    '''

    def __init__(self,
                 rbms: List[RestrictedBoltzmannMachine],
                 optimizer: torch.optim.Optimizer,
                 scheduler):
        '''
        Constructs the DBM

        :param rbms: list of the RBMs, which will be stacked to form the DBM.
        :param optimizer: torch optimizer
        :param scheduler: torch scheduler
        '''
        super().__init__()

        # build a ModuleList containing the RBMs
        self.rbms = nn.ModuleList()
        for rbm in rbms:
            self.rbms.append(rbm)

        # build a list containing all layers
        self.layers = [r.get_visible_layer() for r in self.rbms] + [self.rbms[-1].get_hidden_layer()]

        # build a list containing all interactions
        self.interactions = [r.get_interaction() for r in self.rbms]

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.nlayers = len(self.layers)

    def zero_grad(self) -> None:
        for r in self.rbms:
            r.zero_grad()

    def backward(self,
                 layer_data: List[torch.Tensor],
                 factor: Union[torch.Tensor, float]=1.
                 ) -> None:
        '''
        Given all the layers' data calculate the gradient wrt latent variables

        :param layer_data: list of unit activations of the layers in one batch
        '''

        for i in range(self.nlayers - 1):
            self.layers[i].backward(layer_data[i], factor=factor)
            self.interactions[i].backward(layer_data[i], layer_data[i + 1], factor=factor)
        self.layers[-1].backward(layer_data[-1], factor=factor)

    def conditional_sample_general_meanfield(self,
                                             layer_data: Optional[
                                                            List[
                                                              Optional[
                                                                torch.Tensor]
                                                                ]
                                                            ] = None,
                                             N: int=1,
                                             invtemperature: Union[torch.Tensor, float]=1.,
                                             iterations: int=10,
                                             sample_state: List[bool]=None,
                                             mean: bool=False
                                             ) -> List[torch.Tensor]:
        '''
        Produces a general sample using an ELBO with a mean field approx.
        If the factor is set to 0, the DBM follows a factorizing distribution.

        :param layer_data: list of unit activations of the layers in one batch
        :param N: batch size
        :param invtemperature: a factor to weight the interaction factor.
        :param iterations: number of gibbs steps following mean field approximation
        :param sample_state: list of bools to indicate, if the layer should be sampled or kept during gibbs steps
        :param mean: if false, the returned state is sampled according to the activation probability. \
        If true, the returned state is the activation probability.
        '''

        # if a state is given in layer data, the batch size can be implied
        nbatch = N
        if layer_data is not None:
            for s in layer_data:
                if s is not None:
                    nbatch = s.shape[0]


        # build the starting state
        # for every layer, for which no starting state is defined, sample a new state
        state = [None] * self.nlayers
        if not layer_data == None:
            for i, layer in enumerate(layer_data):
                if layer is not None:
                    state[i] = layer.clone()
                else:
                    state[i] = self.layers[i].sample_cond(N=nbatch)
        else:
            for i in range(self.nlayers):
                state[i] = self.layers[i].sample_cond(N=nbatch)


        # if no explicit starting sample states are given, sample all states, which had no staring state
        if sample_state == None:
            sample_state = [True] * self.nlayers

        # perform the iterative mean field approximation
        for i in range(iterations):
            # update odd layers
            for j in range(1, self.nlayers, 2):
                if sample_state[j]:
                    intterm = self.interactions[j - 1].gradOutput(self.layers[j - 1].transform(state[j - 1]))
                    intterm *= invtemperature
                    if j + 1 < self.nlayers:
                        intterm += invtemperature * self.interactions[j].gradInput(self.layers[j+1].transform(state[j + 1]))
                    if mean:
                        state[j] = self.layers[j].mean_cond(interaction=intterm)
                    else:
                        state[j] = self.layers[j].sample_cond(interaction=intterm)

            # update even layers
            for j in range(0, self.nlayers, 2):
                if sample_state[j]:
                    intterm = 0.
                    if j > 0:
                        intterm += invtemperature * self.interactions[j - 1].gradOutput(self.layers[j-1].transform(state[j - 1]))
                    if j + 1 < self.nlayers:
                        intterm += invtemperature * self.interactions[j].gradInput(self.layers[j+1].transform(state[j + 1]))
                    if mean:
                        state[j] = self.layers[j].mean_cond(interaction=intterm)
                    else:
                        state[j] = self.layers[j].sample_cond(interaction=intterm)
        return state

    def joint_sample(self,
                     N: int=1,
                     iterations: int=10,
                     invtemperature: Union[torch.Tensor, float]=1.,
                     mean: bool=False
                     ) -> List[torch.Tensor]:
        '''
        Gets a joint sample of all layers with batch size N after some Gibbs iterations.
        If the factor is set to 0, the DBM follows a factorizing distribution.

        :param N: [int], batch size
        :param iterations: [int], number of gibbs steps
        :param invtemperature: [float], a factor to weight the interaction factor.
        :param mean: [bool], if false, the returned state is sampled according \
        to the activation probability. If true, the returned state is the \
        activation probability. This can be interpreted as taking the mean \
        over infinitely many samples.
        '''
        return self.conditional_sample_general_meanfield(N=N,
                                                         iterations=iterations,
                                                         invtemperature=invtemperature,
                                                         mean=mean)

    def conditional_sample(self,
                           data: torch.Tensor,
                           iterations: int=10,
                           invtemperature: Union[torch.Tensor, float]=1.
                           ) -> List[torch.Tensor]:
        '''
        Gets a sample of all layers conditioned on given visible data.
        If the invtemperature is set to 0, the DBM follows a factorizing distribution.

        :param data: [torch.tensor(batch_size, image size)], the images in one batch
        :param iterations: [int], number of meanfield approximation steps
        :param invtemperature: [float], a factor to weight the interaction factor.
        :return: [List[torch.Tensor]] conditional sample of all layers
        '''

        # initialize a state of the DBM, where the visible units are according to the data
        # and the hidden units are not initialized
        layer_data = [data] + len(self.rbms) * [None]

        # allow the DBM to sample all states except the visible state
        sample_state = [True] * self.nlayers
        sample_state[0] = False

        # pass the prepared data to the helper function
        return self.conditional_sample_general_meanfield(layer_data=layer_data,
                                                         sample_state=sample_state,
                                                         iterations=iterations,
                                                         invtemperature=invtemperature)

    def ais(self,
            steps: int,
            M: int,
            data: Optional[torch.Tensor]=None,
            log_Z: Optional[torch.Tensor]=None
            ) -> torch.Tensor:
        '''
        Following [1] chapter 4.2.1f.
        AIS has two operation modi. It can either compute the partitioning sum \
        of the DBM, if no data is provided. It can also calculate the \
        probability of a visible state, if data and the partitioning sum \
        are provided.

        :param steps: number of intermediate probability distributions
        :param M: number of samples to take the mean
        :param data: the visible state of the DBM
        :param log_Z: the partitioning sum of the DBM
        :return: Partition sum / Loglikelihood of data
        '''

        with torch.no_grad():

            #stepsize is the change per iteration of the invtemperature
            step_size = 1. / steps
            # beta_k is the current invtemperature
            beta_k = 0

            # if data is provided, don't sample the visible state
            sample_state = [True] * self.nlayers
            if data is not None:
                sample_state[0] = False

            # if no data is provided, initialize with a joint sample of all states, else keep the visible state.
            if data is None:
                state = self.joint_sample(invtemperature=0, iterations=1, N = M)
            else:
                state = self.conditional_sample(data, invtemperature=0, iterations=1)

            log_p_k = None
            # iterativly increase the invtemperature, to change the trivial distribution to the actual.
            for step in range(steps - 1):

                # calculate the unnormalized probability according to the last and the current invtemperature
                log_1 = self.log_free_energy_joint(state, 1)
                log_2 = self.log_free_energy_joint(state, 0)
                log_p_last = beta_k * log_1 + (1 - beta_k) * log_2

                beta_k += step_size


                log_1 = self.log_free_energy_joint(state, 1)
                log_2 = self.log_free_energy_joint(state, 0)

                log_p_curr = beta_k * log_1 + (1 - beta_k) * log_2

                log_p = log_p_curr - log_p_last

                # sample according to the new invtemperature
                state = self.conditional_sample_general_meanfield(state, invtemperature=beta_k, iterations=1,
                                                                  sample_state=sample_state)

                # add the differences of p_k together
                if log_p_k is None:
                    log_p_k = log_p
                else:
                    log_p_k += log_p

            # for numerical stability, subtract the mean before taking the exponent
            normalize = log_p_k.mean()
            exponent = log_p_k - normalize
            w_ais = torch.exp(exponent)

            # ignore samples with underflow/overflow
            w_ais[torch.isinf(w_ais) == 1] = w_ais[torch.isinf(w_ais) != 1].mean()

            # if the partitioning sum has to be computed, the mean over the batch can be taken
            if data is None:
                log_r_ais = torch.log(w_ais.mean()) + normalize
            else:
                log_r_ais = torch.log(w_ais) + normalize

            # for the partitioning sum:
            if data is None:
                num_states = 0

                # calculate the trivial partitioning sum:
                for i in range(0, len(self.layers)):
                    num_states += torch.numel(self.layers[i].bias)
                log_Z_A = num_states * torch.log(torch.tensor(2).float())

                # add the accumulated fractions
                log_Z = log_r_ais + log_Z_A
                return log_Z

            # for the log probability
            else:
                num_states = 0

                # calculate the trivial partitioning sum except for the lowest layer.
                # This is like summing out the not visible units for the trivial distribution.
                for i in range(1, len(self.layers)):
                    num_states += torch.numel(self.layers[i].bias)
                log_Z_A = num_states * torch.log(torch.tensor(2).float())

                # add the accumulated fraction to the trivial distribution and subtract the partitioning sum
                log_p = log_r_ais + log_Z_A - log_Z
                log_p[torch.isinf(log_p) == 1] = log_p[torch.isinf(log_p) != 1].mean()
                return log_p

    def generate(self,
                 N: int=32,
                 gibbs_steps: int=1,
                 mean: bool=True
                 ) -> torch.Tensor:
        '''
        Compatibility function for the generalized evaluataion method. Returns the visible layer of a joint sample.

        :param N: number of samples to generate
        :param gibbs_steps: number of gibbs steps
        :param mean: if false, the returned state is sampled according to the \
        activation probability. If true, the returned state is the \
        activation probability.
        :return: generated samples
        '''
        return self.joint_sample(N=N, mean=mean, iterations=gibbs_steps)[0].reshape([-1,28,28])

    def loglikelihood(self,
                      data: torch.Tensor,
                      log_Z: Optional[torch.Tensor]=None
                      ) -> torch.Tensor:
        '''
        Compute the log-likelihood of a batch of data. In case log_Z
        (estimator of log of partition sum) is known, do not recompute it.

        :param data: data batch
        :param log_Z: known log partition sum estimate
        '''
        if log_Z is None:
            log_Z = self.ais(100,100)
        log_p = self.ais(100, -1, data = data, log_Z = log_Z)
        return log_p

    def log_free_energy_joint(self,
                              state: List[torch.Tensor],
                              invtemperature: Union[torch.Tensor, float]=1.
                              ) -> torch.Tensor:
        '''
        Return an unnormalized logarithmic probability of a hidden+visible
        variable state.
        If the invtemperature is set to 0, the DBM follows a factorizing distribution.

        :param state: The model state to calculate the free energy for
        :param invtemperature: a factor to weight the interaction factor.
        '''

        # the log energy of the DBM can be computed from the energies of the layers and interactions
        # according to [1] chapter 5.3.
        log_en = 0.
        for i in range(self.nlayers):
            log_en += self.layers[i].logprob_joint(state[i])
        for i in range(self.nlayers - 1):
            int_state_in = self.layers[i].transform(state[i])
            int_state_out = self.layers[i+1].transform(state[i+1])
            log_en += invtemperature * self.interactions[i].negenergy(int_state_in, int_state_out)
        return log_en

    def greedy_pretraining(self,
                           data: Iterable[torch.Tensor],
                           epochs: int
                           ) -> None:
        '''
        Implementation of the greedy pretraining algorithm from [1] chapter 5, algorithm 6.
        Nonte, that we do not use pretraining for our evaluation script, as it has a tendency to overfit,
        thus the quality of the results depends heavily on the initialisation.
        Also note, that this implementation can only be used, if every second layer has the same dimension.

        :param data: tqdm object, training data
        :param epochs: [int] number of training epochs per RBM
        '''
        for i in range(len(self.rbms)):
            self.greedy_pretraining_layer(i, data, epochs)
            # the output of the lower RBM becomes the input to the higher
            newdat = []
            for bdat in data:
                newdat.append(self.sample_rbm_layer_hidden(i, bdat))
            data = newdat

    def greedy_pretraining_layer(self,
                                 rbm_num: int,
                                 data: Iterable[torch.Tensor],
                                 epochs: int
                                 ) -> None:
        '''
        A helper function to train the individual RBMs.

        :param rbm_num: [int] the number of the layer in the DBM.
        :param data: [torch.tensor(batch_size, image size)], the images in one batch
        :param epochs: [int] number of training epochs for the RBM
        :return: None
        '''

        # according to the position of the RBM, the missing layer below or above must be compensated.
        if rbm_num == 0:
            visible_interaction_factor = 1.
            hidden_interaction_factor = 2.
        elif rbm_num == len(self.rbms) - 1:
            visible_interaction_factor = 2.
            hidden_interaction_factor = 1.
        else:
            visible_interaction_factor = 1.
            hidden_interaction_factor = 1.

        # weights are mirrored for initialisation.
        if rbm_num > 0:
            self.rbms[rbm_num].get_interaction().weight = self.rbms[rbm_num].get_interaction().weight.T.clone()
        self.rbms[rbm_num].train(data, epochs, self.optimizer,
                                 self.scheduler, visible_interaction_factor,
                                 hidden_interaction_factor)

    def sample_rbm_layer_hidden(self,
                                rbm_num: int,
                                visible: Optional[torch.Tensor]=None,
                                N: int=1
                                ) -> torch.Tensor:
        '''
        A helper function for the greedy pretraining.
        Sample the hidden state of the RBM at position i. If no visible state is given, a random visible state is used.
        :param rbm_num: [int], position of the RBM
        :param visible: [torch.tensor(batch_size, visible size)], the visible state of the RBM
        :param N: [int], if no visible state is given, N must be set to the batch size
        :return: [torch.tensor(batch size, hidden size)], sampled hidden state
        '''

        if visible is not None:
            return self.rbms[rbm_num].sample_hidden(visible=visible)
        else:
            return self.rbms[rbm_num].sample_hidden(N=N)

class DeepBoltzmannMachineLS(DeepBoltzmannMachine):
    '''
    A class handling the training strategy for the DBM
    '''

    def __init__(self,
                 rbms: List[RestrictedBoltzmannMachine],
                 optimizer: torch.optim.Optimizer,
                 scheduler,
                 ksteps: int=1,
                 learning: str = 'CD',
                 nFantasy: int=100):
        '''
        Builds the DBM and initializes the learning strategy
        :param rbms: a list of RBMs
        :param optimizer: torch optimizer
        :param scheduler: torch scheduler
        :param ksteps: number of gibbs steps
        :param learning: the learning strategy
        :param nFantasy: the number of fantasy particles
        '''
        super().__init__(rbms, optimizer, scheduler)
        self.fantasy_state = None
        self.ksteps = ksteps
        self.ls = learning
        self.CD = learning
        self.nFantasy = nFantasy

    def train_model(self,
                    data: Iterable[torch.Tensor],
                    epochs: int=5,
                    device: Optional[torch.device]=None
                    ) -> None:
        '''
        Implements the training module to train the DBM
        :param data: [tqdm object] training data
        :param epochs: number of epochs to train
        :return: None
        '''

        for epoch in range(epochs):
            for i, bdat in enumerate(data):
                if i % 10000 == 0 and isinstance(data, tqdm):
                    self.train_batch(bdat.to(device), iterations=self.ksteps, verbose=True)
                else:
                    self.train_batch(bdat.to(device), iterations=self.ksteps)
            if isinstance(data, tqdm):
                data = tqdm(data)
            #if isinstance(data, tqdm):
            #    data = tqdm(data)

    def train_batch(self,
                    data: torch.Tensor,
                    iterations: int=10,
                    verbose: bool=False):
        '''
        A helper function that takes single batches in trains the model
        :param data: training data batch
        :param iterations: [int], number of gibbs steps
        :param verbose: [bool], if true, training progress is printed.
        :return: None
        '''

        # conditional sample is for the positive phase
        conditional_sample = self.conditional_sample(data, iterations)
        with torch.no_grad():
            if self.CD == 'CD':
                # for CD, a starting state according to the training data is chosen
                starting_state = [None] * self.nlayers
                starting_state[0] = data
                negative_sample = self.conditional_sample_general_meanfield(layer_data= starting_state, iterations=iterations, mean=True)
            elif self.CD == 'PCD':
                # for PCD, a starting state according to the last negative example is chosen
                sample_state = [True] * self.nlayers
                if self.fantasy_state is None:
                    self.fantasy_state = self.conditional_sample_general_meanfield(iterations=iterations,
                                                                                   sample_state = sample_state,
                                                                                   mean=True)
                else:
                    self.fantasy_state = self.conditional_sample_general_meanfield(layer_data=self.fantasy_state,
                                                                                   iterations=iterations,
                                                                                   mean=True,
                                                                                   sample_state = sample_state)
                negative_sample = self.fantasy_state

        self.zero_grad()
        # positive step
        self.backward(conditional_sample, factor=-1)
        # negative step
        self.backward(negative_sample, factor=1)
        # update weights
        self.optimizer.step()
        self.scheduler.step()

        # if verbose, print the training progress
        if verbose:
            #print('learning rate:')
            for param_group in self.optimizer.param_groups:
                print(param_group['lr'])
            conditions = (len(self.rbms) + 1) * [None]
            random_state = []
            #for st in conditional_sample:
                #random_state.append(torch.randn_like(st))
            #unconditional_sample = self.conditional_sample_general_meanfield(conditions,
            #                                                                 iterations=iterations,
            #                                                                 mean=True)
            #self.rbms[0].plot_reconstruction(negative_sample[0][0].reshape([1, -1]),
            #                         unconditional_sample[0][0].reshape([1, -1]))
            log_z = self.ais(100,100)
            print('log_z')
            print(log_z)
            log_p = self.ais(100, -1, data = data, log_Z = log_z)
            print('log_p: ')
            print(log_p.mean())
