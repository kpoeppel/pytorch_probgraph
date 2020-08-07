'''
A module describing a layer of units and a bias energy (in terms of
probabilistic energy terms).
'''
import torch
from typing import Tuple, Union
from numpy import pi
import numpy as np

epsilon = 1e-6

class UnitLayer(torch.nn.Module):
    '''
    Abstract Class for representing layers of random variables of various shape.
    '''
    def __init__(self):
        super().__init__()

    def logprob_cond(self,
                     input: torch.Tensor,
                     interaction: torch.Tensor
                     ) -> torch.Tensor:
        '''
        Returns the conditional logprobability of a sample input given some
        interaction term of the same shape.

        :param input: the input sample/batch
        :param interaction: the exponential interaction term
        :return: the logprobability of an input given some interaction
        '''
        return NotImplementedError

    def sample_cond(self,
                    interaction: torch.Tensor=None,
                    N: int=1
                    ) -> torch.Tensor:
        '''
        Samples from the conditional probability given some interaction term.
        With A being the current unit this is the term B in
        the energy exponential :math:`e^{transf(A) * B}`.

        :param interaction: the exponential interaction term (None == 0)
        :param N: the number of samples to be drawn in case interaction == None
        :return: batch of samples either of size of the interaction batch or N
        '''
        return NotImplementedError

    def mean_cond(self,
                  interaction: torch.Tensor=None,
                  N: int=1
                  ) -> torch.Tensor:
        '''
        Returns the mean of the conditional probability given some interaction tensor.

        :param interaction: the exponential interaction term (None == 0)
        :param N: number of batch copies in case interaction == None
        :return: batch of means either of size of the interaction batch or N
        '''
        return NotImplementedError

    def transform(self,
                  input: torch.Tensor
                  ) -> torch.Tensor:
        '''
        Transforms a value such that the result leads to a linear interaction.
        Random variable x :math:`\\rightarrow` exponential family term :math:`e^{transf(x) * y}`

        :param input: input data to be transformed
        :return: transformed input
        '''
        return NotImplementedError

    def transform_invert(self,
                         transformed_input: torch.Tensor
                         ) -> torch.Tensor:
        '''
        Transforms a value such that the result leads to a linear interaction.
        Random variable exponential family term :math:`e^{x' * y} \\rightarrow x = transinv(x')`

        :param transformed_input: Some transformed variable(s)
        :return: The original (batch of) variable(s) x
        '''
        return NotImplementedError

    def logprob_joint(self,
                      input : torch.Tensor
                      ) -> torch.Tensor:
        '''
        Returns an unnormalized probability weight given some input.
        This is only an unnormalized probability since there are not interactions.
        An interaction can simply be added by + input @ interaction

        :param input: some variable samples
        :return: their unnormalized loglikelihood (no interaction terms only bias)
        '''
        return NotImplementedError

    def free_energy(self,
                    interaction: torch.Tensor
                    ) -> torch.Tensor:
        '''
        Computes the partition sum given an interaction term.
        Example: binary

        .. math::

            -\log(1 + e^{bias + interaction})

        :param interaction: interaction term
        :return: partition sum / normalizing factor of Gibbs distribution
        '''
        return NotImplementedError

    def backward(self,
                 input: torch.Tensor,
                 factor: Union[torch.Tensor, float]=1.
                 ) -> None:
        '''
        Computes the gradient of the internal parameters wrt to the input data.
        :param input: input data
        :return: None
        '''
        pass

class BernoulliLayer(UnitLayer):
    '''
    A UnitLayer of bernoulli units modelled with probabilities as a sigmoid.
    '''
    def __init__(self,
                 bias: torch.Tensor
                 ) -> torch.Tensor:
        '''
        :param bias: Bias for the sigmoid modelling the bernoulli probability.
        '''
        super().__init__()
        self.register_parameter("bias", torch.nn.Parameter(bias))

    def mean_cond(self,
                  interaction: Union[torch.Tensor, None] = None,
                  N: int=1
                  ) -> torch.Tensor:
        if interaction is not None:
            weight = self.bias + interaction
        else:
            weight = torch.zeros([N] + list(self.bias.shape[1:]), device=self.bias.device) + self.bias
        return torch.sigmoid(weight)

    def sample_cond(self,
                    interaction: Union[torch.Tensor, None]=None,
                    N: int=1
                    ) -> torch.Tensor:
        return torch.bernoulli(self.mean_cond(interaction=interaction, N=N))

    def transform(self,
                  input: torch.Tensor
                  ) -> torch.Tensor:
        return input

    def transform_invert(self,
                         transformed_input: torch.Tensor
                         ) -> torch.Tensor:
        return transformed_input

    def logprob_cond(self,
                     input: torch.Tensor,
                     interaction: Union[torch.Tensor, float]=0.
                     ) -> torch.Tensor:
        newdim = list(range(1, len(input.shape)))
        return torch.sum(torch.log(input*torch.sigmoid(interaction + self.bias) + (1.-input)*(1.-torch.sigmoid(interaction + self.bias))), dim=newdim)

    def logprob_joint(self,
                      input: torch.Tensor
                      ) -> torch.Tensor:
        return torch.sum(input * self.bias, dim=list(range(1, len(input.shape))))

    def free_energy(self,
                    interaction: torch.Tensor
                    ) -> torch.Tensor:
        return -torch.sum(torch.log1p(torch.exp(self.bias + interaction)), dim=list(range(1, len(interaction.shape))))

    def backward(self,
                 input: torch.Tensor,
                 factor: Union[torch.Tensor, float]=1.
                 ) -> None:
        if self.bias.requires_grad:
            self.bias.backward((factor*input).sum(dim=0, keepdim=True).detach()/input.shape[0])


class GaussianLayer(UnitLayer):
    '''
    A UnitLayer of Gaussian distributed variables.
    A layer with :math:`-\\frac{x^2}{2 \sigma^2} + x (bias + interaction)` energy.
    '''
    def __init__(self,
                 bias: torch.Tensor,
                 logsigma: torch.Tensor):
        '''
        :param bias: bias for the Gaussian
        :param logsigma: logarithm of the standard deviation sigma
        '''
        super().__init__()
        self.register_parameter("bias", torch.nn.Parameter(bias))
        self.register_parameter("logsigma", torch.nn.Parameter(logsigma))

    def transform(self,
                  input: torch.Tensor
                  ) -> torch.Tensor:
        return input

    def transform_invert(self,
                         transformed_input: torch.Tensor
                         ) -> torch.Tensor:
        return transformed_input

    def mean_cond(self,
                  interaction: Union[torch.Tensor, None] = None,
                  N: int=1
                  ) -> torch.Tensor:
        if interaction is not None:
            return torch.exp(self.logsigma*2)*(interaction + self.bias)
        else:
            return torch.exp(self.logsigma*2)*self.bias.expand(*([N] + list(self.bias.shape[1:])))

    def sample_cond(self,
                    interaction: Union[torch.Tensor, None]=None,
                    N: int=1
                    ) -> torch.Tensor:
        mean = self.mean_cond(interaction=interaction, N=N)
        return mean + torch.normal(torch.zeros_like(mean), torch.ones_like(mean))*torch.exp(self.logsigma)

    def free_energy(self,
                    interaction: torch.Tensor
                    ) -> torch.Tensor:
        return -torch.sum(0.5*(interaction + self.bias)*torch.exp(2*self.logsigma), dim=list(range(1, len(interaction.shape))))

    def logprob_cond(self,
                     input: torch.Tensor,
                     interaction: Union[torch.Tensor, float]=0.
                     ) -> torch.Tensor:
        norm = -0.5*torch.log(2*pi*torch.ones_like(self.logsigma)).sum() - torch.sum(self.logsigma)
        exp =  input * (interaction + self.bias) - (input)**2/2/torch.exp(2*self.logsigma)
        return norm + exp.sum(dim=list(range(1, len(exp.shape))))

    def logprob_joint(self,
                      input: torch.Tensor
                      ) -> torch.Tensor:
        norm = -0.5*torch.log(2*pi*torch.ones_like(self.logsigma)).sum() - self.logsigma.sum()
        exp = -input**2/2/torch.exp(self.logsigma*2) + input*self.bias - 0.5*torch.log(2 * pi) - 2*self.logsigma
        return exp.sum(dim=list(range(1, len(exp.shape)))) + norm

    def backward(self,
                 input: torch.Tensor,
                 factor: Union[torch.Tensor, float]=1.
                 ) -> None:
        if self.bias.requires_grad:
            grad_bias = (factor*input).sum(dim=0, keepdim=True) / input.shape[0]
            self.bias.backward(grad_bias.detach())
        if self.logsigma.requires_grad:
            var = (input**2).sum(dim=0, keepdim=True)/input.shape[0]
            grad_logsigma = factor* var* torch.exp(-2*self.logsigma) - 1.
            self.logsigma.backward(grad_logsigma.detach())


class DiracDeltaLayer(UnitLayer):
    '''
    A Layer to model a dirac delta == copying input-interaction to output
    '''
    def __init__(self, base_shape=(1,), deltafactor=1000., device=None):
        '''
        :param base_shape: shape of the variables (1st is batch dimension)
        :param deltafactor: the inverse variance of the approximating gaussian
        :param device: the device to operate on
        '''
        super().__init__()
        self.deltafactor = deltafactor
        self.base_shape = base_shape
        self.device=device

    def free_energy(self,
                    interaction: torch.Tensor
                    ) -> torch.Tensor:
        return 0.

    def backward(self,
                 input: torch.Tensor,
                 factor: Union[torch.Tensor, float]=1.
                 ) -> None:
        pass

    def transform(self,
                  input: torch.Tensor
                  ) -> torch.Tensor:
        return input

    def logprob_joint(self,
                      input: torch.Tensor
                      ) -> torch.Tensor:
        return torch.zeros([input.shape[0]] + list(self.base_shape[1:]))

    def transform_invert(self,
                         transformed_input: torch.Tensor
                         ) -> torch.Tensor:
        return transformed_input

    def logprob_cond(self,
                     input: torch.Tensor,
                     interaction: Union[torch.Tensor, float]=0.
                     ) -> torch.Tensor:
        return -self.deltafactor*(input - interaction)**2

    def mean_cond(self,
                  interaction: Union[torch.Tensor, None] = None,
                  N: int=1
                  ) -> torch.Tensor:
        if interaction is not None:
            return interaction
        else:
            return torch.zeros([N] + list(self.base_shape[1:]))

    def sample_cond(self,
                    interaction: Union[torch.Tensor, None]=None,
                    N: int=1
                    ) -> torch.Tensor:
        if interaction is None:
            return torch.zeros([N] + list(self.base_shape[1:]), device=self.device)
        else:
            return interaction

class CategoricalLayer(UnitLayer):
    '''
    Essentially a multinomial distribution in the last layer as a one-hot encoding.
    The samples are generated via Gumbel softmax samples.
    Mean values via softmax.
    '''
    def __init__(self, bias : torch.Tensor):
        '''
        :param bias: Categorical Bias, Last dim represents categories
        '''
        #if biasIn.shape[:-1] != biasOut.shape:
        #    raise DimensionError('')

        super().__init__()
        self.register_parameter("bias", torch.nn.Parameter(bias))

    def mean_cond(self,
                  interaction: Union[torch.Tensor, None] = None,
                  N: int=1
                  ) -> torch.Tensor:
        logprobs = self.bias.expand(*([N] + list(self.bias.shape[1:])))
        if interaction is not None:
            #print(logprobs.device, interaction.device)
            logprobs = logprobs + interaction
        return torch.softmax(logprobs, dim=-1)

    def sample_cond(self,
                    interaction: Union[torch.Tensor, None]=None,
                    N: int=1
                    ) -> torch.Tensor:
        # print(self.biasIn.shape, self.biasOut.shape)
        shape = self.bias.shape
        logprobs = self.bias.expand(*([N] + list(shape[1:])))
        if interaction is not None:
            logprobs = logprobs + interaction
        return torch.nn.functional.gumbel_softmax(logprobs, dim=-1, hard=True).detach()

    def logprob_cond(self,
                     input: torch.Tensor,
                     interaction: Union[torch.Tensor, float]=0.
                     ) -> torch.Tensor:
        res = torch.zeros(input.shape[:-1], device=input.device)
        acc = input.sum(dim=-1)
        res[acc < 1. + epsilon] = -np.inf
        res[acc > 1. + epsilon] = -np.inf
        logprobs = torch.logsoftmax(self.bias + interaction, dim=-1, keepdim=True)
        logprob = torch.sum(input * logprobs, dim=list(range(1, len(input.shape))))
        logprob += res.sum(dim=list(range(1, len(res))))

    def transform(self,
                  input: torch.Tensor
                  ) -> torch.Tensor:
        return input

    def transform_invert(self,
                         transformed_input: torch.Tensor
                         ) -> torch.Tensor:
        return transformed_input

    def logprob_joint(self,
                      input: torch.Tensor
                      ) -> torch.Tensor:
        logprob = torch.sum(input * self.bias, dim=list(range(1, len(input.shape))))
        res = torch.zeros(input.shape[:-1], device=input.device)
        acc = input.sum(dim=-1)
        res[acc > 1. + epsilon] = -np.inf
        res[acc < 1. - epsilon] = -np.inf
        logprob += res.sum(dim=list(range(1, len(res))))
        return logprob

    def free_energy(self,
                    interaction: torch.Tensor
                    ) -> torch.Tensor:
        logprobs = torch.nn.functional.log_softmax(self.bias + interaction, dim=-1)
        return torch.sum(logprobs, dim=list(range(1, len(interaction))))

    def backward(self,
                 input: torch.Tensor,
                 factor: Union[torch.Tensor, float]=1.
                 ) -> None:
        if self.bias.requires_grad:
            self.bias.backward((input*factor).sum(dim=0, keepdim=True)/input.shape[0])
