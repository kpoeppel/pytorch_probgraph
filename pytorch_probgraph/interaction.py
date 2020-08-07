'''
A Module providing some Interactions between unitlayers (in the sense of
probabilistic energy terms).
'''
from operator import mul
from functools import reduce
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from typing import Union, Tuple
from .utils import ListModule
from .utils import Projection, Expansion1D, Truncation1D
from .utils import Expansion2D, InvertGrayscale

class Interaction(nn.Module):
    '''
    | A General class for Interactions between input and output as an energy. \
    Could be anything even non-linear interactions.

    | E.g. one could use a torch-module all that is needed that things \
    are differentiable for both input and output (energy is something \
    like a negative loss).
    | Note that input and output have to be transformed for some distributions \
    from the Exponential Family \
    (https://en.wikipedia.org/wiki/Exponential_family#Interpretation).
    | For all methods, this transformation is assumed to be done previously.
    '''
    def __init__(self):
        super(Interaction, self).__init__()

    def negenergy(self,
                  input: torch.Tensor,
                  output: torch.Tensor,
                  factor: Union[torch.Tensor, float] = 1.,
                  **kwargs
                  ) -> torch.Tensor:
        '''
        Defines the (negative) interaction energy, e.g. something like
        input @ W @ output for a fully connected interaction,
        with @ being a matrix multiplication.

        :param input: Input values ()
        :param output: Output values
        :param factor: general factor on top (can also be a tensor of batch_dim)
        :param **kwargs: additional optional parameters
        '''
        return 0.

    def gradInput(self,
                  output: torch.Tensor,
                  input: Union[torch.Tensor, None] = None,
                  factor: Union[torch.Tensor, float] = 1.,
                  **kwargs
                  ) -> torch.Tensor:
        '''
        The negenergy gradient wrt the input. Can be calculated analytically or by
        using autograd. For non-linear interactions, the input is needed.

        :param output: the output value to interact with
        :param input: an input value for which the gradient is calculated
        :param factor: general factor on top (can also be a tensor of batch_dim)
        :param **kwargs: additional optional parameters
        :return: gradient of negenergy wrt. input transformed variable
        '''
        return NotImplementedError

    def gradOutput(self,
                   input: torch.Tensor,
                   factor: Union[torch.Tensor, float] = 1.,
                   **kwargs
                   ) -> torch.Tensor:
        '''
        The negenergy gradient wrt the output. Can be calculated analytically or
        by using autograd.

        :param input: the input values
        :param factor: general factor on top (can also be a tensor of batch_dim)
        :param **kwargs: additional optional parameters
        :return: gradient of negenergy wrt. output transformed variable
        '''
        return NotImplementedError

    def backward(self,
                 input: torch.Tensor,
                 output: torch.Tensor,
                 factor: Union[torch.Tensor, float] = 1.,
                 **kwargs
                 ) -> torch.Tensor:
        '''
        The energy gradient wrt the internal parameters. It is added to the
        parameters in their .grad part.
        Only computes the gradient and doesn't return anything.

        :param input: input values
        :param output: output values
        :param factor: general factor on top (can also be a tensor of batch_dim)
        :param **kwargs: additional optional parameters
        :return: None
        '''
        return NotImplementedError


class InteractionLinear(Interaction):
    '''
    A simple linear Interaction.
    '''

    def __init__(self,
                 inputShape: Union[Tuple[int], int] = 1,
                 outputShape: Union[Tuple[int], int] = 1,
                 weight: Union[torch.Tensor, None] = None,
                 dev_factor: float = 1.,
                 batch_norm: bool = False):
        '''
        Init of a InteractionLinear between two layers of inputShape and outputShape.

        :param inputShape: Size or Shape (Tuple of Ints) of input UnitLayer
        :param outputShape: Size or Shape of output UnitLayer
        :param weight: externally define weight matrix
        :param dev_factor: Deviation factor for Xavier initialization
        :param batch_norm: If batch_norm should be computed for negenergy
        '''

        super().__init__()
        self.batch_norm = batch_norm
        if weight is not None:
            self.weight = nn.Parameter(weight)
        else:
            if isinstance(inputShape, int):
                self.inputShape = (inputShape,)
                self.outputShape = (outputShape,)
                self.inputSize = inputShape
                self.outputSize = outputShape
            else:
                self.inputShape = inputShape
                self.outputShape = outputShape
                self.inputSize = reduce(mul, inputShape, 1)
                self.outputSize = reduce(mul, outputShape, 1)
            weight = dev_factor * 6. / (self.inputSize + self.outputSize) * torch.randn([self.inputSize, self.outputSize])
            self.weight = nn.Parameter(weight)


    def negenergy(self,
                  input: torch.Tensor,
                  output: torch.Tensor,
                  factor: Union[torch.Tensor, float] = 1.,
                  **kwargs
                  ) -> torch.Tensor:
        negenergy = factor * (output.reshape(-1, self.outputSize) * (input.reshape(-1, self.inputSize) @ self.weight)).sum(1)
        if self.batch_norm:
            negenergy = negenergy.sum()/ output.shape[0]
        return negenergy

    def gradInput(self,
                  output: torch.Tensor,
                  input: Union[torch.Tensor, None] = None,
                  factor: Union[torch.Tensor, float] = 1.,
                  **kwargs
                  ) -> torch.Tensor:
        x = factor*output.reshape(-1, self.outputSize) @ self.weight.t()
        return x.reshape(*tuple([-1] + list(self.inputShape)))

    def gradOutput(self,
                   input: torch.Tensor,
                   factor: Union[torch.Tensor, float] = 1.,
                   **kwargs
                   ) -> torch.Tensor:
        x = factor * input.reshape(-1, self.inputSize) @ self.weight
        return x.reshape(*tuple([-1] + list(self.outputShape)))

    def backward(self,
                 input: torch.Tensor,
                 output: torch.Tensor,
                 factor: Union[torch.Tensor, float] = 1.,
                 **kwargs
                 ) -> torch.Tensor:
        gw = (factor*input.reshape(-1, self.inputSize)).t() @ output.reshape(-1, self.outputSize) / input.shape[0]
        #print("GW:", gw)
        if self.weight.grad is None:
            self.weight.grad = gw
        else:
            self.weight.grad += gw

    def zero_grad(self) -> None:
        self.weight.grad = None

    def plot_weight(self, mesh_size=None):
        if mesh_size == None:
          mesh_size = int(np.sqrt(self.inputSize))
        for i in range(10):
            plt.subplot(2, 5, i + 1)
            plt.tight_layout()
            plt.imshow(self.weight[:,i].reshape([mesh_size, mesh_size]).detach().cpu(), cmap='gray', interpolation='none')
            plt.xticks([])
            plt.yticks([])
        plt.show()

class InteractionModule(Interaction):
    '''
    A class taking a torch module as interaction between two layers. Note that only
    for linear interactions (Linear + Conv) layers this makes sense. For nonlinear
    models only gradients are used, leading to potentially wrong results.
    '''

    def __init__(self, module: torch.nn.Module, inputShape=None):
        '''
        :param module: The torch Module to be used. Usually only linear ones make sense
        :param inputShape: The shape of the input UnitLayer, needed for gradInput
        '''
        super().__init__()
        self.module = module
        if inputShape is not None:
            self.lastInputShape = inputShape
        else:
            self.lastInputShape = torch.Size([1,1])

    def enableModuleGrad(self, enable: bool=True) -> None:
        '''
        Enables/Disables the internal gradient calculation inside the module.
        :param enable: [bool] If internal module gradients are enabled.
        '''
        if enable:
            for p in self.module.parameters():
                p.requires_grad = True
        else:
            for p in self.module.parameters():
                p.requires_grad = False

    def negenergy(self,
                  input: torch.Tensor,
                  output: torch.Tensor,
                  factor: Union[torch.Tensor, float] = 1.,
                  **kwargs
                  ) -> torch.Tensor:
        self.lastInputShape = input.shape
        negenergy = (factor * output * self.module.forward(input)).sum(list(range(1, len(output.shape))))
        return negenergy.detach()

    def gradInput(self,
                  output: torch.Tensor,
                  input: Union[torch.Tensor, None] = None,
                  factor: Union[torch.Tensor, float] = 1.,
                  **kwargs
                  ) -> torch.Tensor:
        '''
        In the case of variable layer shapes the input is needed.
        '''
        if input is None:
            inp = torch.ones([output.shape[0]] + list(self.lastInputShape[1:]), device=output.device, requires_grad=True)
        else:
            inp = input
        output = torch.tensor(output.data, requires_grad=False, device=output.device)
        if isinstance(factor, torch.Tensor):
            factor = factor.detach()
        #    input.requires_grad = True
        with torch.enable_grad():
            self.enableModuleGrad(False)
            outprime = self.module.forward(inp)
        #self.enableModuleGrad(False)
            outprime.backward(output * factor)
        del output
        return inp.grad.detach()

    def gradOutput(self,
                   input: torch.Tensor,
                   factor: Union[torch.Tensor, float] = 1.,
                   **kwargs
                   ) -> torch.Tensor:
        # print(self.module.forward(input).shape)
        self.lastInputShape = input.shape
        self.enableModuleGrad(False)
        #print("MGO:", self.module.forward(input).shape, input.shape)
        return factor*self.module.forward(input).detach()

    def backward(self,
                 input: torch.Tensor,
                 output: torch.Tensor,
                 factor: Union[torch.Tensor, float] = 1.,
                 **kwargs
                 ) -> torch.Tensor:
        with torch.enable_grad():
            self.enableModuleGrad(True)
            outprime = self.module.forward(input)
            outprime.backward(output * factor)

class InteractionPoolMapIn1D(InteractionModule):
    '''
    A class for a Mapping of a tensor to a tensor of different shape
    [ ... , N ] -> [ ... , N / m , m + 1]
    where only the first m elements are filled in the last dim
    '''
    def __init__(self, poolsize: int):
        '''
        :param poolsize: Pooling size in 1D
        '''
        module = torch.nn.Sequential(
            Expansion1D(poolsize),
            Projection((0, ), (poolsize, ), (poolsize,), (0,), (poolsize, ), (poolsize+1,))
        )
        super().__init__(module)

    def backward(self,
                 input: torch.Tensor,
                 output: torch.Tensor,
                 factor: Union[torch.Tensor, float] = 1.,
                 **kwargs
                 ) -> torch.Tensor:
        '''
        No internal parameters here.
        '''
        pass

class InteractionPoolMapOut1D(InteractionModule):
    '''
    A class for a Mapping between tensors
    [ ..., N, m+1] -> [ ..., N]
    where only the last 1 element is taken
    '''
    def __init__(self, poolsize: int):
        '''
        :param poolsize: Pooling size in 1D
        '''
        module = torch.nn.Sequential(
            Projection((poolsize,), (poolsize+1,), (poolsize+1,), (0,), (1,), (1,)),
            Truncation1D(poolsize),
            InvertGrayscale()
        )

        super().__init__(module)

    def backward(self,
                 input: torch.Tensor,
                 output: torch.Tensor,
                 factor: Union[torch.Tensor, float] = 1.,
                 **kwargs
                 ) -> torch.Tensor:
        '''
        No internal parameters here.
        '''
        pass


class InteractionReversed(Interaction):
    '''
    A class for reverting an interaction (exchanging input and output).
    '''
    def __init__(self, interaction):
        '''
        :param interaction: The Interaction to be reversed.
        '''
        super().__init__()
        self.interaction = interaction

    def gradInput(self,
                  output: torch.Tensor,
                  input: Union[torch.Tensor, None] = None,
                  factor: Union[torch.Tensor, float] = 1.,
                  **kwargs
                  ) -> torch.Tensor:
        return self.interaction.gradOutput(output, factor=factor)

    def gradOutput(self,
                   input: torch.Tensor,
                   factor: Union[torch.Tensor, float] = 1.,
                   **kwargs
                   ) -> torch.Tensor:
        return self.interaction.gradInput(input, factor=factor)

    def backward(self,
                 input: torch.Tensor,
                 output: torch.Tensor,
                 factor: Union[torch.Tensor, float] = 1.,
                 **kwargs
                 ) -> torch.Tensor:
        return self.interaction.backward(output, input, factor=factor)

    def negenergy(self,
                  input: torch.Tensor,
                  output: torch.Tensor,
                  factor: Union[torch.Tensor, float] = 1.,
                  **kwargs
                  ) -> torch.Tensor:
        return self.interaction.negenergy(output, input, factor, **kwargs)


class InteractionPoolMapIn2D(InteractionModule):
    '''
    Interaction merging pooling input to a common tensor of
    pooling input and output.
    '''
    def __init__(self, poolsize1: int, poolsize2: int):
        module = torch.nn.Sequential(
            Expansion2D(poolsize1, poolsize2),
            Truncation1D(poolsize2),
            Projection((0,), (poolsize1*poolsize2,), (poolsize1*poolsize2,),
                       (0,), (poolsize1*poolsize2,), (poolsize1*poolsize2+1,)),
        )
        super().__init__(module)

    def backward(self,
                 input: torch.Tensor,
                 output: torch.Tensor,
                 factor: Union[torch.Tensor, float] = 1.,
                 **kwargs
                 ) -> torch.Tensor:
        '''
        No internal parameters here.
        '''
        pass


class InteractionPoolMapOut2D(InteractionModule):
    '''
    Interaction mapping a ProbMaxPool layer back to an image.
    '''
    def __init__(self, poolsize1: int, poolsize2: int):
        p1p2 = poolsize1 * poolsize2
        module = torch.nn.Sequential(
            Projection((p1p2,), (p1p2+1,), (p1p2+1,), (0,), (1,), (1,)),
            Truncation1D(1),
            InvertGrayscale()
        )
        super().__init__(module)

    def backward(self,
                 input: torch.Tensor,
                 output: torch.Tensor,
                 factor: Union[torch.Tensor, float] = 1.,
                 **kwargs
                 ) -> torch.Tensor:
        '''
        No internal parameters here.
        '''
        pass

class InteractionSequential(Interaction):
    '''
    Combines Interactions sequentially with no random UnitLayers
    (i.e. DiracDeltaLayers) in between.
    '''
    def __init__(self, *interactions):
        '''
        :param interaction: List of Interactions to be concatenated.
        '''
        super().__init__()
        self.interactions = ListModule(*interactions)

    def negenergy(self,
                  input: torch.Tensor,
                  output: torch.Tensor,
                  factor: Union[torch.Tensor, float] = 1.,
                  **kwargs
                  ) -> torch.Tensor:
        inputs = [input]
        outputs = [output]
        for inter in self.interactions:
            inputs.append(inter.gradOutput(inputs[-1], factor=factor, **kwargs))
        for inter in reversed(self.interactions):
            outputs = [inter.gradInput(outputs[0])] + outputs
        # forward backward style algorithm
        negen = 0.
        for n, inter in enumerate(self.interactions):
            negen = negen + inter.negenergy(inputs[n], outputs[n+1], factor=factor, **kwargs)
        return negen

    def gradInput(self,
                  output: torch.Tensor,
                  input: Union[torch.Tensor, None] = None,
                  factor: Union[torch.Tensor, float] = 1.,
                  **kwargs
                  ) -> torch.Tensor:
        if input:
            inputs = [input]
            for inter in self.interactions:
                inputs.append(inter.gradOutput(inputs[-1], factor=1., **kwargs))
        else:
            inputs = [None]*len(self.interactions)
        for n, inter in enumerate(reversed(self.interactions)):
            inp = inputs[len(self.interactions)-n-1]
            if inp is not None:
                inp2 = torch.tensor(inp.data, requires_grad=True, device=inp.device)
            else:
                inp2 = None
            output = inter.gradInput(output, input=inp2, factor=1., **kwargs)
            if inp2 is not None:
                del inp2
        return factor*output

    def gradOutput(self,
                   input: torch.Tensor,
                   factor: Union[torch.Tensor, float] = 1.,
                   **kwargs
                   ) -> torch.Tensor:
        for inter in self.interactions:
            input = inter.gradOutput(input, factor=1., **kwargs)
        return factor*input

    def backward(self,
               input: torch.Tensor,
               output: torch.Tensor,
               factor: Union[torch.Tensor, float] = 1.,
               **kwargs
               ) -> torch.Tensor:
        inputs = [input]
        outputs = [output]
        for inter in self.interactions:
            inputs.append(inter.gradOutput(inputs[-1], factor = 1., **kwargs))
        for n, inter in enumerate(reversed(self.interactions)):
            device = inputs[len(self.interactions)-n-1].device
            inp = torch.tensor(inputs[len(self.interactions)-n-1].data, requires_grad=True, device=device)
            outputs = [inter.gradInput(outputs[0], input=inp, factor=1.)] + outputs
            del inp
        # forward backward style algorithm
        for n, inter in enumerate(self.interactions):
            inter.backward(inputs[n], outputs[n+1], factor=factor)
