import torch
from typing import Tuple, List

class ListModule(torch.nn.Module):
    '''
    Implements a List Module effectively taking a list of Modules and storing
    them. The modules can be indexed as for a usual list.
    '''
    def __init__(self,
                 *args: List[torch.nn.Module]):
        '''
        :param *args: List of Modules
        '''
        super().__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx: int):
        if idx < 0:
            return self.__getitem__(self.__len__() + idx)
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


class Reverse(torch.nn.Module):
    '''
    Reverts any module, with forward replacing backward and
    backward replacing forward.
    '''
    def __init__(self, module):
        '''
        :param module: the module to be reverted
        '''
        super().__init__()
        self.module = module

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        '''
        :param input: input data
        :returns: output data
        '''
        return self.module.backward(input)

    def backward(self, output: torch.Tensor) -> torch.Tensor:
        '''
        :param output: output data
        :returns: input data / gradient
        '''
        return self.module.forward(input)


class Projection(torch.nn.Module):
    '''
    | A class for a projection of an input to a different shape \
    effectively mapping from
    | [..., inshape[1] .. inshape[-1]] -> [..., outshape[1] .. outshape[-1]]
    | only going over the subelements.
    | Example input (4,6) to (4,5) (shapes):
    | with instart (0, 1) inend (4, 5) outstart (0, 0), outend (4, 4) \
    maps essentially input[:, 1:5] to a new tensor output[:4, 0:4] with shape \
    (4, 5)
    | Non-indexed elements in the output are set to zero.

    '''
    def __init__(self,
                 instart: Tuple[int],
                 inend: Tuple[int],
                 inshape: Tuple[int],
                 outstart: Tuple[int],
                 outend: Tuple[int],
                 outshape: Tuple[int]):
        '''
        :param instart: List of start indices of different dimensions in input
        :param inend: End indices (exclusive) in input
        :param inshape: Real input shapes (dimension sizes)
        :param outstart: List of start indices of different dimensions in output
        :param outend: End indices (exclusive) in output
        :param outshape: Real output shapes (dimension sizes)
        '''
        super().__init__()
        self.inindex = tuple([slice(instart[i], inend[i], 1) for i in range(len(inshape))])
        self.outindex = tuple([slice(outstart[i], outend[i], 1) for i in range(len(outshape))])
        self.inshape = inshape
        self.outshape = outshape

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        '''
        :param input: Input tensor
        :returns: output tensor
        '''
        inindex = [slice(None, None, 1) for _ in range(len(input.shape) - len(self.inshape))]
        outindex = inindex
        inindex = tuple(inindex + list(self.inindex))
        outindex = tuple(outindex + list(self.outindex))
        outshape = [input.shape[i] for i in range(len(input.shape) - len(self.inshape))]
        outshape += self.outshape
        output = torch.zeros(outshape, device=input.device, requires_grad=False)
        output[outindex] += input[inindex]
        #print(self.inshape, self.outshape, input.shape, outshape)
        #print("Projection", output.shape)
        return output

    def backward(self, output: torch.Tensor) -> torch.Tensor:
        '''
        :param output: output tensor to backward through module
        :returns: input gradient
        '''
        outindex = [slice(None, None, 1) for _ in range(len(output.shape) - len(self.outshape))]
        inindex = outindex
        outindex = tuple(outindex + list(self.outindex))
        inindex = tuple(inindex + list(self.inindex))
        inshape = [output.shape[i] for i in range(len(output.shape) - len(self.inshape))]
        inshape += self.inshape
        input = torch.zeros(inshape, device=output.device, requires_grad=input.requires_grad)
        input[inindex] += output[outindex]
        #print("ProjectionBack", input.shape)
        return input

class Expansion1D(torch.nn.Module):
    '''
    Adds a dimension to the tensor with certain size, dividing the now second
    last dimension.
    '''
    def __init__(self, expsize: int):
        '''
        :param expsize: Size of new last dimension
        '''
        super().__init__()
        self.expsize = expsize

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        '''
        :param input: input tensor
        :returns: output tensor
        '''
        shape = list(input.shape)
        newshape = shape[:-1] + [shape[-1]/self.expsize, self.expsize]
        return input.reshape(newshape)

class Truncation1D(torch.nn.Module):
    '''
    Merges the two last dimensions to one. Last dimension shape is needed for
    backward operation.
    '''
    def __init__(self, shape: int):
        '''
        :param shape: size of the last dimension
        '''
        super().__init__()
        self.shape = shape

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        '''
        :param input: input tensor
        :returns: output tensor
        '''
        newshape = list(input.shape[:-2]) + [input.shape[-2] * input.shape[-1]]
        #print("TrFor", input.shape, newshape)
        return input.reshape(newshape)

    def backward(self, output: torch.Tensor) -> torch.Tensor:
        '''
        :param output: output to put backwards
        :returns: input gradient
        '''
        #print("TrBack", input.shape, list(input.shape[:-2]) + [self.shape1, self.shape2])
        return output.reshape(list(output.shape[:-2]) + [output.shape[-1]/self.shape, self.shape])

class Expansion2D(torch.nn.Module):
    '''
    Expands a tensor in the last two dimensions, effectively to a coarse grid
    of smaller grids.
    '''
    def __init__(self, expsize1: int, expsize2: int):
        '''
        :param expsize1: size of the second last dimension to be created
        :param expsize2: size of the last dimension to be created
        '''
        super().__init__()
        self.expsize1 = expsize1
        self.expsize2 = expsize2

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        '''
        :param input: input tensor
        :returns: output tensor
        '''
        shape = list(input.shape)
        # print(shape)
        newshape = shape[:-2] + \
                   [shape[-2]//self.expsize1,
                    shape[-1]//self.expsize2,
                    self.expsize1,
                    self.expsize2]
        sliceshape = list(newshape)
        sliceshape[-4] = 1
        sliceshape[-3] = 1
        output = torch.zeros(newshape, device=input.device)
        baseslice = [slice(None, None, 1) for _ in range(len(shape)-2)]
        for i in range(shape[-2]//self.expsize1):
            for j in range(shape[-1]//self.expsize2):
                inslice = tuple(baseslice + \
                    [slice(self.expsize1*i, self.expsize1*(i+1)),
                     slice(self.expsize2*j, self.expsize2*(j+1))])
                outslice = tuple(baseslice + \
                    [i,
                     j,
                     slice(None, None, 1),
                     slice(None, None, 1)])
                #print(inslice, outslice, input.shape, output.shape)
                #print(input[inslice].shape)
                #print(outslice)
                #print(output[outslice].shape)
                output[outslice] += input[inslice] #.view(sliceshape)
        return output

class Truncation2D(torch.nn.Module):
    '''
    A module merging the last two dimensions, merging coarse scale in grid
    of dimensions -4, -3 and finer resolution in dimensions -2, -1 to
    one fine grained grid with two dimensions less.
    '''
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        '''
        :param input: input tensor
        :returns: output tensor
        '''
        shape = input.shape
        outputshape = list(input.shape[:-2])
        expsize1 = input.shape[-2]
        expsize2 = input.shape[-1]
        outputshape[-2] *= input.shape[-2]
        outputshape[-1] *= input.shape[-1]
        baseslice = [slice(None, None, 1) for _ in range(len(outputshape)-2)]
        output = torch.zeros(outputshape, device=input.device, requires_grad=False)
        for i in range(shape[-4]):
            for j in range(shape[-3]):
                outslice = tuple(baseslice + \
                    [slice(expsize1*i, expsize1*(i+1)),
                     slice(expsize2*j, expsize2*(j+1))])
                inslice = tuple(baseslice + \
                    [i,
                     j,
                     slice(None, None, 1),
                     slice(None, None, 1)])
                output[outslice] += input[inslice]
        #print("Trunc2D", input.shape, output.shape)
        return output

class InvertGrayscale(torch.nn.Module):
    '''
    Invert the input around 1, sensible for grayscale images in [0,1]
    distributions.
    '''
    def __init__(self):
        super().__init__()
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        '''
        :param input: input tensor
        :returns: output tensor
        '''
        return 1. - input
