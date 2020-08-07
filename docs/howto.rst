==================================
A small HowTo on PyTorch-ProbGraph
==================================

------------
Introduction
------------

PyTorch-ProbGraph is a library bringing the Probabilistic Graphical Framework
to PyTorch (`<https://pytorch.org>`_), orthogonal to (`<https://pyro.ai>`_),
with a narrow focus on the traditional Restricted Boltzmann Machine,
Deep Boltzmann Machine, Deep Belief Network and Helmholtz Machine as well
as their convolutional variants.

The core modules (all torch.nn.Modules) are the UnitLayer, representing some
random distributed variables and the Interaction, representing (directed or
undirected) interactions/links between these UnitLayers.

A hierarchical graphical model is now built combining these in one of the
following models or their variants:
- Restricted Boltzmann Machine (Contrastive Divergence / Persistent Contrastive Divergence)
- Deep Boltzmann Machine
- Deep Belief Network
- Helmholtz Machine (Wake-Sleep / Reweighted Wake Sleep)

--------------
A simple Model
--------------
.. code-block:: python

    from pytorch_probgraph.unitlayer import BernoulliLayer, CategoricalLayer
    from pytorch_probgraph.interaction import InteractionLinear
    from pytorch_probgraph.rbm import RestrictedBoltzmannMachineCD
    from pytorch_probgraph.dbn import DeepBeliefNetwork

    ## Load data as some iterator over training batches
    data = get_data_from_somewhere()

    # Define the layers (always take 1 for the first=batch dimension)
    blayer0 = BernoulliLayer(torch.zeros([1, 20], requires_grad=False))
    blayer1 = CategoricalLayer(torch.zeros([1, 5], requires_grad=True))
    blayer2 = BernoulliLayer(torch.zeros([1, 30], requires_grad=True))

    ## Define interactions between layers
    interaction0 = InteractionLinear(blayer0.bias.shape[1:], blayer1.bias.shape[1:])
    interaction1 = InteractionLinear(blayer0.bias.shape[1:], blayer1.bias.shape[1:])

    ## Define Restricted Boltzmann Machines to be stacked
    rbm0 = RestrictedBoltzmannMachineCD(blayer0, blayer1, interaction0)
    rbm1 = RestrictedBoltzmannMachineCD(blayer1, blayer2, interaction1)

    ## Define the optimizer and the Deep Belief Network
    opt = torch.opt.Adam(chain(rbm0.parameters(), rbm1.parameters()))
    dbn = DeepBeliefNetwork([rbm0, rbm1], opt)

    ## Train on data
    dbn.train(data, epochs=10)

    ## Generate a batch of 10 samples
    dbn.sample(N=10)

    ## Estimate some free energy of the setting
    dbn.free_energy_estimate(data)
