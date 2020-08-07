.. PyHolo documentation master file, created by
   sphinx-quickstart on Mon Jul 2 13:50:42 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyTorch-ProbGraph's documentation!
=============================================

What is PyTorch-ProbGraph?
--------------------------

PyTorch-ProbGraph is a library based on amazing `PyTorch <https://pytorch.org>`_
to easily use and adapt directed and undirected Hierarchical Probabilistic
Graphical Models. These include Restricted Boltzmann Machines, Deep Belief
Networks, Deep Boltzmann Machines and Helmholtz Machines (Sigmoid Belief Networks).

Models can be set up in a modular fashion, using UnitLayers, layers of
Random Units and Interactions between these UnitLayers.
Currently, only Gaussian, Categorical and Bernoulli units are available,
but an extension can be made to allow all kinds of distributions
from the Exponential family.
(see `<https://en.wikipedia.org/wiki/Exponential_family>`_)

The Interactions are usually only linear for undirected models, but can be built
from arbitrary PyTorch torch.nn.Modules (using forward and the backward gradient).
There is a pre-implemented fully-connected InteractionLinear, one for using
existing torch.nn.Modules and some custom Interactions / Mappings to enable
Probabilistic Max-Pooling. Interactions can also be connected without intermediate
Random UnitLayers with InteractionSequential.

Using these UnitLayers and Interactions, Restricted Boltzmann Machines,
Deep Belief Networks, Deep Boltzmann Machines and Helmholtz Machines can be
defined. Undirected models can be trained using Contrastive Divergence / Persistent Contrastive Divergence
learning or Greedy Layerwise Learning / Pretraining (for deep models).
The directed Helmholtz Machine can be trained using either traditional Wake-Sleep
Learning or Reweighted Wake-Sleep.

This library was built by Korbinian Poeppel and Hendrik Elvers during a
Practical Course "Beyond Deep Learning - Uncertainty Aware Models" at TU Munich.
Disclaimer: It is built as an extension to PyTorch and not directly affiliated.

References
----------

Ian Goodfellow and Yoshua Bengio and Aaron Courville,
`<http://www.deeplearningbook.org>`_

Jörg Bornschein, Yoshua Bengio Reweighted Wake-Sleep
`<https://arxiv.org/abs/1406.2751>`_

Geoffrey Hinton, A Practical Guide to Training Restricted Boltzmann Machines
`<https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf>`_

Ruslan Salakhutdinov, Learning Deep Generative Models
`<https://tspace.library.utoronto.ca/handle/1807/19226>`_

Honglak Lee et al., Convolutional Deep Belief Networks for Scalable Unsupervised Learning of Hierarchical
Representations, ICML09

G.Hinton, S. Osindero A fast learning algorithm for deep belief nets


.. toctree::
   :maxdepth: 3
   :caption: Contents:

   api

   howto


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
