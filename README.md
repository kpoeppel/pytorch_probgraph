# README of "PyTorch-ProbGraph"

## What is PyTorch-ProbGraph?

PyTorch-ProbGraph is a library based on amazing PyTorch (https://pytorch.org)
to easily use and adapt directed and undirected Hierarchical Probabilistic
Graphical Models. These include Restricted Boltzmann Machines,
Deep Belief Networks, Deep Boltzmann Machines and Helmholtz
Machines (Sigmoid Belief Networks).

Models can be set up in a modular fashion, using UnitLayers, layers of Random Units and Interactions between these UnitLayers.
Currently, only Gaussian, Categorical and Bernoulli units are available, but an extension can be made to allow all kinds of distributions from the Exponential family.
(see https://en.wikipedia.org/wiki/Exponential_family)

The Interactions are usually only linear for undirected models, but can be built
from arbitrary PyTorch torch.nn.Modules (using forward and the backward gradient).

There is a pre-implemented fully-connected InteractionLinear, one for using
existing torch.nn.Modules and some custom Interactions / Mappings to enable
Probabilistic Max-Pooling. Interactions can also be connected without intermediate
Random UnitLayers with InteractionSequential.

This library was built by Korbinian Poeppel and Hendrik Elvers during a Practical Course "Beyond Deep Learning - Uncertainty Aware Models" at TU Munich.
Disclaimer: It is built as an extension to PyTorch and not directly affiliated.

## Documentation
A more detailed documentation is included, using the Sphinx framework.
Go inside directory 'docs' and run 'make html' (having Sphinx installed).
The documentation can then be found inside the _build sub-directory.

## Examples
There are some example models, as well as an evaluation script in the `examples`
folder.

## License
This library is distributed in a ... license.

## References
Ian Goodfellow and Yoshua Bengio and Aaron Courville,
http://www.deeplearningbook.org

Jörg Bornschein, Yoshua Bengio Reweighted Wake-Sleep
https://arxiv.org/abs/1406.2751

Geoffrey Hinton, A Practical Guide to Training Restricted Boltzmann Machines
https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf

Ruslan Salakhutdinov, Learning Deep Generative Models
https://tspace.library.utoronto.ca/handle/1807/19226

Honglak Lee et al., Convolutional Deep Belief Networks for Scalable Unsupervised Learning of Hierarchical
Representations, ICML09

G.Hinton, S. Osindero A fast learning algorithm for deep belief nets
