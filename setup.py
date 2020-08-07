from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='PyTorch-ProbGraph',
      version='0.0.1',
      description='Hierarchical Probabilistic Graphical Models in PyTorch',
      long_description=long_description,
      author='Korbinian Poeppel, Hendrik Elvers',
      author_email='korbinian.poeppel@tum.de, hendrik.elvers@tum.de',
      url='https://github.com/kpoeppel/pytorch_probgraph/',
      packages=['pytorch_probgraph'],
      install_requires=['torch', 'numpy', 'matplotlib', 'tqdm',
                        'sphinx_rtd_theme', 'sphinx', 'setuptools'],
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        ],
      python_requires='>=3.6',
)
