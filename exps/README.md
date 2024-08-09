# Radar experiments

Experiments related to the RADAR project (collaboration PM x Léo).

## Organisation

```bash
lib/                # libraries used in the experiments
                    #   (python modules, git submodules, python wheels, ...)
    python-fids/    # FIDS python library
    efc/            # EnergyFlowClassifier python wheel
    ...
data/               # data used in the experiments
    ...
baselines/          # comparison baselines
    bertoli2022/    # Bertoli et al. 2022, "Generalizing intrusion detection 
                    #   for heterogeneous networks: A stacked-unsupervised
                    #   federated learning approach"
    ...
                    # each experiment is in its own folder
trustfids/          # main trustfids package
    x-eval/         # cross-evaluation algorithm and test
        ...
    clustering/     # clustering algorithms and test. 
        ...
    __main__.py     # package input

...
proto.py            # function prototypes 
pyproject.toml      # Poetry configuration file
poetry.lock         # Poetry lock file (should not be modified manually)
flake.nix           # Nix Flake (for reproducible development environment)
flake.lock          # Nix lock file (should not be modified manually)
```

## Getting started

Clone the repository:

```bash
# clone the repository with submodules, as libs might be in submodules
git clone --recursive git@gitlab.imt-atlantique.fr/l20lavau/trust-fids.git
# you can also pull the submodules after cloning
git submodule update --init --recursive
```

Install the dependencies:

```bash
# this project uses poetry for dependency management
poetry install
```

Download the data (see [data/README.md](data/README.md)):

```bash
# some datasets are already in the repository, for the others, use ./data/origin/get_dataset.sh
cd ./data/origin
./get_dataset.sh -d nf_v2
```

Run the experiments:

```bash
# run a comman`
poetry run python path/to/script.py
# or open a shell
poetry shell
python path/to/script.py
```

## Python module

Basic usage:

```bash
# entrypoint
poetry run python -m trustfids 
# print config
poetry run python -m trustfids --cfg job
# print hydra's config
poetry run python -m trustfids --cfg hydra
# enable debug mode
poetry run python -m trustfids xp.debug=true
```

Run the cross-evaluation experiment:

```bash
poetry run python -m trustfids strategy=simfedxeval server=xevalserver client=xevalclient
```
