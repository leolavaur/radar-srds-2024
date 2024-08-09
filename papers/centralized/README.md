# Distributed Cross-evaluation for Reputation-aware Model Weighting in Federated Learning

This repository contains the sourcews for the paper "Distributed Cross-evaluation for Reputation-aware Model Weighting in Federated Learning" by Léo Lavaur, Pierre-Marie Lechevalier, et al., currently under review.

## Abstract

**TODO**

## Usage

The LaTeX source code is written on Overleaf and synchronized with this repository. 

### Compilation

A Makefile is provided to compile the paper. It requires `latexmk` and `biber` to be installed. To compile the paper, simply run:

```bash
make
```

To clean the build directory, run:

```bash
make clean
```

### Syncing with Overleaf

To sync the repository with Overleaf, make sure to have the correct remote set up:

```bash
git remote add overleaf-trustfids -m master https://git.overleaf.com/65030d488505f5c233148941
```

Then, to pull the latest changes from Overleaf, run:

```bash
# From the repository root
git subtree pull --squash --prefix papers/centralized/src overleaf-trustfids master
```

It is not recommended to push changes from the repository to Overleaf, as it may cause conflicts. 