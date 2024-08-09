# Trust-FIDS datasets

This folder contains the datasets used in the Trust-FIDS project. All datasets are stored in GZIP
compressed CSV files. The `reduced` and `sampled` are taken from Bertoli2022.

## Creating adversary datasets
To create the adversary datasets : 
```
cd exps
poetry run python ./trustfids/utils/create_adversary.py
```