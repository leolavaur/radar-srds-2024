# RADAR

Mono-repo for all materials and experiments related to the RADAR project (collaboration PM x Léo).

## Organisation

- `papers/centralized`: Collaboration considering a federated but centrally-aggregated ML-IDS.
- `exps/results`: experiments results related to both collaboration, and future works.
- [_Project Management_](https://gitlab.imt-atlantique.fr/l20lavau/trust-fids/-/boards/257): issue
  board for planning.

In each paper directory, the `src` folder is a submodule pointing towards the related overleaf.

## Getting started

Setup the authentication for overleaf:

```bash
# example for leo
git config --global credential.https://git.overleaf.com.username leo.lavaur@imt-atlantique.fr
```

Clone the repository:

```bash
# with the overleaf link
git clone --recursive git@gitlab.imt-atlantique.fr:l20lavau/trust-fids.git
# you can also clone the repository without the papers by removing the `--recursive` option
```

Pull and commit the last changes of Overleaf:

```bash
git submodule update --recursive --remote --merge
git add .
git commit -m "Update Overleaf submodule"
git push
```
