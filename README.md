[![SDOFM-Banner](assets/SDO-FM_Banner.png)](https://black.readthedocs.io/en/stable/)
<h2 align="center">SDO-FM: A foundation model for the Sun</h2>


<p align="center">
<a href="https://github.com/psf/black/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/licence-Anne%3F-8A2BE2.svg"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://wandb.ai/fdlx/sdofm/"><img alt="Weights and Biases" src="https://img.shields.io/badge/Weights_&_Biases-FFCC33?style=flat&logo=WeightsAndBiases&logoColor=black"></a>
<a href="https://console.cloud.google.com/cloud-build/builds?project=sdo-fm-2024"><img alt="Cloud Build" src="https://img.shields.io/badge/cloud_build-blue.svg"></a>
<a href="https://github.com/spaceml-org/SDO-FM/actions/workflows/black.yml"><img alt="Lint" src="https://github.com/spaceml-org/SDO-FM/actions/workflows/black.yml/badge.svg?branch=main"></a>
</p>

SDO-FM is envisioned as a ‘multi-modal’ foundation model, integrating instruments on SDO; AIA, EVE, HMI. We will be looking to experiment with the scope multi-modal and choose datasets that are most complimentary to the task. The project will also investigate if the Foundation model can either replicate or leverage [SDOML](https://sdoml.org) (the data product developed in FDL.AI). 

#### Repo structure
```bash
├── assets              # images for this readme
├── experiments         # configuration files for different trials 
├── notebooks           # visualisation/testing ipynb
├── scripts             # entrypoint and highest level executors
├── sdofm               # python package
│   ├── datasets        # dataloaders/modules
│   ├── finetuning      # modules for finetuning
│   ├── models          # model components 
└── └── pretraining     # modules for pretraining
```

## Setup
### Installation
SDO-FM can be installed locally by directly installing the package in this repository.
```bash
pip install -e .
```

### Usage
To run any task we assume execution inside a container with the image described in the [Dockerfile](Dockerfile) and Hydra configurations, these are kept in the [experiments](experiments) directory. The entry point is [main.py](scripts/main.py) and args will select a configuration:
```bash
python scripts/main.py --config-name=default
```
CLI overrides are still possible with this selection but be aware of some shells not escaping quotes or sqaure brackets:
```bash
python scripts/main.py --config-name=default experiment.seed=37
```

### Datasets

| Name 	| Description 	| Granularity & Source 	|
|---	|---	|---	|
|  NASA’s Solar Dynamics Observatory (SDO) [Pesnell et al. 2012](https://ui.adsabs.harvard.edu/link_gateway/2012SoPh..275....3P/doi:10.1007/s11207-011-9841-3)  | Three instruments:<br><ul><li>Atmospheric Imaging Assembly (AIA) 2 ultraviolet, 1600 & 1700 Å 7 extreme ultraviolet, 94, 131, 171, 193, 211, 304, and 335 Å.</li><li>Helioseismic and Magnetic Imager (HMI) - visible filtergrams processed into: photospheric Dopplergrams line-of-sight magnetograms vector magnetograms.</li><li>EUV Variability Experiment (EVE) - EUV spectral irradiance from 1 to 1050 Å. MEGS disperse EUV light from full disk of the Sun and corona onto a 1024 x 2048 charge coupled device.</li></ul> | AIA: [Lemen et al. 2012](https://ui.adsabs.harvard.edu/link_gateway/2012SoPh..275...17L/doi:10.1007/s11207-011-9776-8).<br>HMI: [Hoeksema et al. 2014](https://ui.adsabs.harvard.edu/link_gateway/2014SoPh..289.3483H/doi:10.1007/s11207-014-0516-8).<br>EUV: [Woods et al. 2012](https://ui.adsabs.harvard.edu/link_gateway/2012SoPh..275..115W/doi:10.1007/s11207-009-9487-6).<br>Processed into 512x512/0.6, 512x512/0.5 arcsec for machine learning ready: [Galvez et al. 2019](https://iopscience.iop.org/article/10.3847/1538-4365/ab1005) via [sdoml.org](sdoml.org). |

## Pre-training
```bash
python scripts/main.py --config-name=pretrain_tiny
```


## Fine-tuning
### Science objective 1: Dimming
```bash
python scripts/main.py --config-name=dimming_tiny
```
### Science objective 2: TBD
### Science objective 3: TBD


## Additional Documentation
TODO

## Citation 
```bib
@software{SDOFM_2024,
    title           = {{Solar Dynamics Observatory Foundation Model}},
    repository-code = {https://github.com/spaceml-org/SDO-FM},
    year            = {2024}
}
```
