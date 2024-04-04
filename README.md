[![SDOFM-Banner](assets/SDO-FM_Banner.png)](https://black.readthedocs.io/en/stable/)
<h2 align="center">SDO-FM: A foundation model for the Sun</h2>


<p align="center">
<a href="https://github.com/psf/black/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/licence-Anne%3F-8A2BE2.svg"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://wandb.ai/fdlx/sdofm/"><img alt="Weights and Biases" src="https://img.shields.io/badge/Weights_&_Biases-FFCC33?style=flat&logo=WeightsAndBiases&logoColor=black"></a>
<a href="https://console.cloud.google.com/cloud-build/builds?project=sdo-fm-2024"><img alt="Cloud Build" src="https://img.shields.io/badge/cloud_build-blue.svg"></a>
</p>

SDO-FM is envisioned as a ‘multi-modal’ foundation model, integrating instruments on SDO; AIA, EVE, HMI. We will be looking to experiment with the scope multi-modal and choose datasets that are most complimentary to the task. The project will also investigate if the Foundation model can either replicate or leverage [SDOML](https://sdoml.org) (the data product developed in FDL.AI). 

## Installation and usage
### Installation
SDO-FM can be installed locally by directly installing the package in this repository.
```bash
pip install -e .
```

### Usage
To run any task we run within a container with the image described in the [Dockerfile](Dockerfile) and Hydra configurations, these are kept in the [experiments](experiments) directory. The entry point is [main.py](scripts/main.py) and args will select a configuration:
```bash
python scripts/main.py --config-name=default
```
CLI overrides are still possible with this selection but be aware of some shells not escaping quotes or sqaure brackets, e.g:
```bash
python scripts/main.py --config-name=default experiment.seed=37
```