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

### Data
Since its launch in 2010, NASA’s Solar Dynamics Observatory (SDO) [Pesnell et al. 2012](https://ui.adsabs.harvard.edu/link_gateway/2012SoPh..275....3P/doi:10.1007/s11207-011-9841-3) has continuously monitored Sun’s activity, delivering a wealth of valuable scientific data for heliophysics researchers with the use of three instruments:

- The Atmospheric Imaging Assembly (AIA) [Lemen et al. 2012](https://ui.adsabs.harvard.edu/link_gateway/2012SoPh..275...17L/doi:10.1007/s11207-011-9776-8), which captures 4096 x 4096 resolution images (with 0.6 arcsecond pixel size) of the full Sun in two ultraviolet (centered at 1600, and 1700 Å), seven extreme ultraviolet (EUV) centered at 94, 131, 171, 193, 211, 304, and 335 Å, and one visible (centered at 4500 Å) wavelength band.
- The Helioseismic and Magnetic Imager (HMI) [Schou et al. 2012](https://ui.adsabs.harvard.edu/link_gateway/2012SoPh..275..229S/doi:10.1007/s11207-011-9842-2) captures visible wavelength filtergrams of the full Sun at 4096 x 4096 resolution (a pixel size of 0.5 arcsecond), which are then processed into a number of data products, including photospheric Dopplergrams, line-of-sight magnetograms, and vector magnetograms [Hoeksema et al. 2014](https://ui.adsabs.harvard.edu/link_gateway/2014SoPh..289.3483H/doi:10.1007/s11207-014-0516-8).
 - The EUV Variability Experiment (EVE) [Woods et al. 2012](https://ui.adsabs.harvard.edu/link_gateway/2012SoPh..275..115W/doi:10.1007/s11207-009-9487-6) monitors the solar EUV spectral irradiance from 1 to 1050 Å. This is done by utilizing multiple EUV Grating Spectrographs (MEGS) that disperse EUV light from the full disk of the Sun and its corona onto a 1024 x 2048 charge coupled device (CCD).

#### The SDO ML Dataset
SDO for Machine Learning is available from [sdoml.org](https://sdoml.org). The SDO ML Dataset (covering 2010 - 2023) was originally published as [Galvez et al. 2019](https://iopscience.iop.org/article/10.3847/1538-4365/ab1005), and is hosted on the Stanford Digital Repository in Numpy’s compressed array format (.npz).

In version 2.0, we present an update to the work outlined in [Galvez et al. 2019](https://iopscience.iop.org/article/10.3847/1538-4365/ab1005), in which the full dataset has been converted to cloud friendly Zarr (.zarr) format. In addition, SDO/AIA data has been updated to account for a change in calibration after 2019. In addtion to the change in calibration, this updated format includes:

FITS header/keyword information (such as observation time, and exposure time).
Processes for continually updating the data until the present day.

## Pre-training
```bash
python scripts/main.py --config-name=pretrain
```


## Fine-tuning
### Science objective 1
### Science objective 2
### Science objective 3
```bash
python scripts/main.py --config-name=so1
```

## Inference
```bash
python scripts/main.py --config-name=deploy
```

## Additional Documentation


## Citation 
```bib
@software{SDOFM_2024,
    title           = {{Solar Dynamics Observatory Foundation Model}},
    repository-code = {https://github.com/spaceml-org/SDO-FM},
    year            = {2024}
}
```