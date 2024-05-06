# sits_change_harmonic

Harmonic Model for ChangeDetection based on FORCE Datacube

## Description

This repository contains the code necessary to run Changedetection for Satellite Image Time Series with [Harmonic Model](https://www.sciencedirect.com/science/article/abs/pii/S0034425715000590) based on the [FORCE Datacube](https://force-eo.readthedocs.io/en/latest/index.html). 
Harmonic Model based on reference period will be used to predict expected spectral values. Those values will be compared with real spectral values regarding an uncertainty (standard deviation) and disturbance state can change with 3 consecutive times below or above threshold. 

Results can be plots for points:

<img src="img/change.png" width="300" height="200" /> <img src="img/nochange.png" width="300" height="200" />

And results can [README.md](..%2FSITS_classification%2FREADME.md)be grid based where every pixel has:
- the first date where the disturbance occurred
- 90th percentile for disturbance residuals over the entire time period
- 90th percentile for disturbance residuals within specified time ranges


## Getting Started

### Dependencies

* GDAL, ...
* sudo apt-get install xterm
* Cuda-capable GPU ([overview here](https://developer.nvidia.com/cuda-gpus))
* force docker Version 3.7.11


### Installation

* conda create --name SITSchange python==3.9
* conda activate SITSchange
* cd /path/to/repository
* pip install -r requirements.txt
* sudo apt-get install xterm

### Executing program

* set config_path for directory structure
* set parameters in harmonic_main and execute file

## Help/Known Issues

* None yet

# Info

## Authors

* Benjamin Stöckigt

## Version History

* 0.1
    * Initial Release

## License

GPL-3.0 license

## Acknowledgments

Inspiration, code snippets, etc.

* [FORDEAD](https://fordead.gitlab.io/fordead_package/)
* [FORCE Tutorials](https://force-eo.readthedocs.io/en/latest/howto/udf_py.html)
