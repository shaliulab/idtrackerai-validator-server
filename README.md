# Purpose

Validate results of experiments in the flyhostel platform


# How to install in Linux


## Download

```
git clone git@github.com:shaliulab/idtrackerai-validator-server
cd idtrackerai-validator-server
```

# Make conda/mamba environment (optional)
````
conda create --name validator python=3.10.4
conda activate validator
````

## Install python dependencies (tested on Python 3.10.4)
```
git submodule update --init
pip install idtrackerai
pip install flyhostel
pip install .
```
# Install npm if not installed

# Setup
```
echo "export FLYHOSTEL_VIDEOS='/path/to/flyhostel_data/videos'" >> ~/.bashrc
```
