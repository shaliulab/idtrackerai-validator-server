# Purpose

Validate results of experiments in the flyhostel platform


# How to install in Linux


## Download

```
mkdir -p $HOME/opt
cd $HOME/opt
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
## Set up npm

*Install npm if not available in your machine. Google how.

Then run

```
cd $HOME/opt/idtrackerai-validator-server/idtrackerai-validator-client
npm start
```


# Setup
```
echo "export FLYHOSTEL_VIDEOS='/path/to/flyhostel_data/videos'" >> ~/.bashrc
```


# Run


## Run backend

Spawn a terminal and run
```
cd $HOME/opt/idtrackerai-validator-server
python idtrackerai_validator_server/main.py
```



## Run frontend

Spawn a terminal and run
```
cd $HOME/opt/idtrackerai-validator-server/idtrackerai-validator-client
npm start
```
