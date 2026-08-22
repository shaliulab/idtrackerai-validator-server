# Purpose

Validate results of experiments in the flyhostel platform


# How to install in Linux


## Download

```
mkdir -p $HOME/opt
cd $HOME/opt
git clone git@github.com:shaliulab/idtrackerai-validator-server
```

# Make conda/mamba environment (optional)
````
conda create --name validator python=3.10.4
conda activate validator
````

## Install python dependencies (tested on Python 3.10.4)
```
cd $HOME/opt/idtrackerai-validator-server
git submodule update --init
pip install ./idtrackerai ./flyhostel .
```
## Set up npm

* Install npm if not available in your machine. Google how.


Then run

```
cd $HOME/opt/idtrackerai-validator-server/idtrackerai-validator-client
npm install
```


# Setup
```
echo "export FLYHOSTEL_VIDEOS='/path/to/flyhostel_data/videos'" >> ~/.bashrc
FLYHOSTEL_VIDEOS='/path/to/flyhostel_data/videos'
```


# Structure database

The application assumes the experiment folders are saved in a database as follows

`$FLYHOSTEL_VIDEOS/FlyHostelN/GROUPSIZEX/FOLDER`

example:

`'/path/to/flyhostel_data/videos/FlyHostel1/3X/2026-08-19_14-00-00'`

Inside `2026-08-19_14-00-00` the application expects a collection of .mp4 files named 

`000001.mp4`
`000002.mp4`
`000003.mp4`

and so on

and a sqlite file called 

`FlyHostel1_3X_2026-08-19_14-00-00.db`

Once you have a database structured like that, run:

```
cd $FLYHOSTEL_VIDEOS
find -maxdepth 4 -mindepth 4 -regex .*FlyHostel.*db -not -name index.db > index.txt
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

The application should then be available under http://localhost:3000