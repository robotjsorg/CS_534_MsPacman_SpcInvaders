# CS_534_MsPacman
Reinforcement Learning for Ms. Pacman

## Installation Instructions
```
pip install gym gym[atari]
```

On OSX or maxOS:
```
brew install cmake boost boost-python sdl2 swig wget
```

On Ubuntu 14.04:
```
apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
```

## Github
```
git clone https://github.com/jmcmahon443/CS_534_MsPacman.git
```

## Sample Agents
```
python do_nothing.py
```
```
python do_random.py
```

## Analysis
These functions analyze whatever `.csv` files are in the `/analysis/` folder.
```
python last_100_avg_std.py
```
```
python mv_avg_plot.py
```
