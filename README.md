# CS_534_MsPacman
Reinforcement Learning for Ms. Pacman and Space Invaders

## Deep Reinforcement Learning
### [DQN Algortihm][1]
This implementation of reinforcement learning aims to capture the idea of end-to-end learning. The agent takes the complete game frame (RGB values of each pixel) and maps it to actions. At every state (frame) **s**, an action **a** is chosen according to epsilon greedy approach and the agent gets a reward **r** and proceeds to the next state **s'**. 
**Q - value** update is performed for this transition as:
                            ```
                            Q(s,a) = R + gamma*max(Q(s',a))
                            ```
**gamma** is the discount factor

### Experience Replay
Every transition **s - a - r - s'** is stored in a memory buffer of constant size. At a certain predefined frequency, this memory is sampled randomly and fed to the neural network to calculate the loss given by:
                            ```
                            Loss = sum(Q(s,a)_old - Q(s,a)_new).^2
                            ```
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
## DQN Agent
This agent takes the complete frame, downsamples it and converts it into grayscale. The 2-D frame matrix is converted to a single row vector which is fed into the neural network. Thus the network works on matrix_breadth x matrix_height size feature set.
```
python dqn.py
```
Or if working on a university cluster
```
sbatch dqn.py
```

## CNN Agent
This agent takes the complete RGB frame and stakes **4** consecutive frames as input to the neural network. The CNN agent effectively, learns the best features in order to closely approximate the Q - value function.
```
python cnn.py
```
Or if working on university cluster
```
sbatch cnn.py
```

## Analysis
These functions analyze whatever `.csv` files are in the `/analysis/` folder.
```
python last_100_avg_std.py
```
```
python mv_avg_plot.py
```
## References:
[1]:https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

[1][http://neuro.cs.ut.ee/demystifying-deep-reinforcement-learning/]

[2][https://jaromiru.com/2016/09/27/lets-make-a-dqn-theory/]
