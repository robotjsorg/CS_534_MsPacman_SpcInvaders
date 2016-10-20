import gym
env = gym.make('MsPacman-v0')
print(env.action_space)
#> Discrete(2)
print(env.observation_space)

from gym import spaces
space = spaces.Discrete(8) # Set with 8 elements {0, 1, 2, ..., 7}
x = space.sample()
print(x)
assert space.contains(x)
assert space.n == 8
