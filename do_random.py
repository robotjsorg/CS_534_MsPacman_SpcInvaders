import gym
env = gym.make('MsPacman-v0')
for _ in range(20):
    observation = env.reset()
    for t in range(100000):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break