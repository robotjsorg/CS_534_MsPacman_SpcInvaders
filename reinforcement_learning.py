import gym
import numpy as np

env = gym.make('MsPacman-v0')

def run_episode(env):
    observation = env.reset()
    reward_total = 0
    parameters = np.random.rand(9, 100800)

    for t in xrange(2000):
        env.render()
        observation = observation.reshape([100800, 1])
        score = np.matmul(parameters, observation)
        action = np.argmax(score)
        observation, reward, done, info = env.step(action)
        reward_total = reward_total + reward

        if done:
            break
    return parameters, reward_total

def train():
    num_episodes = 1000
    best_param = 0
    best_reward = 0
    average_reward = 0

    for n in xrange(1000):
        print "Episode: ", n
        parameters, reward = run_episode(env)
        average_reward = ((n)*average_reward + reward)/(n+1)
        print("reward")
        print(reward)
        print("average_reward")
        print(average_reward)

        if reward > best_reward:
            best_reward = reward
            best_param = parameters

        if reward == 2000:
            break

if __name__ == "__main__":
    p, r = train()
    print p
    print r
    run_episode(env, p)