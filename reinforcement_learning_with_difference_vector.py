import gym
import cv2
import numpy as np

env = gym.make('MsPacman-v0')

def run_episode(env):
    observation = env.reset()
    last_observation = 0
    reward_total = 0

    for t in xrange(2000):
        env.render()
        
        observation = observation.reshape([100800, 1])

        if t == 0:
            parameters = np.random.rand(9, 100800)

        else:
            diff_observation = observation - last_observation
            changed_pixels = np.transpose(np.nonzero(diff_observation))
            changed_pixels = changed_pixels[:, 0]
            parameters = np.zeros((9, 100800))

            for pixel in changed_pixels:
                parameters[:, pixel] = np.random.rand(9)

        last_observation = observation

        score = np.matmul(parameters, observation)              # calculate each action's score
        action = np.argmax(score)                               # choose the action with the highest score

        observation, reward, done, info = env.step(action)
        reward_total = reward_total + reward

        if done:
            break
    return parameters, reward_total

def train():
    num_episodes = 1000
    best_param = 0
    best_reward = 0

    for n in xrange(1000):
        print "Episode: ", n
        parameters, reward = run_episode(env)

        if reward > best_reward:
            best_reward = reward
            best_param = parameters

        if reward == 2000:
            break

    return best_param, best_reward

if __name__ == "__main__":
    p, r = train()
    print p
    print r
    run_episode(env, p)