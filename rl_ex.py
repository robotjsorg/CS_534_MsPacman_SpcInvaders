import gym, os, io
import numpy as np

functionname, _ = os.path.splitext(__file__)
n = 0
filename = "analysis/"+functionname+str(n)+".txt"
while os.path.isfile(filename):
    n = n + 1
    filename = "analysis/"+functionname+str(n)+".txt"
print filename
with io.FileIO(filename, "w") as file:
    file.write("Episode, Score, Timesteps\n")

env = gym.make('MsPacman-v0')

def run_episode(env, best_parameters):
    observation = env.reset()
    reward_total = 0

    parameters = np.random.rand(9, 100800)

    timesteps = 0
    for t in xrange(2000):
        env.render()
        observation = observation.reshape([100800, 1])
        
        dice = np.random.rand()
        threshold = 0.9
        if dice > threshold:
            score = np.matmul(parameters, observation)
        else:
            score = np.matmul(best_parameters, observation)

        action = np.argmax(score)

        observation, reward, done, info = env.step(action)
        reward_total = reward_total + reward

        if done:
            timesteps = t + 1
            break
    return parameters, reward_total, timesteps

def train():
    best_param = np.random.rand(9, 100800)
    best_reward = 0

    for e in xrange(1000):
        parameters, reward, timesteps = run_episode(env, best_param)

        print "Episode %d finished with score of %d after %d timesteps" % (e+1, reward, timesteps)
        with io.FileIO(filename, "a") as file:
            file.write("%d, %d, %d\n" % (e+1, reward, timesteps))

        if reward > best_reward:
            best_reward = reward
            best_param = parameters

if __name__ == "__main__":
    p, r = train()
    print p
    print r
    run_episode(env, p)