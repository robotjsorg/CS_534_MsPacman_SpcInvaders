import gym, os, io
import numpy as np

functionname, _ = os.path.splitext(__file__)
filename = "analysis/"+functionname+str(n)+".csv"
while os.path.isfile(filename):
    filename = "analysis/"+functionname+str(n)+".csv"
print filename
with io.FileIO(filename, "w") as file:
    file.write("Episode, Score\n")

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
    best_param = 0
    best_reward = 0

    for e in xrange(1000):
        parameters, reward = run_episode(env)

        print "Episode %d finished with score of %d" % (e+1, reward)
        with io.FileIO(filename, "a") as file:
            file.write("%d, %d\n" % (e+1, reward))

        if reward > best_reward:
            best_reward = reward
            best_param = parameters

if __name__ == "__main__":
    p, r = train()
    print p
    print r
    run_episode(env, p)