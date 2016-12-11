import gym, os, io
import numpy as np
from skimage.transform import resize
from skimage.color import rgb2gray

functionname, _ = os.path.splitext(__file__)
n = 0
filename = "analysis/"+functionname+str(n)+".csv"
while os.path.isfile(filename):
    n = n + 1
    filename = "analysis/"+functionname+str(n)+".csv"
print filename
with io.FileIO(filename, "w") as file:
    file.write("Episode, Score\n")

env = gym.make('SpaceInvaders-v0')
actLen = env.action_space.n
vecLen = np.prod(np.shape(env.observation_space.low))
newImgDim = 84
newVecLen = newImgDim**2

def run_episode(env, best_parameters):
    observation = env.reset()
    observation = preprocess(observation)

    reward_total = 0

    parameters = np.random.rand(actLen, newVecLen)

    for t in xrange(2000):
        # env.render()
        dice = np.random.rand()
        threshold = 0.6
        if dice > threshold:
            score = np.matmul(parameters, observation)
        else:
            score = np.matmul(best_parameters, observation)

        action = np.argmax(score)

        observation, reward, done, info = env.step(action)
        observation = preprocess(observation)
        reward_total = reward_total + reward

        if done:
            break
    return parameters, reward_total

def train():
    best_param = np.random.rand(actLen, newVecLen)
    best_reward = 0

    for e in xrange(1000):
        parameters, reward= run_episode(env, best_param)

        print "Episode %d finished with score of %d" % (e+1, reward)
        with io.FileIO(filename, "a") as file:
            file.write("%d, %d\n" % (e+1, reward))

        if reward > best_reward:
            best_reward = reward
            best_param = parameters

def preprocess(state):
    state = resize(rgb2gray(state), (newImgDim, newImgDim))
    state = state.reshape([newVecLen, 1])
    return state

if __name__ == "__main__":
    p, r = train()
    print p
    print r
    run_episode(env, p)