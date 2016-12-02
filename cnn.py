#!/bin/python
#SBATCH -N 1
#SBATCH -p exclusive
#SBATCH -o tf_test.out
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:2
FRAME_WIDTH = 84
FRAME_HEIGHT = 84
STATE_LENGTH = 4

import math,random, numpy, gym

#from keras.models import Sequential
#from keras.models import load_model
#from keras.layers import *
#from keras.optimizers import *

import json
from keras import initializations
from keras.initializations import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam

from skimage.transform import resize
from skimage.color import rgb2gray
import h5py
class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.model = self.createModel()

    def createModel(self):
        '''model = Sequential()

        model.add(Dense(output_dim=64, activation='relu', input_dim=stateCnt))
        model.add(Dense(output_dim=64, activation='relu', input_dim=stateCnt))
        model.add(Dense(output_dim=actionCnt, activation='linear'))

        opt = RMSprop(lr=0.00025)
        model.compile(loss='mse', optimizer=opt)
        model.add(Convolution2D(32, 8, 8, subsample=(4, 4), activation='relu', input_shape=(STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT)))
        model.add(Convolution2D(64, 4, 4, subsample=(2, 2), activation='relu'))
        model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.num_actions,activation = "linear"))'''

        model = Sequential()
        model.add(Convolution2D(32, 8, 8, subsample=(4,4),init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same',input_shape=(4,84,84)))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 4, 4, subsample=(2,2),init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3, subsample=(1,1),init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same'))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512, init=lambda shape, name: normal(shape, scale=0.01, name=name)))
        model.add(Activation('relu'))
        model.add(Dense(self.actionCnt,init=lambda shape, name: normal(shape, scale=0.01, name=name)))
       
        adam = Adam(lr=1e-6)
        model.compile(loss='mse',optimizer=adam)


        return model

    def loadModel(self,filename):
        self.testModel = load_model(filename)


    def train(self, x, y, epoch=1, verbose=0):
        self.model.fit(x, y, batch_size=64, nb_epoch=epoch, verbose=verbose)

    def predict(self, s):
        return self.model.predict(s)

    def predictTest(self,s):
        self.loadModel('pacman-basic.h5')
        return self.testModel.predict(s)

    def predictOne(self, s):
        return self.predict(s.reshape(1, self.stateCnt)).flatten()\


    def predictOneTest(self,s):
        return self.predictTest(s.reshape(1,self.stateCnt)).flatten()

class Memory:
    a = []
    def __init__(self,capacity):
        self.capacity  = capacity

    def add2memory(self, s):
        if len(self.a)>self.capacity :
            self.a.pop(0)

        self.a.append(s)

    def sample(self,n):
        n = min(n,len(self.a))
        return random.sample(self.a,n)

    def getMemoryLength(self):
        return len(self.a)

MEMORY_CAPACITY = 100000
BATCH_SIZE = 64

GAMMA = 0.99

MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.001      # speed of decay

class Agent:
    def __init__(self,stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.brain = Brain(self.stateCnt,self.actionCnt)
        self.memory = Memory(MEMORY_CAPACITY)

    def act(self,state):
        if random.random() > MIN_EPSILON:
            #action = numpy.argmax(self.brain.predictOne(state))
            action = numpy.argmax(self.brain.predict(state))
        else:
            action = random.randint(0,self.actionCnt - 1)

        return action

    def actTest(self,state):
        action = numpy.argmax(self.brain.predictOneTest(state))
        return action


    def observe(self,sample):
        self.memory.add2memory(sample)

    def memLen(self):
        return self.memory.getMemoryLength()

    def replay(self):
        batch = self.memory.sample(BATCH_SIZE)
        batchLen = len(batch)

        inputs = numpy.zeros((batchLen, 4, 84, 84))
        targets = numpy.zeros((inputs.shape[0], self.actionCnt))

        for i in range(batchLen):
            state_t = batch[i][0]
            action_t = batch[i][1]
            reward_t = batch[i][2]
            state_t1 = batch[i][3]
            terminal_state = batch[i][4]

            inputs[i] = state_t
            targets[i] = self.brain.predict(state_t)
            q_sa = self.brain.predict(state_t1)

            if terminal_state:
                targets[i,action_t] = reward_t
            else:
                targets[i,action_t] = reward_t + GAMMA*numpy.amax(q_sa)

        self.brain.train(inputs,targets)

MEMORY_TRAINING_BEGIN = 1000

class Environment:
    def __init__(self, problem):
        self.problem = problem
        self.env = gym.make(problem)

    def run(self, agent):
        x = self.env.reset()
        R = 0
        x_t = self.preprocessorConv(x)
        s_t = numpy.stack((x_t, x_t, x_t, x_t), axis=0)
        s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])
        while True:
            a = agent.act(s_t)

            x_t1, r, done, info = self.env.step(a)
            x_t1 = self.preprocessorConv(x_t1)
            x_t1 = x_t1.reshape(1,1,84,84)
            s_t1 = numpy.append(x_t1, s_t[:, :3, :, :], axis=1)
            agent.observe( (s_t, a, r, s_t1,done) )

            if agent.memLen() > MEMORY_TRAINING_BEGIN:
                agent.replay()

            s_t = s_t1
            R += r

            if done:
                break
	return R

    def preprocess(self,state):
        state = state[0:171,:]
        state = resize(rgb2gray(state), (84, 84))
        state = state.reshape(1,7056)
        return state

    def preprocessorConv(self,state):
        state = resize(rgb2gray(state), (84, 84))
        return state

functionname, _ = os.path.splitext(__file__)
n = 0
filename = "analysis/"+functionname+str(n)+".csv"
while os.path.isfile(filename):
    n = n + 1
    filename = "analysis/"+functionname+str(n)+".csv"
print filename
with io.FileIO(filename, "w") as file:
    file.write("Episode, Score\n")

TEST_FLAG = False
PROBLEM = 'MsPacman-v0'
env = Environment(PROBLEM)

stateCnt  = env.env.observation_space.shape[0]
stateCnt = 7056
actionCnt = env.env.action_space.n

agent = Agent(stateCnt, actionCnt)

i = 1
bestReward = 200

while True:
    R = env.run(agent)
    i = i + 1
    if R > bestReward:
        agent.brain.model.save("pacman-basic.h5")
        bestReward = R

    print "Episode %d finished with score of %d" % (i+1, R)
    with io.FileIO(filename, "a") as file:
        file.write("%d, %d\n" % (i+1, R))
