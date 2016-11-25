#!/bin/python
import random, numpy, math, gym

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *


class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.model = self.createModel()

    def createModel(self):
        model = Sequential()

        model.add(Dense(output_dim=64, activation='relu', input_dim=stateCnt))
        model.add(Dense(output_dim=actionCnt, activation='linear'))

        opt = RMSprop(lr=0.00025)
        model.compile(loss='mse', optimizer=opt)

        return model

    def train(self, x, y, epoch=1, verbose=0):
        self.model.fit(x, y, batch_size=64, nb_epoch=epoch, verbose=verbose)

    def predict(self, s):
        return self.model.predict(s)

    def predictOne(self, s):
        return self.predict(s.reshape(1, self.stateCnt)).flatten()


class Memory:
    a = []
    def __init__(self,capacity):
        self.capacity  = capacity

    def add2memory(self, s):
        if len(self.a)>self.capacity :
            self.a.pop(0)

        self.a.append(s)

    def getMemoryLength(self):
        return len(self.a)

    def sample(self,n):
        n = min(n,len(self.a))
        return random.sample(self.a,n)

MEMORY_CAPACITY = 100000
BATCH_SIZE = 64

GAMMA = 0.99

MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.001      # speed of decay
MEMORY_TRAINING_BEGIN = 1000     #Memory Length till which no training should start

class Agent:
    def __init__(self,stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.brain = Brain(self.stateCnt,self.actionCnt)
        self.memory = Memory(MEMORY_CAPACITY)

    def act(self,state):
        if random.random() > MIN_EPSILON:
            action = numpy.argmax(self.brain.predictOne(state))
        else:
            action = random.randint(0,self.actionCnt - 1)

        return action

    def memLen(self):
        return self.memory.getMemoryLength()


    def observe(self,sample):
        self.memory.add2memory(sample)

    def replay(self):
        batch = self.memory.sample(BATCH_SIZE)
        batchLen = len(batch)

        print "Batch: ",batch[0][0].shape

        no_state = numpy.zeros(self.stateCnt)

        s = numpy.array([o[0] for o in batch])
        s_ = numpy.array([(no_state if o[3] is None else o[3]) for o in batch])

        p = self.brain.predict(s)
        p_ = self.brain.predict(s_)

        x = numpy.zeros((batchLen,self.stateCnt))
        y= numpy.zeros((batchLen,self.actionCnt))

        for i in range(batchLen):
            l = batch[i]
            pr = l[0]
            ac = l[1]
            r = l[2]
            ne = l[3]

            t = p[i]
            if ne==None:
                t[ac] = r
            else:
                t[ac] = r + GAMMA*numpy.amax(p_[i])

            x[i] = pr
            y[i] = t

        self.brain.train(x,y)

class Environment:
    def __init__(self, problem):
        self.problem = problem
        self.env = gym.make(problem)

    def run(self, agent):
        s = self.env.reset()
        R = 0

        while True:
            self.env.render()

            a = agent.act(s)

            s_, r, done, info = self.env.step(a)

            if done: # terminal state
                s_ = None

            agent.observe( (s, a, r, s_) )

            #if agent.memLen() > MEMORY_TRAINING_BEGIN:
            agent.replay()

            s = s_
            R += r

            if done:
                break

        print("Total reward:", R)


PROBLEM = 'CartPole-v0'
env = Environment(PROBLEM)

stateCnt  = env.env.observation_space.shape[0]
actionCnt = env.env.action_space.n

agent = Agent(stateCnt, actionCnt)


while True:
    env.run(agent)

    #agent.brain.model.save("cartpole-basic.h5")

