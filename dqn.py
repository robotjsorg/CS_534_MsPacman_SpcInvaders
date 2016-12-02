import random, numpy, gym, os, io

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

from skimage.transform import resize
from skimage.color import rgb2gray

class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.model = self.createModel()

    def createModel(self):
        model = Sequential()

        model.add(Dense(output_dim=64, activation='relu', input_dim=stateCnt))
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
LAMBDA = 0.001 # speed of decay

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

    def observe(self,sample):
        self.memory.add2memory(sample)

    def memLen(self):
        return self.memory.getMemoryLength()

    def replay(self):
        batch = self.memory.sample(BATCH_SIZE)
        batchLen = len(batch)

        #print "Batch: ",batch[0][0].shape
        #print "BatchLen: ",batchLen
        no_state = numpy.zeros((1,self.stateCnt))

        s = numpy.array([o[0] for o in batch])
        s_ = numpy.array([(no_state if o[3] is None else o[3]) for o in batch])

        s = s[:,0,:]
        #print s_.shape
        s_ = s_[:,0,:]
        #print s.shape
        #print s_.shape

        p = self.brain.predict(s)
        p_ = self.brain.predict(s_)

        #print p.shape

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

MEMORY_TRAINING_BEGIN = 1000

class Environment:
    def __init__(self, problem):
        self.problem = problem
        self.env = gym.make(problem)

    def run(self, agent):
        s = self.env.reset()
        R = 0
        s = self.preprocess(s)
        while True:
            #self.env.render()
            #s = self.preprocess(s)
            a = agent.act(s)

            s_, r, done, info = self.env.step(a)
            s_ = self.preprocess(s_)
            if done: # terminal state
                s_ = None
            #s_ = self.preprocess(s_)
            agent.observe( (s, a, r, s_) )

            if agent.memLen() > MEMORY_TRAINING_BEGIN:
                #print "In replay"
                agent.replay()

            s = s_
            R += r

            if done:
                break
        return R

    def preprocess(self,state):
        state = state[0:171,:]
        state = resize(rgb2gray(state), (84, 84))
        state = state.reshape(1, 7056)
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

PROBLEM = 'MsPacman-v0'
env = Environment(PROBLEM)

stateCnt  = env.env.observation_space.shape[0]
stateCnt = 7056
actionCnt = env.env.action_space.n

agent = Agent(stateCnt, actionCnt)

for e in range(500):
    R = env.run(agent)
    
    print "Episode %d finished with score of %d" % (e+1, R)
    with io.FileIO(filename, "a") as file:
        file.write("%d, %d\n" % (e+1, R))