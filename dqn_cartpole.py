from keras.models import Sequential
from keras.layers import Dense
import gym
env = gym.make('CartPole-v0')
import numpy as np
import random
import collections


model = Sequential()
model.add(Dense(20,init='uniform',input_shape=(4,),activation='relu'))
model.add(Dense(20,init = 'uniform', activation='relu'))
model.add(Dense(2,init = 'uniform', activation = 'linear'))

model.compile(loss = 'mse',optimizer='rmsprop')

#Intialize memory
memory = collections.deque(maxlen=100000)
epsilon = 0.1
gamma = 0.9
epi = 1
r = 0
while True:
    observation = env.reset()
    for i in xrange(200):
        #print "Getting on: ",i
        state = observation.reshape(1,4)
        qSa = model.predict(state, batch_size=1)

        if random.random() > 0.1:             #Exploitation
            action = np.argmax(qSa)
        else:
            action = random.choice([0,1])
        #Make the transition
        new_state, reward, done, info = env.step(action)
        new_state = new_state.reshape(1,4)
        #Store transtion in memory
        memory.append([state, action, reward, new_state])
        #Now train
        if len(memory) > 1000:             #Change this
            batch_num = 1000
            #print "Hurr"
        else:
            batch_num = len(memory)
        mini_batch = random.sample(memory, batch_num )
        #print "Minibatch: ",mini_batch
        #target = [0]*batch_num
        target = []
        target_vector = np.zeros((batch_num,2))
        state_vector = np.zeros((1,4))
        for j in xrange(batch_num):
            #print "Batch_num",batch_num
            #target[j] = model.predict(mini_batch[j][0], batch_size=1)
            target_vector[j] = model.predict(mini_batch[j][0], batch_size=1)
            #print "Target Shape: ",target[j].shape
            #print "Action: ",mini_batch[j][1]
            #print "val J: ",j
            #print "Done", done
            target_vector[j, mini_batch[j][1]] = mini_batch[j][2] + gamma*np.max(model.predict(mini_batch[j][3], batch_size=1))
            #state_vector.append(mini_batch[j][0])
            state_vector = np.vstack([state_vector, mini_batch[j][0]])
            #target_vector[j][:] = target[j]
        #print "Target vector shape: ",target_vector.shape
        state_vector = np.array(state_vector)
        #target_vector = np.array(target)
        model.train_on_batch(state_vector[1:,:], target_vector)
        #model.fit(state_vector[1:, :], target_vector, batch_size=batch_num%32, nb_epoch=1)
        if done:
            r = r + i
            if epi%10==0:
                print epi, "Survived for ", i, "Average : ",r
                r = 0
            break
        state = new_state
    epi = epi + 1


