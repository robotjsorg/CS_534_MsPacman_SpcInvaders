import gym
env = gym.make('MsPacman-v0')
import numpy as np

def run_episode(env, parameters):
    time_steps = 200
    observation = env.reset()
    reward_total = 0
    for t in xrange(2000):
        env.render()
        observation = observation.reshape([100800,1])
        parameters = np.random.rand(9,100800)
        score= np.matmul(parameters,observation)
        action = np.argmax(score)
        observation, reward, done, info = env.step(action)
        reward_total = reward_total + reward
        if done:
            break
    return reward_total


def train():
    num_episodes = 1000
    best_param = 0
    best_reward = 0
    for n in xrange(1000):
        print "Episode: ",n
        parameters = np.random.rand(9,100800)
        reward = run_episode(env,parameters)
        if reward>best_reward:
            best_reward = reward
            best_param = parameters
        if reward==2000:
            print "This is one is COOOLLLL!!!!!"
            break
    return best_param,best_reward

#def test(parameters):
    #for i in xrange(30):
        #run episode



if __name__=="__main__":
    p,r = train()
    print "Training Done!!!!"
    print "Best Stuff"
    print p
    print r
    print "Best Shit"
    run_episode(env,p)