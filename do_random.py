import gym, os, io

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
for e in range(20):
    reward_total = 0
    observation = env.reset()
    for t in range(100000):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        reward_total = reward_total + reward
        if done:
            print "Episode %d finished with score of %d after %d timesteps" % (e+1, reward_total, t+1)
            with io.FileIO(filename, "a") as file:
                file.write("%d, %d, %d\n" % (e+1, reward_total, t+1))
            break