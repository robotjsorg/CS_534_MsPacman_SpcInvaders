import numpy as np
import matplotlib.pyplot as plt

#For CNN
data_cnn = np.loadtxt('tf_test_cnn.txt',delimiter = ",")
avg_cnn = []
total_cnn = 0
rewards_cnn = []
for i in range(1,data_cnn.shape[0]):
	if i>100:
		x = rewards_cnn.pop(0)
	rewards_cnn.append(data_cnn[i,1])
	avg_cnn.append(sum(rewards_cnn)/100.0)
	
#For DQN
data_dqn = np.loadtxt('tf_test_dqn.out',delimiter = ",")

total_dqn = 0
avg_dqn = []
rewards_dqn = []
for i in range(1,data_dqn.shape[0]):
	if i>100:
		x = rewards_dqn.pop(0)
	rewards_dqn.append(data_dqn[i,1])
	avg_dqn.append(sum(rewards_dqn)/100.0)

#For Do Random
data_random = np.loadtxt('do_random0.txt',delimiter = ",")
total_random = 0
avg_random = []
rewards_random = []
for i in range(1,data_random.shape[0]):
	if i>100:
		x = rewards_random.pop(0)
	rewards_random.append(data_random[i,1])
	avg_random.append(sum(rewards_random)/100.0)


print np.sum(data_cnn[:,1])/data_cnn.shape[0]
print np.sum(data_dqn[:,1])/data_dqn.shape[0]
print np.sum(data_random[:,1])/data_random.shape[0]

plt.plot(avg_cnn,'-r')
plt.plot(avg_dqn,'-b')
plt.plot(avg_random,'-g')
plt.show()
