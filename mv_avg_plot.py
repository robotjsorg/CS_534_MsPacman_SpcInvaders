import matplotlib.pyplot as plt
import numpy as np
import os

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

plt.hold(True)
for file in os.listdir('./analysis'):
    if file.endswith('.csv'):
        d = np.loadtxt(open('analysis/'+file, 'rb'), delimiter=',', skiprows=1)
        plt.plot(d[:,0], d[:,1], marker='o', linestyle='')
        mv_av = movingaverage(d[:,1], 10)
        plt.plot(d[:,0], mv_av)

# plt.xlim(0, 500)
plt.xlabel('Episode')
plt.ylabel('Score')
plt.grid(True)
plt.show()