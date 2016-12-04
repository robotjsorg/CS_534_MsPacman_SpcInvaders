import matplotlib.pyplot as plt
import numpy as np
import os

N = 100

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

plt.hold(True)
for file in os.listdir('./analysis'):
    if file.endswith('.csv'):
        d = np.loadtxt(open('analysis/'+file, 'rb'), delimiter=',', skiprows=1)
        mv_av = movingaverage(d[:,1], N)
        file = file.replace(' ', '')[:-4].lower()

        # plt.plot(d[:,0], d[:,1], marker='o', linestyle='')
        plt.plot(d[:,0], mv_av, label=file)

plt.xlim(0, 1000)
plt.title('Moving Average N = '+str(N))
plt.legend(loc=3)
plt.xlabel('Episode')
plt.ylabel('Score')
plt.grid(True)
plt.show()