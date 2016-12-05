import matplotlib.pyplot as plt
import numpy as np
import os

N = 100

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

fig = plt.figure()
ax = plt.subplot(111)
ax.hold(True)
for file in os.listdir('./analysis'):
    if file.endswith('.csv'):
        d = np.loadtxt(open('analysis/'+file, 'rb'), delimiter=',', skiprows=1)
        mv_av = movingaverage(d[:,1], N)
        file = file.replace(' ', '')[:-4].lower()
        ax.plot(d[:,0], mv_av, label=file)

plt.xlim(0, 1000)
plt.title('Moving Average N = '+str(N))
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('Episode')
plt.ylabel('Score')
ax.grid(True)
plt.show()