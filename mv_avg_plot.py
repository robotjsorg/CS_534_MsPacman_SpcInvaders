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
        file = file.replace(' ', '')[:-4].lower()

        color = ''
        if file.startswith('do_nothing'):
            color = 'c'
        if file.startswith('do_random'):
            color = 'r'
        if file.startswith('rl'):
            color = 'g'
        if file.startswith('dqn'):
            color = 'b'
        if file.startswith('cnn'):
            color = 'm'

        # if file.startswith('dqn_99_spc_inv') or file.startswith('cnn_99_spc_inv'):
        if file.startswith('dqn_99_spc_inv'):
            f1000d = d[1000:, :]
            mv_av = movingaverage(f1000d[:, 1], N)
            ax.plot(d[1000:, 0], mv_av, label=file+'_first_1000', color=color)
            l1000d = d[:1000, :]
            mv_av = movingaverage(l1000d[:, 1], N)
            ax.plot(d[:1000, 0], mv_av, label=file+'_last_1000', color=color)
        else:
            mv_av = movingaverage(d[:,1], N)
            ax.plot(d[:,0], mv_av, label=file, color=color)


plt.xlim(0, 1000)
plt.title('Moving Average N = '+str(N))
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('Episode')
plt.ylabel('Score')
ax.grid(True)
plt.show()