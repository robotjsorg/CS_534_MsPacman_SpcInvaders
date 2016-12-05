import matplotlib.pyplot as plt
import numpy as np
import os

def movingmax(interval):
    mv_max_mem = 0
    mv_max = []
    for i in range(0, len(interval)):
    	d = interval[i]
    	if d > mv_max_mem:
    		mv_max_mem = d
    		mv_max.append(d)
    	else:
    		mv_max.append(mv_max_mem)
    mv_max = np.array(mv_max)
    return mv_max

fig = plt.figure()
ax = plt.subplot(111)
ax.hold(True)
for file in os.listdir('./analysis'):
    if file.endswith('.csv'):
        d = np.loadtxt(open('analysis/'+file, 'rb'), delimiter=',', skiprows=1)
        file = file.replace(' ', '')[:-4].lower()
        mv_max = movingmax(d[:,1])
        ax.plot(d[:,0], mv_max, label=file)

plt.xlim(0, 1000)
plt.title('Moving Maximum')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('Episode')
plt.ylabel('Score')
ax.grid(True)
plt.show()