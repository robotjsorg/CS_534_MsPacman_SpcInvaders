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

plt.hold(True)
for file in os.listdir('./analysis'):
    if file.endswith('.csv'):
        d = np.loadtxt(open('analysis/'+file, 'rb'), delimiter=',', skiprows=1)
        mv_max = movingmax(d[:,1])
        file = file.replace(' ', '')[:-4].lower()

        # plt.plot(d[:,0], d[:,1], marker='o', linestyle='')
        plt.plot(d[:,0], mv_max, label=file)

plt.xlim(0, 100)
plt.title('Moving Maximum')
plt.legend(loc=4)
plt.xlabel('Episode')
plt.ylabel('Score')
plt.grid(True)
plt.show()