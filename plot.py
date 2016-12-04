import matplotlib.pyplot as plt
import numpy as np
import os

plt.hold(True)
for file in os.listdir('./analysis'):
    if file.endswith('.csv'):
        d = np.loadtxt(open('analysis/'+file, 'rb'), delimiter=',', skiprows=1)
        file = file.replace(' ', '')[:-4].lower()

        plt.plot(d[:,0], d[:,1], label=file)

plt.xlim(0, 1000)
plt.title('Raw Data')
plt.legend(loc=3)
plt.xlabel('Episode')
plt.ylabel('Score')
plt.grid(True)
plt.show()