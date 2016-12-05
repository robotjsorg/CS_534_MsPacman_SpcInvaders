import numpy as np
import os

N = 100

for file in os.listdir('./analysis'):
    if file.endswith('.csv'):
        d = np.loadtxt(open('analysis/'+file, 'rb'), delimiter=',', skiprows=1)
        file = file.replace(' ', '')[:-4].lower()
        cutoff = np.shape(d[:,1])[0]-N
        last100 = d[:,1][cutoff:]
        print file+': '+str(np.mean(last100))+' +/- '+str(np.std(last100))