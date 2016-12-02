import numpy as np
import os

for file in os.listdir('./analysis'):
    if file.endswith('.csv'):
        d = np.loadtxt(open('analysis/'+file, 'rb'), delimiter=',', skiprows=1)
        file = file.replace(' ', '')[:-4].lower()
        print file+': '+str(np.mean(d[:,1]))+' +/- '+str(np.std(d[:,1]))