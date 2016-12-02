import os, io, sys
import numpy as np

filename = sys.argv[1]
print sys.argv

filename = filename.replace(' ', '')[:-4].upper()
filename = "analysis/"+filename+"_test"+".csv"
print filename

f = open(filename, 'r')
f = np.array(f)
print(f)

with io.FileIO(filename, "w") as file:
    file.write("Episode, Score\n")

for e in range(0, len(f[0])):
    print "Episode %d finished with score of %d" % (e+1, f[1][e])
    with io.FileIO(filename, "a") as file:
        file.write("%d, %d\n" % (e+1, f[1][e]))
f.close()