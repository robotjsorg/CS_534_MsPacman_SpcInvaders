import matplotlib.pyplot as plt
import numpy as np

# read the input data file
d = np.loadtxt(open("analysis/rl0.csv","rb"),delimiter=",",skiprows=1)
d = np.transpose(d)
print(d[1])

# make a plot of it
# plt.plot(d[1], marker="o", linestyle="")
# plt.show()

# make a moving average n = 10
a = []
mv_avg = 0
for i in range(0, len(d[1])):
	if i > 10:
		mv_avg = 
		a.append(mv_avg)
	else:
		a.append(d[1][i])

# plot it again
plt.plot(d[1], marker="o", linestyle="")
plt.show()