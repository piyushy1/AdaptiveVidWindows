from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection='3d')

# Data for three-dimensional scattered points

batch= [1,5,10,25,30,40,50,60,70,80,90,100]
throughput1= [100,150,170,250,260,265,270,272,275]
latency1= [5,15,20,22,24,30,35,38,40]
memory = [2,3,4,8,10,14,15,17,19]
cpu = [1,2,3,4,5,6,7,8,9]
throughput2= [105,155,175,252,270,290,300,310,325]
latency2 = [10,20,25,35,44,50,55,58,60]

#zdata = 15 * np.random.random(100)
#xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
#ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
#img =ax.scatter3D(throughput, latency,batch, s= memory, c= cpu)
ax.scatter3D(batch,throughput1, latency1, s= memory, c = '#2CBDFE')
ax.scatter3D(batch,throughput2, latency2, s= memory, c = '#F5B14C')
ax.set_xlabel('Batch')
ax.set_ylabel('Throughput')
ax.set_zlabel('Latency')
#fig.colorbar(img)

plt.show()