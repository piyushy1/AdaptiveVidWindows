from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection='3d')

# Data for three-dimensional scattered points

batch= [1,5,10,15,20,25,30,50,100]
throughput= [100,150,170,250,260,265,270,272,275]
latency= [5,15,20,22,24,30,35,38,40]
memory = [2,3,4,8,10,14,15,17,19]
cpu = [1,2,3,4,5,6,7,8,9]
#zdata = 15 * np.random.random(100)
#xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
#ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
img =ax.scatter3D(memory, latency,batch, c= memory, cmap= plt.hot(), s= throughput)
ax.set_xlabel('Throughput')
ax.set_ylabel('Latency')
ax.set_zlabel('Batch')
fig.colorbar(img)

plt.show()

# import pandas as pd
# import plotly
# import plotly.graph_objs as go
#
#
# #Read cars data from csv
# #data = pd.read_csv("cars.csv")
#
# #Set marker properties
# #markercolor = data['city-mpg']
#
# #Make Plotly figure
# fig1 = go.Scatter3d(x= throughput,
#                     y= latency,
#                     z= batch,
#                     marker=dict(size=cpu,
#                                 color=memory,
#                                 opacity=1,
#                                 reversescale=False,
#                                 colorscale='Blues'),
#                     line=dict(width=0.02),
#                     mode='markers')
#
# #Make Plot.ly Layout
# mylayout = go.Layout(scene=dict(xaxis=dict( title="throughput"),
#                                 yaxis=dict( title="latency"),
#                                 zaxis=dict(title="batch")),)
#
# #Plot and save html
# plotly.offline.plot({"data": [fig1],
#                      "layout": mylayout},
#                      auto_open=True,
#                      filename=("4DPlot.html"))





