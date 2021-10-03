from plyfile import PlyData, PlyElement
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
x = []
y = []
z = []
I = []
data = PlyData.read("./lidar_data/0.ply")
for temp_data in data.elements[0][0:2500]:
    x.append(float(temp_data[0]))
    y.append(float(temp_data[1]))
    z.append(float(temp_data[2]))
    I.append(float(temp_data[3]))

# fig = plt.figure()
# ax = Axes3D(fig)

x = np.array(x)
y = np.array(y)

z = np.array(z)
I = np.array(I)
x.shape = (50, 50)
x = np.mat(x).T
y.shape = (50, 50)
y = np.mat(y).T
I.shape = (50, 50)
I = np.mat(I).T

# ax.scatter3D(x, y, z, c = I, marker = '.', s=20, label='')
plt.contourf(x, y, I)
plt.contour(x, y, I)
plt.show()
