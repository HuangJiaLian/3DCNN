'''
@Description: 可视化3D的数据
@Author: Jack Huang
@Github: https://github.com/HuangJiaLian
@Date: 2019-09-22 17:42:20
@LastEditors: Jack Huang
@LastEditTime: 2019-09-22 23:03:47
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import h5py


def show3d_character(img_3d, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 15)
    ax.set_zlim(0, 15)
    ax.set_title(title)
    box_size = 16
    x = []
    y = []
    z = [] 
    for i in range(box_size):
        for j in range(box_size):
            for k in range(box_size):
                if img_3d[j,i,k] > 0.1:
                    x.append(i)
                    y.append(j)
                    z.append(k)
                
    img = ax.scatter( z, x, y, c=img_3d[x,y,z],s=10, cmap=plt.hot())
    fig.colorbar(img)
    plt.show()


dataset = h5py.File('./input/full_dataset_vectors.h5', 'r')

x_test = dataset["X_test"][:]
y_test = dataset["y_test"][:]

for index in range(10,40):
    img_3d = np.reshape(x_test[index],(16,16,16))
    img_3d_label = y_test[index]
    show3d_character(img_3d, str(img_3d_label) + ' in 3D')