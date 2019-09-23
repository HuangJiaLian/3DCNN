'''
@Description: 
@Author: Jack Huang
@Github: https://github.com/HuangJiaLian
@Date: 2019-09-22 14:51:12
@LastEditors: Jack Huang
@LastEditTime: 2019-09-22 17:25:20
'''

from keras.layers import Conv3D, MaxPool3D, Flatten, Dense
from keras.layers import Dropout, Input, BatchNormalization
from sklearn.metrics import confusion_matrix, accuracy_score
from plotly.offline import iplot, init_notebook_mode
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta
import plotly.graph_objs as go
from matplotlib.pyplot import cm
from keras.models import Model
import numpy as np
import keras
import h5py


# 读入数据
dataset = h5py.File('./input/full_dataset_vectors.h5', 'r')

x_train = dataset["X_train"][:]
x_test = dataset["X_test"][:]

y_train = dataset["y_train"][:]
y_test = dataset["y_test"][:]


print ("x_train shape: ", x_train.shape)
print ("y_train shape: ", y_train.shape)

print ("x_test shape:  ", x_test.shape)
print ("y_test shape:  ", y_test.shape)

# 观察输入数据: 将三维数据 flatten 成了一维数据
# x_train shape:  (10000, 4096) 训练集:10000张3d图片, 每一张3d图片的形状是16x16x16
# y_train shape:  (10000,) 
# x_test shape:   (2000, 4096)  测试集:2000张3d图片,  每一张3d图片的形状是16x16x16
# y_test shape:   (2000,)

# 可视化数据
# with h5py.File("./input/train_point_clouds.h5", "r") as points_dataset:        
#     digits = []
#     for i in range(10):
#         digit = (points_dataset[str(i)]["img"][:], 
#                  points_dataset[str(i)]["points"][:], 
#                  points_dataset[str(i)].attrs["label"]) 
#         digits.append(digit)
        
# x_c = [r[0] for r in digits[0][1]]
# y_c = [r[1] for r in digits[0][1]]
# z_c = [r[2] for r in digits[0][1]]
# trace1 = go.Scatter3d(x=x_c, y=y_c, z=z_c, mode='markers', 
#                       marker=dict(size=12, color=z_c, colorscale='Viridis', opacity=0.7))

# data = [trace1]
# layout = go.Layout(height=500, width=600, title= "Digit: "+str(digits[0][2]) + " in 3D space")
# fig = go.Figure(data=data, layout=layout)
# iplot(fig)
    
# 要使用2D的卷积，我们首先将每一张图片转化成3D的形状: width, height, channel(r/g/b).
# 要使用3D的卷积，我们首先将每一张图片转化成4D的形状: length, breadth, height, channel(r/g/b).

# np.ndarray的意思是N dimensional array  
## Introduce the channel dimention in the input dataset 
xtrain = np.ndarray((x_train.shape[0], 4096, 3)) # 这里的(10000, 4096, 3)是ndarray的形状，随机初始化
xtest = np.ndarray((x_test.shape[0], 4096, 3))
print('x_train.shape[0]', x_train.shape[0]) # 10000
print('x_test.shape[0]', x_test.shape[0]) # 2000

## 这里有点晕，到时可以只用一个通道便好
## iterate in train and test, add the rgb dimention 
def add_rgb_dimention(array):
    scaler_map = cm.ScalarMappable(cmap="Oranges")
    array = scaler_map.to_rgba(array)[:, : -1]
    return array
for i in range(x_train.shape[0]):
    xtrain[i] = add_rgb_dimention(x_train[i])
for i in range(x_test.shape[0]):
    xtest[i] = add_rgb_dimention(x_test[i])

## convert to 1 + 4D space (1st argument represents number of rows in the dataset)
xtrain = xtrain.reshape(x_train.shape[0], 16, 16, 16, 3)
xtest = xtest.reshape(x_test.shape[0], 16, 16, 16, 3)

# Label变成One-Hot的
## convert target variable into one-hot
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# (10000,10)
print(y_train.shape)


# 搭建神经网络结构
## input layer
input_layer = Input((16, 16, 16, 3))

## convolutional layers
conv_layer1 = Conv3D(filters=8, kernel_size=(3, 3, 3), activation='relu')(input_layer)
conv_layer2 = Conv3D(filters=16, kernel_size=(3, 3, 3), activation='relu')(conv_layer1)

## add max pooling to obtain the most imformatic features
pooling_layer1 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer2)

conv_layer3 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')(pooling_layer1)
conv_layer4 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu')(conv_layer3)
pooling_layer2 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer4)

## perform batch normalization on the convolution outputs before feeding it to MLP architecture
pooling_layer2 = BatchNormalization()(pooling_layer2)
flatten_layer = Flatten()(pooling_layer2)

## create an MLP architecture with dense layers : 4096 -> 512 -> 10
## add dropouts to avoid overfitting / perform regularization
dense_layer1 = Dense(units=2048, activation='relu')(flatten_layer)
dense_layer1 = Dropout(0.4)(dense_layer1)
dense_layer2 = Dense(units=512, activation='relu')(dense_layer1)
dense_layer2 = Dropout(0.4)(dense_layer2)
output_layer = Dense(units=10, activation='softmax')(dense_layer2)

## define the model with input layer and output layer
model = Model(inputs=input_layer, outputs=output_layer)


model.compile(loss=categorical_crossentropy, optimizer=Adadelta(lr=0.1), metrics=['acc'])
model.fit(x=xtrain, y=y_train, batch_size=128, epochs=50, validation_split=0.2)

pred = model.predict(xtest)
pred = np.argmax(pred, axis=1)
print(pred) 