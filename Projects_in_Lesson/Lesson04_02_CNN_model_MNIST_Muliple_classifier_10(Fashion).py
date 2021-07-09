'''
    MNIST fashion Data Classifier

    - CNN 모델 구성

    * 이미지를 인식하기 위해서는 픽셀의 위치와 다른 픽셀과의 관계가 유용한 정보임
    * DNN은 단순히 픽셀을 늘어놓은 상태로 데이터를 입력받아 분석했지만 실제 이미지의 픽셀간의 상대적인 정보를 수용할 수 없음 - 중요정보의 손실
    * CNN은 각 픽셀의 위치와 관계를 그대로 사용하기 위한 방법임
'''


import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Activation, MaxPool2D, Dropout
from tensorflow.keras.optimizers import Adam
from keras.utils import np_utils # onehot encoding 용
from tensorflow.keras import datasets

# dataset split
(X_train, Y_train), (X_test, Y_test) = datasets.fashion_mnist.load_data()

print('X_train shape :', X_train.shape)
print('X_test shape :', X_test.shape)
print('Y_train shape :', Y_train.shape)
print('Y_test shape :', Y_test.shape)

label = 'Tshirt trouser pullover dress coat sandal shirt sneaker bag ankleboot'.split()
print(label)

# 임의의 샘플 확인
my_sample = np.random.randint(60000)
plt.imshow(X_train[my_sample], cmap='gray')
plt.show()
print(Y_train[my_sample])
print(X_train[my_sample])

# one hot encoding
# target data를 one hot encoding한 형태로 변형해서 새로운 변수로 저장
y_train = np_utils.to_categorical(Y_train)
y_test = np_utils.to_categorical(Y_test)


# 형태 확인
print(Y_train[4325])
print(y_train[4325])


# * CNN 모델은 기본 DNN 모델처럼 Flatten된 형태로 입력을 주지 않음
# * Nomalization : pixel의 값을 0~1 사이의 값으로 정규화
# * 차원을 늘린 상태로 reshape 수행
# (ex. 60000 x 28 x 28 ->> 60000 x 28 x 28 x 1)

# Normalization
x_train = X_train / 255
x_test = X_test / 255

# 모델에 입력되는 input을 list로 주기 위해 reshape
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
print(x_train.shape)
print(x_test.shape)


# modeling
model = Sequential()
model.add(Conv2D(32, padding= 'same',kernel_size= (3,3), input_shape=(28,28,1), activation='relu'))
model.add(MaxPool2D(padding='same', pool_size=(2,2)))
model.add(Conv2D(64, padding= 'same',kernel_size= (3,3), activation='relu'))
model.add(MaxPool2D(padding='same', pool_size=(2,2)))
model.add(Conv2D(128, padding= 'same',kernel_size= (3,3), activation='relu'))
model.add(MaxPool2D(padding='same', pool_size=(2,2)))

model.add(Flatten())
model.add(Dropout(0.2))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
print(model.summary())

'''
### Conv2D
* output : 필터의 개수= 필터가 적용된 추출 이미지의 개수
* input shape은 현재 28x28x1 이나 컬러이미지가 되면 28x28x3임
* Input 이미지와 Filter를 합성곱 연산해서 이미지의 특성을 추출
* padding : same은 출력 이미지가 입력이미지와 사이즈가 같도록 auto로 padding 적용
* stride는 default 1

### Maxpool2D
* conv 층에서 추출된 특성을 더욱 강조함
* padding : pooling 에 적용하는 커널의 크기가 맞지 않을 경우 padding을 적용
* size 축소 : 풀링 사이즈에 따라 원본이미지의 사이즈와 픽셀이 감소됨
* stride는 풀링 사이즈와 동일(겹치는 부분 없음)

### Flatten
* Dense layer에 input으로 주기 위한 reshape
* 모든 픽셀값을 일렬로 늘림

### parameters calculation
* 필터의 픽셀 + 각 node 마다 bias 1 : 3 x 3(9) x node(32) + bias (32) = 320
'''


# compiling
# model.compile(loss='mse', optimizer='adam', metrics=['categorical_accuracy'])
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])


# training
fit_hist = model.fit(x_train, y_train, epochs=20, batch_size=128,validation_split=0.2)


# evaluation
score = model.evaluate(x_test, y_test, verbose=0)
print('loss :', score[0])
print('accuracy :', score[1])

# visualization
plt.plot(fit_hist.history['categorical_accuracy'])
plt.plot(fit_hist.history['val_categorical_accuracy'])
plt.title('TR ACC Vs. VAL ACC')
plt.legend()
plt.show()


# Model Test
my_sample = np.random.randint(10000)
plt.imshow(X_test[my_sample])
print('Sample is : ',label[Y_test[my_sample]])
pred = model.predict(x_test[my_sample].reshape(-1, 28,28,1))
print(pred)
print('Prediction is : ', label[np.argmax(pred)])

'''
loss : 0.2825651168823242
accuracy : 0.9194999933242798

Sample is :  sneaker

[[4.0136686e-15 1.5471628e-18 2.6079580e-17 2.5430447e-13 7.8954146e-19
  1.8599566e-13 1.0075120e-16 1.0000000e+00 1.6844684e-15 3.6407621e-14]]
  
Prediction is :  sneaker

'''