'''
    MNIST fashion Data Classifier

    - DNN 모델 구성
'''


import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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


# 28*28 = 784개의 input을 받아 출력 10의 다중분류 모델 구현하기 위해
# 60000*28*28 의 데이터를 60000*784 형태로 변형

# train 데이터를 60000*784 형태로 변형
x_train = X_train.reshape(-1, 28*28)
x_test = X_test.reshape(-1, 28*28)

# scaling : 각 data를 255로 나누면 0~1 사이의 값으로 변환됨
x_train = x_train/255
x_test = x_test/255

print(x_train.shape)


# modeling
model = Sequential()

# method 01 : input_dim 적용
# model.add(Dense(784, input_dim = 784, activation='relu'))

# method 02 : imput_shape 적용
model.add(Dense(784, input_shape=(784,), activation='relu'))

model.add(Dense(516, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
print(model.summary())


# compiling
# model.compile(loss='mse', optimizer='adam', metrics=['categorical_accuracy'])
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])

# 손실함수 종류 : 'mse', 'categorical_crossentropy'
# optimizer 종류 : adam, rmsprop, adagrad ....
# metrics 종류 : 분류 문제에는 주로 accuracy, categorical_accuracy

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


my_sample = np.random.randint(10000)
plt.imshow(X_test[my_sample])
print('Sample is : ',label[Y_test[my_sample]])
pred = model.predict(x_test[my_sample].reshape(-1, 784))
print(pred)
print('Prediction is : ', label[np.argmax(pred)])


'''
loss : 0.6706947088241577
accuracy : 0.8866000175476074

Sample is :  coat

[[6.3823529e-05 7.4269745e-05 1.3205617e-02 3.0095622e-01 6.4892197e-01
  5.0031979e-12 3.6127426e-02 1.5336304e-06 6.4910972e-04 3.5689014e-11]]

Prediction is :  coat
'''