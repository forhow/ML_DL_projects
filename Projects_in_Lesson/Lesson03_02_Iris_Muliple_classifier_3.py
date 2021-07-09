'''
    Iris 다중 분류
'''

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# 데이터 확인
# key와 list가 들어있는 Dict. like의 bunch 객체
iris = load_iris()
print(type(iris))
print('-------- Data Shape-------')
print('Data', iris.data.shape)
print('Label', iris.target.shape)
print('First five data : \n', iris.data[0:5])
print('Frist five label : \n', iris.target[0:5])
print('iris dataset keys\n', iris.keys())

print(iris.target_names)
print(iris.feature_names)
# ['setosa' 'versicolor' 'virginica']
# 0 - setosa, 1 -versicolor, 2-virginica
# sepal : 꽃 받침
# petal : 꽃 잎
# 데이터는 각각 50개씩 순서대로 정렬되어 있음


# training 할 데이터와 target을 지정
x = iris.data
y = iris.target.reshape(-1,1)
print(y.shape)


# target에 대한 onehot encoding
# sparse : 좌표형태로 표현, False시 희소행렬로 표현
encoder = OneHotEncoder(sparse=False)
encoded_y = encoder.fit_transform(y)
print(encoded_y.shape)
print(encoded_y[48:55])

# train / test data split
X_train, X_test, y_train, y_test = train_test_split(x,
                                                    encoded_y,
                                                    test_size = 0.2,)
print('X_train shape :', X_train.shape)
print('X_test shape :', X_test.shape)
print('y_train shape :', y_train.shape)
print('y_test shape :', y_test.shape)


# 데이터 수치가 크지 않기 때문에 scaling의 skip

# modeling
model = Sequential()
model.add(Dense(256, input_dim = 4, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(3, activation='softmax'))
# 다중분류에서는 마지막 함수를 softmax로 사용


# compile
# - adam에 learning rate 적용 위해 Adam 객체 생성
opt = Adam(lr=0.001)
model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
# 다중분류의 경우 - loss='categorical_crossentropy'
# metrics - 2진 분류일 경우 - binary_accuracy
#         -  그외 accuracy
print(model.summary())

# 모델 학습
fit_hist = model.fit(X_train, y_train,batch_size=5, epochs=10, verbose=1)

# 모델 평가
score = model.evaluate(X_test, y_test, verbose=0)
print('loss : ', score[0])
print('ACC :', score[1])

# 정확도 시각화
plt.plot(fit_hist.history['accuracy'])
plt.show()


# 직접 test
labels = iris.target_names
my_sample = np.random.randint(30)
sample = X_test[my_sample]  # [4.7 3.2 1.6 0.2]

# 모델에 데이터 줄 경우 리스트로 줘야 함
# 모델의 예측에 필요한 데이터 개수가 4개인데 데이터를 그대로 줄경우 첫 번째 데이터만 입력됨
# 4개의 데이터를 하나의 리스트로 전달하기 위해 차원을 늘려 input data로 가공함
sample = sample.reshape(1,4)   # [[4.7 3.2 1.6 0.2]]

pred = model.predict(sample)
print(pred)
print(y_test[my_sample])
print('Target : ', labels[np.argmax(y_test[my_sample])])
print('Prediction after learning is : ', labels[np.argmax(pred)])
