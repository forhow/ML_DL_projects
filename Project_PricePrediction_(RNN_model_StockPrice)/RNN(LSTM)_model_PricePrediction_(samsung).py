'''
    Stock Price Prediction

    - 삼성전자 주가 예측

    - RNN 모델 구성
     * 직전 28일간 종가 Data 활용, 다음날 주가를 예측
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers  import Adam

# 데이터 업로드 및 dataframe 로드

raw_data = pd.read_csv('../datas/Samsung.csv')
print(raw_data.head())

print(raw_data.tail())

print(raw_data.info())


# 전처리 수행
# - na 값 처리

# 종가 예측 위한 target data 분리
data_close = raw_data[['Close']]
print(data_close.head())

# data 정렬 및 결측치 삭제 위한 nan 데이터 확인
data_close = data_close.sort_values('Close')
print(data_close.head())
print(data_close.tail())

# date : object로 지정된 type을 date type으로 수정
# 자료의 순서를 지정해서 입력으로 주기 위해 date time으로 수정함

raw_data['Date'] = pd.to_datetime(raw_data['Date'])
raw_data.set_index('Date', inplace=True)
print(raw_data.head())

# 시각화 확인
raw_data['Close'].plot()
plt.show()

# 특정 기간의 종가 데이터만 추출
data = raw_data['2019-06-15':'2020-06-14'][['Close']]
print(data)
print(data.info())

# nan 값은 계산은 되나 계산 결과는 nan 으로 반환 됨
data = data.dropna()
print(data.info())

data.plot()
plt.show()

# scaling
# - 주가가격 등 값의 의미가 있는 경우에 표준화 하면 안되고
# - 값의 크기가 비례적으로 남아있을 수 있도록 MinMaxScaler를 사용

from sklearn.preprocessing import MinMaxScaler

minmaxscaler = MinMaxScaler()
scaled_data = minmaxscaler.fit_transform(data)
print(scaled_data)
# scaling 수행 후 dataframe은 깨지고 np array로 변환됨

print(scaled_data[:1])
print(scaled_data.shape)


# 28개씩 의 데이터를 입력
# 각 데이터 끝의 다음 데이터를 target data로 지정

sequence_X = []
sequence_Y = []

for i in range(len(scaled_data)-28):
  _x = scaled_data[i:i+28]
  _y = scaled_data[i+28]
  if i is 0:
    print(_x, '->', _y)
  sequence_X.append(_x)
  sequence_Y.append(_y)

sequence_X = np.array(sequence_X)
sequence_Y = np.array(sequence_Y)

# print(sequence_X[1])
# print(sequence_Y[1])
print(sequence_X.shape)
print(sequence_Y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(sequence_X, sequence_Y, test_size=0.2)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

# LSTM 사용시 activation 함수는 hyperbolic tangent 사용
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2]), activation= 'tanh'))
model.add(Flatten())
# 값 자체를 예측하기 때문에 마지막 단에 activation function 사용하지 않음
model.add(Dense(1))

model.compile(loss='mse', optimizer= 'adam')

print(model.summary())

fit_hist = model.fit(X_train, Y_train, epochs=300,
                     validation_data=(X_test, Y_test), shuffle=False)

plt.plot(fit_hist.history['loss'])
plt.plot(fit_hist.history['val_loss'])
plt.title('Decrease of Loss')
plt.show()

# 예측 test
predict = model.predict(X_test)

plt.plot(Y_test, label = 'actual')
plt.plot(predict, label='predict')
plt.title('Price Prediction')
plt.legend()
plt.show()
