'''
    Stock Price Prediction

    - 셀트리온 주가 예측

    - RNN 모델 구성
     * 직전 28일간 종가 Data 활용, 다음날 주가를 예측
'''

# Module Import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

# 데이터 로드
raw_data = pd.read_csv('../datas/068270.KS.csv')
print(raw_data.columns)
print(raw_data.head())
print(raw_data.info())

"""### 전처리 필요 사항
1. Date : Datetime type 변환 및 index 적용
2. Adj. close : 수정종가 삭제- 종가로 대체
3. NaN : 결측값 삭제
"""

# 전처리 
# 1. 결측값 삭제
raw_data = raw_data.dropna()
print(raw_data.info())

# 전처리
# 2. date 형식 object-> datetime으로 변경
raw_data['Date'] = pd.to_datetime(raw_data['Date'])
print(raw_data.info())

# 전처리
# 3. 자료 순서 지정위해 date를 index로 설정
raw_data.set_index('Date', inplace=True)
print(raw_data.info())

# 전처리
# 4. 수정종가 삭제
raw_data = raw_data.drop('Adj Close', axis=1)
print(raw_data.info())

print(raw_data.columns)

# 전처리 
# 5. Scaling

from sklearn.preprocessing import MinMaxScaler
data_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

scaled_data = data_scaler.fit_transform(raw_data[[ 'Open', 'High', 'Low', 'Volume']])
scaled_target = target_scaler.fit_transform(raw_data[['Close']])

scaled_data = np.array(scaled_data)
scaled_target = np.array(scaled_target.reshape(3942,-1))

print(scaled_data.shape)
print(scaled_target.shape)

# 전처리 
# 6. Sequential data로 변환
# - 28 일간의 데이터를 학습을 위한 1개의 데이터로 변환시킴

sequence_X = []
sequence_Y = []

for i in range(len(scaled_data)-28):
  _x = scaled_data[i:i+28]
  _y = scaled_target[i+28]
  sequence_X.append(_x)
  sequence_Y.append(_y)

# 전처리
# 7. 모델에 입력하기 위한 array 형식으로 전환
sequence_X = np.array(sequence_X)
sequence_Y = np.array(sequence_Y)


# data 내용 확인
print(sequence_X[:1])
print(sequence_Y[:1])

# data 구조 확인
print(sequence_X.shape)
print(sequence_Y.shape)

# train / test set split
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(sequence_X, sequence_Y, test_size=0.2, shuffle= False )

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# modeling
# 먼저 학습한 데이터에 대한 가중치 적어져, 출력값 근처의 값들만 유의미해짐
model = Sequential()
model.add(SimpleRNN(50, input_shape=(X_train.shape[1], X_train.shape[2]), activation= 'tanh'))
model.add(Flatten())
# 값 자체를 예측하기 때문에 마지막 Layer에 activation function 사용하지 않음
model.add(Dense(1))

# Compiling
model.compile(loss='mse', optimizer= 'adam')

print(model.summary())

# Training
fit_hist = model.fit(X_train, Y_train, epochs=50, validation_data= (X_test, Y_test), shuffle= False)

# Visualization - Decrease of Loss
plt.plot(fit_hist.history['loss'], label='loss')
plt.plot(fit_hist.history['val_loss'], label= 'val_loss')
plt.legend()
plt.show()

# Prediction
predict = model.predict(X_test)

# Visualization - Prediction Vs Actual Price
plt.plot(Y_test[-30:], label='actual')
plt.plot(predict[-30:], label = 'predict')
plt.grid(True)
plt.legend()
plt.show()