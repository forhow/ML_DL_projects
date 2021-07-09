'''
    Keras를 이용한 분류 - AND, OR, XOR 문제 해결

      - 선형회귀 활용 : 입력값에 대한 결과값을 예측


    XOR (배타적 논리합) : 값이 서로 다를 때만 1, 같으면 0

    numpy : python은 효율이 떨어지고 성능이 떨어지는 언어이나 numpy에서 행렬계산을 지원함으로써 자원을 효율적으로 사용하도록 지원함
'''

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Train data
train_data = np.array([[0,0],[0,1],[1,0],[1,1]], 'float32')

'학습할 연산 선택 (주석 해제)'
# 1. xor
target_data = np.array([[0],[1],[1],[0]], 'float32')
# # 2. and
# target_data = np.array([[0],[0],[0],[1]], 'float32')
# # 3. or
# target_data = np.array([[0],[1],[1],[1]], 'float32')

'Modeling'
#  sequential 객체 생성
model = Sequential()

# input 2개의 data를 32개의 출력을 갖는 Fully connect 층으로 연결
model.add(Dense(32, input_dim = 2, activation='relu'))
# 각 노드마다 입력 2개 (64개)와 각 노드마다 bias 32개 = 총 96개

# 1개의 출력을 갖는 fully connect
model.add(Dense(1, activation='sigmoid'))
# 노드에 입력 32개, bias 1개 = 총 33개
# 0.5 이상이면 True, 0.5 이하이면 False

model.compile(loss='mse', optimizer='adam', metrics=['binary_accuracy'])
print(model.summary())


# 학습 수행
fit_hist = model.fit(train_data,
                     target_data,
                     epochs=500,
                     verbose=0)
# binary accuracy는 분류가 정확하게 이뤄졌는지 확인
# loss 감소량은 최적의 선을 찾기 위한 지표
# loss : 분류가 정확하게됐을 때 그 선이 최적화된 선인지를 구분하는 용도로 사용

# XOR 문제는 차원을 늘리면서 해결가능, 모델은 32차원으로 늘리면서 해결하도록 함

# 하지만 층을 점차 늘리면서 기울기 소실문제 발생, relu를 사용함으로써 해결


inp = list(map(int, input('연산대상 입력 예) 1 1').split()))
qwe = np.array(inp)
print('입력값 : ', qwe)
qwe = qwe.reshape(1,2)
print('resahpe : ', qwe)
print('결과값 : ', model.predict(qwe)[0][0].round())

'''
target data로 학습하도록 해서 xor, or, and 등 연산가능
'''