'''
    일반적인 머신러닝 예시

    - 섭씨온도 - 화씨온도 변환기
'''


'''
    일반적인 데이터를 통한 학습
'''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
import numpy as np
import matplotlib.pyplot as plt

# 온도 변환 함수 작성
def cel_to_fah(x):
  return x*1.8+32

# 온도 데이터 만들기
data_C = np.array(range(0,100))
data_F = cel_to_fah(data_C)
# print(data_C)
# print(data_F)

# 온도 데이터 정규화
scaled_data_C = data_C / 100
scaled_data_F = data_F / 100
# print(scaled_data_C)
# print(scaled_data_F)

# 모델 생성 (모델 summary 확인)
model = Sequential()
model.add(InputLayer(input_shape=(1,)))
model.add(Dense(1))
model.compile(loss='mse', optimizer='rmsprop')
print(model.summary())

# 학습 전 예측 결과 확인
print('학습 전 ', model.predict([0.01]))

# 모델 학습
fit_hist = model.fit(scaled_data_C,
                     scaled_data_F,
                     epochs=1000,
                     verbose=0)

# 학습 후 예측결과 확인
print('학습 후 ',model.predict([0.01]))

plt.plot(fit_hist.history['loss'])
plt.show()



'''
    잡음이 있는 데이터를 통해 학습
'''
# 평균이 0, 표준편차가 0.05인, 100개 데이터
# 표준편차 : 전체 분포의 좌우 3분위에 99% 데이터가 포함
noise = np.array(np.random.normal(0, 0.05, 100))
# print(noise)

# 변환 결과 데이터 노이즈 삽입
noised_scaled_data_F = np.array([])
for data in scaled_data_F:
  noised_scaled_data_F = np.append(noised_scaled_data_F,
                                   np.random.normal(0, 0.05, 100) + data)

# 변환 대상 데이터 개수 증가
#  - 같은 데이터를 100개씩 만듦
noised_scaled_data_C = []
for data in range(0, 100):
  for i in range(0, 100):
    noised_scaled_data_C.append(data)
noised_scaled_data_C = np.array(noised_scaled_data_C)
noised_scaled_data_C = noised_scaled_data_C / 100

# 산점도 확인
plt.scatter(noised_scaled_data_C, noised_scaled_data_F)
plt.show()

# 시각화 확대 그림
fig = plt.figure(figsize=(50,50))
ax = fig.add_subplot(111)
ax.scatter(x=noised_scaled_data_C,
           y= noised_scaled_data_F,
           alpha = 0.2,
           s=200,
           marker='+')
plt.show()


# 새로운 모델 (노이즈 섞인 데이터 학습위한 모델)
model2 = Sequential()
model2.add(InputLayer(input_shape=(1,)))
model2.add(Dense(1))
model2.compile(loss='mse', optimizer= 'rmsprop')
print(model2.summary())

#  학습 전 예측결과 확인
print('Noise 학습 전', model2.predict([0.01]))

#  학습 수행
fit_hist = model2.fit(noised_scaled_data_C,
                      noised_scaled_data_F,
                      epochs = 30,
                      verbose=0)
#  학습결과 시각화
plt.plot(fit_hist.history['loss'])
plt.show()

#  학습 후 예측결과 확인
print('Noise 학습 후', model2.predict([0.01]))


'''
    학습의 과정 풀이
'''
'''
파라미터가 많아지면 수식이 복잡해지므로 연쇄법칙(chain rule) 적용
- 연쇄법칙 : 합성함수의 미분은 합성함수를 구성하는 각 함수의 미분의 곱으로 나타낼 수 있다

z = t^2 , t = x+y 일 때,
 합성함수 z의 미분(∂z/∂x)은 합성함수를 구성하는 각 함수(z=t^2, t=x+y)의
 미분의 곱(∂z/∂t * ∂t/∂x)으로 나타낼 수 있음 (∂t가 약분됨)
'''
import numpy as np
def celc2fahr(x):
  return x * 1.8 + 32

#  덧셈연산 그래프
class add_graph:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y
        return out

    # 미분 - 추가함수
    def backward(self, dout):
        dx = 1 * dout
        dy = 1 * dout
        return dx, dy

# 곱셈 연산 그래프
class mul_graph:
    def __init__(self):
        self.x = None
        self.y = None
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out
   # 미분 - 추가함수
    def backward(self, dout):
        dx = self.y * dout
        dy = self.x * dout
        return dx, dy

#  평균제곱오차 평균 클래스
class mse_graph:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
        self.x = None

    # mse를 구하는 함수
    # y = 결괏값(실제값), t = target 값(예측값)
    def forward(self, y, t):
        self.t = t
        self.y = y
        self.loss = np.square(self.t - self.y).sum() / self.t.shape[0]
        return self.loss

    def backward(self, x, dout=1):
        data_size = self.t.shape[0]
        dweight_mse = (((self.y - self.t) * x).sum() * 2 / data_size)
        dbias_mse = (self.y - self.t).sum() * 2 / data_size
        return dweight_mse, dbias_mse


# 곱셈그래프와 덧셈그래프 객체 생성
weight_graph = mul_graph()
bias_graph = add_graph()

# random normal은 정규분포 확률
# random uniform은 모든 확률 동일
weight = np.random.uniform(0, 5, 1)
print(weight)

bias = 0
data_C = np.array(range(0, 100))
data_F = celc2fahr(data_C)

# 데이터 생성
scaled_data_C = data_C / 100
scaled_data_F = data_F / 100
print(scaled_data_C)
print(scaled_data_F)

#  순전파 계산
weighted_data = weight_graph.forward(weight, scaled_data_C)
predict_data = bias_graph.forward(weighted_data, bias)
print(predict_data)

#  역전파 계산
dout = 1
dbias, dbiased_data = bias_graph.backward(dout)
dweight, dscaled_data_C = weight_graph.backward(dbiased_data)
print('dbias :', dbias)
print('dweight : ', dweight)

'''
mse : 오차제곱평균, 오차값(실제값 - 예측값)을 제곱해서 모든 오차값들의 평균값을 구함
    단, mse의 그래프는 대칭임으로 방향은 알 수 없음
weight-mse : mse값을 미분해서 -이면 오른쪽으로, +이면 왼쪽으로 움직이게 함
'''
# mse 값 구하기
mseGraph = mse_graph()
mse = mseGraph.forward(predict_data, scaled_data_F)
print(mse)

# mse의 미분값(기울기) 구하기
weight_mse_gradient, bias_mse_gradient = mseGraph.backward(scaled_data_C)
print(weight_mse_gradient)
print(bias_mse_gradient)

'''
lerning rate : 기울기 0지점 찾기위해 weight-mse 적용했을 때,
    mse 값을 너무 큰 수준으로 변경하면 오차값이 발산함, 적절한 수준으로 조절

weight와 bias가 동시에 수정되서 서로에게 영향을 미쳐 오차값을 감소시길 수 없음.

어떤 값이 결괏값에 영향을 더 많이 미치는지는 모르기 때문에
이를 특정하기 위해 weight에 대한 미분값과, bias에 대한 미분값이 필요
'''

# weight에 대한 mse 미분값
learning_rate = 0.1
learning_weight = weight - learning_rate * weight_mse_gradient * np.average(dweight)
print('before learning weight : ', weight)
print('after learning weight : ', learning_weight)

# bias에 대한 mse 미분값
learned_bias = bias - learning_rate * bias_mse_gradient * dbias
print('before learning bias : ', bias)
print('after learning bias : ', learned_bias)

# weight와 bias의 변화상태 확인
# epochs 횟수 지정과 같은 작용,
error_list = []
weight_list = []
bias_list = []
for i in range(1000):
    # forward
    weighted_data = weight_graph.forward(weight, scaled_data_C)
    predict_data = bias_graph.forward(weighted_data, bias)
    # backward
    dout = 1
    dbias, dbiased_data = bias_graph.backward(dout)
    dweight, dscaled_data_C = weight_graph.backward(dbias)
    # mse
    mse = mseGraph.forward(predict_data, scaled_data_F)

    error_list.append(mse)
    weight_mse_gradient, bias_mse_gradient = mseGraph.backward(scaled_data_C)

    weight_list.append(weight)
    weight = weight - learning_rate * weight_mse_gradient * np.average(dweight)

    bias_list.append(bias)
    bias = bias - learning_rate * bias_mse_gradient * dbias

weight_list.append(weight)
bias_list.append(bias)

print('weight :', weight)
print('bias :', bias)

# val_loss
print('error_list \n', error_list)

# weight
print('weight_list\n',weight_list)

# bias
print('bias_list \n',bias_list)

# 그래프로 시각화 - error
import matplotlib.pyplot as plt
plt.plot(error_list)
plt.title('error')
plt.show()

# 그래프로 시각화 - weight
plt.plot(weight_list)
plt.title('weight')
plt.show()

# 그래프로 시각화 - bias
plt.plot(bias_list)
plt.title('bias')
plt.show()