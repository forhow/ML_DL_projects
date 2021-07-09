'''
    학습의 원리
    - fit 함수 처리 과정
    - 밑바닥부터 시작하는 딥러닝 148 page ~
'''


import numpy as np


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


apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

# 계산그래프 객체 생성
mul_apple_graph = mul_graph()
mul_orange_graph = mul_graph()
add_apple_orange_graph = add_graph()
mul_tax_graph = mul_graph()

# 순전파 계산 실행
apple_price = mul_apple_graph.forward(apple, apple_num)
orange_price = mul_orange_graph.forward(orange, orange_num)
all_price = add_apple_orange_graph.forward(apple_price, orange_price)
total_price = mul_tax_graph.forward(all_price, tax)
print(total_price)

# 계산 결과에 대한 미분계산 (역전파 계산)
# - 각 요소들이 결과에 얼마나 영향을 미치는지 확인
# - 영향도 크기에 따라 값을 조정하기 위함

# total price 미분
dprice = 1

# all price 미분 (dout = 1로 설정)
dall_price, dtax = mul_tax_graph.backward(dprice)
print('dall_price, dtax ', dall_price, dtax, '\n')

# apple price, orange price에 대한 미분
dapple_price, dorange_price = add_apple_orange_graph.backward(dall_price)
print('dapple_price, dorange_price', dapple_price, dorange_price, '\n')

# orange가격과 orange 개수에 대한 미분
dorange, dorange_num = mul_orange_graph.backward(dorange_price)
print('dorange, dorange_num', dorange, dorange_num, '\n')

# apple가격과 apple 개수에 대한 미분
dapple, dapple_num = mul_apple_graph.backward(dapple_price)
print('dapple, dapple_num', dapple, dapple_num, '\n')

# 각 요소가 1만큼 증가했을 때 전체 값에 영향을 끼치는 정도를 표현함
# ex) 사과 가격 1이 증가하면 총액에 2.2 가 증가함
print('dApple', dapple)
print('dApple_num', dapple_num)
print('dOrange', dorange)
print('dOrange_num', dorange_num)

''' 출력결과
dall_price, dtax  1.1 650

dapple_price, dorange_price 1.1 1.1

dorange, dorange_num 3.3000000000000003 165.0

dapple, dapple_num 2.2 110.00000000000001

dApple 2.2
dApple_num 110.00000000000001
dOrange 3.3000000000000003
dOrange_num 165.0
'''