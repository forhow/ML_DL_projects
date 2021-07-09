'''
    Heart Disease Classifier

    여러 의학적 속성값을 바탕으로 환자의 심장병 여부를 판단
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

column_name = ['age', 'sex', 'cp', 'treshbps', 'chol', 'fbs', 'restecg',
               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'hsl', 'HeartDisese']


raw_data = pd.read_excel('./datas/heart-disease.xlsx', header=None,
                         names = column_name)
# head / tail의 data 확인
print(raw_data.head(5))
print(raw_data.tail(5))
print(raw_data.describe())
print(raw_data.info())

# dtypes: float64(1), int64(10), object(3)에서 object는 계산할 수 없는 범주임
# 파일 확인해보면 '?'로 들어간 결측으로, 어떻게 전처리 할 것인지 결정해야 함
# - raw를 버리는 방향으로 진행

clean_data = raw_data.replace('?', np.nan)
# 숫자가 아니지만 숫자로 처리되고, 연산결과는 nan이 됨

# nan 값을 찾아 삭제
clean_data = clean_data.dropna()
print(clean_data.info())


# train data와 target data 분리
# column_name에서 마지막 data 삭제하고 pop 한 데이터를 return
# column_name.append('HeartDisese')
keep = column_name.pop()
print(keep)
print(column_name)

training_data = pd.DataFrame(clean_data.iloc[:,:13], columns=column_name)
target = pd.DataFrame(clean_data.iloc[:,-1], columns=[keep])
print(training_data.head())
print(target.head())

# target 분석 ; 심장병 환자의 수
print(target['HeartDisese'].sum())

# target 분석 ; 심장병 유병률
print(target['HeartDisese'].mean())

# 데이터 자체가 편향성을 가지고 있는 경우(특정 데이터의 희소성이 있는 경우) 보정이 필요
# ex) 전체 인구에 대한 희귀병에 대한 정확도는 99.9%로 음성으로 판단해버리는 경우 있음
# 일반 case와 유병 case를 적절한 비율로 맞춰서 데이터 준비


# scaling 필요
# 나이와 콜레스테롤 수치를 봤을 때, 그대로 학습시키면 콜레스테롤 수치가 유병판단을 좌우함
# 정규분포, 표준정규분포 등의 값으로
# minmax - 최소0 최대 1
# normalizing - 정규분포
# standardscale - 표준정규분포


# StandardScaler
# 표준정규분포를 따르는 형태로 표준화시킴
# 평균 0, 표준편자 1
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(training_data)

# scaling 수행하면 dataframe 형태가 풀리기 때문에 다시 dataframe 화
scaled_data = pd.DataFrame(scaled_data, columns=column_name)

# describe 형태를 Transpose 해서 출력
print(scaled_data.describe().T)


# 테스트 데이터 분리
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_data,
                                                    target,
                                                    test_size = 0.30)
print('X_train shape :', X_train.shape)
print('X_test shape :', X_test.shape)
print('y_train shape :', y_train.shape)
print('y_test shape :', y_test.shape)


# 모델 생성
model = Sequential()
# 13 * 512 +512 = 14 * 512
model.add(Dense(512, input_dim= (13), activation='relu'))
# 1차 테스트 후 과적합 확인되어 dropout 적용
model.add(Dropout(0.2))
# 512 * 256 + 256
model.add(Dense(256, activation='relu'))
# 1차 테스트 후 과적합 확인되어 dropout 적용
model.add(Dropout(0.2))
# 256 * 128 + 128
model.add(Dense(128, activation='relu'))
# 1차 테스트 후 과적합 확인되어 dropout 적용
model.add(Dropout(0.2))
# 128 * 1 + 1
model.add(Dense(1, activation='sigmoid'))
print(model.summary())


# compile
model.compile(loss='mse', optimizer='adam', metrics = ['binary_accuracy'])

# 매 학습마다 기록이 저장되는 변수 선언 및 학습 수행
fit_hist = model.fit(X_train,
                     y_train,
                     epochs=100,
                     batch_size= 50,
                     validation_split=0.2)


# fit_hist에 저장되는 결과 종류
# - ('loss', 'binary_accuracy', 'val_loss', 'val_binary_accuracy')
print(tuple(fit_hist.history))


# 시각화 확인
# - blue : binary_accuracy
# - orange : val_binary_accuracy
plt.plot(fit_hist.history['binary_accuracy'])
plt.plot(fit_hist.history['val_binary_accuracy'])
plt.title('Train ACC Vs. Val_ACC')
plt.legend(('TrACC','valACC'))
plt.show()

# 모델의 평가
score = model.evaluate(X_test, y_test, verbose=0)
print(score)
# evaluate 함수 결과는 (loss, accuraccy) 로 반환
print('Keras DNN model loss :', score[0])
print('Keras DNN model accuracy :', score[1])

'''
    Keras DNN model loss : 0.15440785884857178
    Keras DNN model accuracy(F1 score) : 0.8014981150627136
'''