'''
    Titanic Survivor - Advance
'''

import seaborn as sns
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 15)
pd.set_option('display.max_row', 500)

# source data 확인
raw_data = sns.load_dataset('titanic')
print(raw_data.head(10))
print(raw_data.info())

'''
    전처리
'''

# null 값 확인
raw_data.isnull()

# null 값의 개수 확인
raw_data.isnull().sum()

# deck column 삭제
# thresh = 500 : na개수가 500개 이상이면 삭제
# axis=1 : 열 제거
clean_data = raw_data.dropna(axis=1, thresh = 500)
print(clean_data.columns)

# age의 na값을 다른값으로 대체
# - 평균, 최소/최대, 0 등 다른 값으로 대체하는 방법이 있음
mean_age = clean_data['age'].mean()
print(mean_age)

# age의 nan값 확인 - data 5
print(clean_data.head(10))

# age column에 nan 값을 채움
# inplace = True : 원본 데이터프레임을 수정
clean_data['age'].fillna(mean_age, inplace=True)
print(clean_data.head(10))

# 중복의미의 데이터 삭제
# embark/embarked_town 중 embarked_town
# survived/alive 중 alive
clean_data.drop(['embark_town', 'alive'], axis=1, inplace=True)
print(clean_data.head(10))


# embarked에 존재하는 null 값 2개
print(clean_data['embarked'][825:830])

# 일부 값이 결측인 경우 최빈값, 이전값 등으로 대체
# method = ffill, bfill : 이전값, 이후값으로 채우는 옵션
clean_data['embarked'].fillna(method='ffill', inplace=True)
print(clean_data['embarked'][825:830])

# null 값 유무 확인
print(clean_data.isnull().sum())

# data 자료형 확인
print(clean_data.info())


# object data 변환
# replace해야 할 내용이 많은 경우 dictionary 형태로 입력 가능
clean_data['sex'].replace({'male':0, 'female':1}, inplace=True)
print(clean_data.info())

# 컬럼에 저장된 모든 값들에서 중복을 제외하고 보여줌
print(clean_data['sex'].unique())
print(clean_data['embarked'].unique())

# 범주형 데이터를 숫자로 바꾸면 관계를 다른 의미로 해석할 수 있음
# - 명목척도가 아닌 비율척도로 이해


# 바꿔야 할 데이터가 많은 경우 replace는 효율이 떨어짐
# 문자/카테고리 데이터를 변환하려면LabelEncoder(), OneHotEncoder() 로 바꿈
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
onehot_encoder = preprocessing.OneHotEncoder()

# 컬럼의 값별로 개수 확인
print(clean_data['embarked'].value_counts())
    # S    644
    # C    169
    # Q     78
    # Name: embarked, dtype: int64


# LabelEncoder() : label을 숫자로 변환
# 알파벳 순서 또는 가장 먼저 나오는 데이터 순 0, 그 다음 1, 2 순서로 변환됨

# object data인 embarked를 LabelEncoder()로 변환
clean_data['embarked'] = label_encoder.fit_transform(clean_data['embarked'])
print(clean_data['embarked'].unique())

# object 데이터인 who를 LabelEncoder()로 변환
clean_data['who'] = label_encoder.fit_transform(clean_data['who'])
print(clean_data['who'].unique())

# category 데이터인 class를 LabelEncoder()로 변환
clean_data['class'] = label_encoder.fit_transform(clean_data['class'])
print(clean_data['class'].unique())

# 변환된 자료형태 확인
print(clean_data.info())


# bool type 자료형 변환
clean_data['adult_male'] = clean_data['adult_male'].astype('int64')
clean_data['alone'] = clean_data['alone'].astype('int64')
print(clean_data.info())


# Dataframe indexing 권장사항 : loc 또는 iloc 사용
# target data 분리
target = pd.DataFrame(clean_data.iloc[:, 0], columns=['survived'])
train_data = clean_data.drop('survived', axis=1)
print(train_data)
print(target)

# value data 분리 ; 수치형으로 사용할 데이터들을 따로 분리
value_data = train_data[['age', 'fare']]
print(value_data.head())

# 수치형으로 사용할 value data에 대해 scaling 수행
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(value_data)
value_data = pd.DataFrame(scaled_data, columns=value_data.columns)
print(value_data.head())

# 데이터 확인 ; 표준 정규분포를 따르는 값으로 변환됨
print(value_data.describe())


# 기존 dataframe에서 분리된 value data 삭제
train_data.drop(['age', 'fare'], axis=1, inplace=True)
print(train_data.head())


# 범주형 성격의 숫자 데이터를 onehot encodeing 형태로 변환
# pclass에 대한 onehot encoding 작성
onehot_data = pd.get_dummies(train_data['pclass'])
print(onehot_data.head())

# 원본 데이터를 모두 onehot encoding으로 변환
onehot_data = pd.get_dummies(train_data, columns=train_data.columns)
print(onehot_data.head())

# 변환 자료형 확인
print(onehot_data.info())


# value data와 onehot data를 합쳐서 train data로 생성
training_data = pd.concat([value_data, onehot_data], axis=1)
print(training_data.info())
    # 2개의 수치 데이터와, 32개의 onehot data로 병합됨

# trainset / tsetset split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(training_data,
                                                    target,
                                                    test_size = 0.2,)
print('X_train shape :', X_train.shape)
print('X_test shape :', X_test.shape)
print('y_train shape :', y_train.shape)
print('y_test shape :', y_test.shape)


# modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(512, input_dim = 34,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())


# model compiling
model.compile(loss='mse', optimizer='adam', metrics=['binary_accuracy'])
fit_hist = model.fit(X_train, y_train,
                     batch_size = 50,
                     epochs= 50,
                     validation_split=0.2)

# 시각화 확인
import matplotlib.pyplot as plt
plt.plot(fit_hist.history['binary_accuracy'])
plt.plot(fit_hist.history['val_binary_accuracy'])
plt.show()

# 모델 평가
score = model.evaluate(X_test, y_test, verbose=0)
print('loss :', score[0])
print('accuracy :', score[1])