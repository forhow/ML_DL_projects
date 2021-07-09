'''
과제

    Titanic survivor prediction

- titanic_survivor_prediction 파일 참조
'''


###################################
import seaborn as sns
import pandas as pd
import numpy as np

# dataframe 전제 데이터 보기
pd.set_option('display.max_columns', 15)
pd.set_option('display.max_row', 500)


# source data 확인
raw_data = sns.load_dataset('titanic')
# print(raw_data.head(10))
# print(raw_data.info())

'''
Pre-processing
1. column 삭제 
- class : pclass가 대체
- deck : 결측값 다수로 삭제
- alive : survived column으로 대체
- embark town : embarked 로 대체
- who : sex와 age로 대체
- adult_male : sex와 age로 대체

2. data 변환
- sex : object - int (male 0/female1)
- alone : bool - int (false 0 / True 1)
- embark : s - 2
           c - 1
           q - 0

3. 결측값 처리
- age : 승객의 평균 나이로 치환 

'''

'''
    Pre-processing
'''
raw_data = raw_data.drop(['class', 'deck', 'alive', 'embark_town', 'who', 'adult_male'], axis=1)
raw_data = raw_data.replace('male', 0)
raw_data = raw_data.replace('female', 1)
raw_data = raw_data.replace(False, 0)
raw_data = raw_data.replace(True, 1)
raw_data['age'].fillna(raw_data.age.mean().round(), inplace=True)
raw_data = raw_data.replace('S', 2)
raw_data = raw_data.replace('C', 1)
raw_data = raw_data.replace('Q', 0)
raw_data = raw_data.dropna()
# print(raw_data.info())
# print(raw_data.head(10))


'''
    Data Separation
'''
train_data = raw_data.iloc[:, 1:]
train_data_cols = train_data.columns
# print(train_data_cols)
target = raw_data['survived']
# print(train_data.info())
# print(target)


'''
    Scaling
'''
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(train_data)
scaled_data = pd.DataFrame(scaled_data, columns=train_data_cols)
# print(scaled_data)

from sklearn.preprocessing import MinMaxScaler

scaler_mm = MinMaxScaler()
mm_scaled_data = scaler_mm.fit_transform(train_data)
mm_scaled_data = pd.DataFrame(scaled_data, columns=train_data_cols)

'''
    Test data Split
'''
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(mm_scaled_data,
                                                    target,
                                                    test_size=0.3)
print('X_train shape :', X_train.shape)
print('X_test shape :', X_test.shape)
print('y_train shape :', y_train.shape)
print('y_test shape :', y_test.shape)

'''
    Modeling
'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(512, input_dim=(8), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())

'''
    Compile
'''

model.compile(loss='mse', optimizer='adam', metrics=['binary_accuracy'])

fit_hist = model.fit(X_train,
                     y_train,
                     epochs=100,
                     batch_size=32,
                     validation_split=0.2)

'''
    Evaluation
'''
score = model.evaluate(X_test, y_test, verbose=1)
print('Keras DNN model loss :', score[0])
print('Keras DNN model accuracy(F1 score) :', score[1])