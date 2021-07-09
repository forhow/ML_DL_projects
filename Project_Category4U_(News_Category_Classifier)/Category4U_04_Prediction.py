
"""
    Prediction

    * 저장된 모델 로드 및 새로운 데이터를 통해 category 분류 테스트

    1. Setting
     - Data Load : Model, Token, Encoding, Stopwords, Object Data
    2. Object Data Setting
     - 중복제거
    3. Object Data - Target Data Pre-processing
    - a) Label encoding
    - b) Onehot-Encoding
    4. Object Data - Train Data Pre-processing
    - a) 형태소 분리
    - b) 불용어 제거 및 재조합
    - c) 단어 Tokenizing (Tokenizing file 저장)
    - d) 형태소 개수 및 최장 문장길이 확인(학습모델 설정정보)
    - e) Padding
    5. Prediction
    6. Accuracy Check

"""

from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np
from tensorflow.keras.models import *
import pickle
from konlpy.tag import Okt

'1. Setting'
# pandas option 설정
pd.set_option('display.unicode.east_asian_width', True)

# Model Load
model = load_model('../model/news_classification_0.75.h5')

# Token mapping file Load
with open('../datasets/CAT4U_category_token.pickle', 'rb') as f:
  token = pickle.load(f)

# Label encoding mapping file Load
with open('../datasets/CAT4U_category_encoder.pickle', 'rb') as f:
  encoder = pickle.load(f)
category = encoder.classes_
print(category)

# Stopwords Load
stopwords = pd.read_csv('../datas/stopwords.csv')

# Object Data Load
df_news_today = pd.read_csv('../datas/news_titles_today.csv', index_col=0)
print(df_news_today.head())
print(df_news_today.info())


'2. Object Data Setting'
# title의 중복확인
col_dup = df_news_today['title'].duplicated()
print(col_dup)
sum_dup = df_news_today.title.duplicated().sum()
print(sum_dup)

# title의 중복 제거
df_news_today = df_news_today.drop_duplicates(subset=['title'])
sum_sup = df_news_today.title.duplicated().sum()
print(sum_dup)
df = df_news_today

# 중복 제거 후 index reset
df.reset_index(drop=True, inplace=True)

# Data Split : Predict Data(X) / Target(Actual) Data(Y)
X = df['title']
Y = df['category']
# print(X[1748])
# print(X.shape)
# print(len(X))


'3. Object Data - Target Pre-processing'
label_Y = encoder.transform(Y)
print(label_Y)
print(len(label_Y))

onehot_Y = to_categorical(label_Y)
print(onehot_Y)


'4. Object Data - Train Data Pre-processing'
okt = Okt()
for i in range(len(X)):
  X[i] = okt.morphs(X[i])

print(X)

for i in range(len(X)) :
  result = []
  for j in range(len(X[i])):
    if len(X[i][j]) > 1:
      if X[i][j] not in list(stopwords['stopword']):
        result.append(X[i][j])
  X[i] = ' '.join(result)

print(X)

# 단어 tokenization : Token화 정보를 갖고있는 token을 통해 새로운 문장을 각 단어별로 숫자를 배정
tokened_X = token.texts_to_sequences(X)
print(tokened_X[0])

# padding 적용
from tensorflow.keras.preprocessing.sequence import pad_sequences
X_pad = pad_sequences(tokened_X, 27)
print(X_pad[:10])


'5. Prediction'
# prediction
predict = model.predict(X_pad)
print(predict[0])

pred_category = []
for pred in predict:
  pred_category.append(category[np.argmax(pred)])
print(pred_category)

df['predict'] = pred_category
print(df.info())


'6. Accuracy Check'
df['OX'] = 0
print(df.info())

for i in range(len(df)):
  if df['category'][i] == df['predict'][i]:
    df['OX'][i] = 'O'
  else:
    df['OX'][i] = 'X'

print(df.OX.value_counts()/len(df))
print(df.iloc[-30:, 0:3])

