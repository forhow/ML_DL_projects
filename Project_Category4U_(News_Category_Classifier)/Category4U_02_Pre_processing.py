"""
    Crawling Data Pre-processing

    1. csv 파일 로드
    2. 중복데이터 제거
    3. Target Data 처리
    - a) Target Column 선정 및 label encoding (Encoding 정보 file 저장)
    - b) Label encoding을 Onehot-Encoding으로 변환
    4. Train Data 처리
    - a) 형태소 분리
    - b) 불용어 제거 및 재조합
    - c) 단어 Tokenizing (Tokenizing file 저장)
    - d) 형태소 개수 및 최장 문장길이 확인(학습모델 설정정보)
    - e) Padding
    5. Train / Test Dataset 저장

    Project 파일관리
    - 총 6개 파일
    from 01_crawling - 1. crawling data (.csv)
    from 02_preprocessing - 2. Label Encoding (.pickle)
                          - 3. Tokeninzing (.pickle)
                          - 4. Dataset (.npy)
    from 03_modeling - 5. model(.h5)

    from  6. stopword.csv (stopword 저장)

"""

import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
from konlpy.tag import Okt
import pickle

'1. csv 파일 로드'
pd.set_option('display.unicode.east_asian_width', True)

df = pd.read_csv('../datas/news_titles_data.csv', index_col=0)
print(df)
print(df.info())


'2. 중복데이터 제거'
# title의 중복확인
col_dup = df['title'].duplicated()
print(col_dup)
sum_dup = df.title.duplicated().sum()
print(sum_dup)

# title의 중복 제거
df = df.drop_duplicates(subset=['title'])
sum_sup = df.title.duplicated().sum()
print(sum_dup)

# 제거된 데이터의 인덱스 재지정
df.reset_index(drop=True, inplace=True)

# Train data, Target Data 지정
X = df['title']
Y = df['category']


'3. Target Data 처리'

'3.a) Target Column 선정 및 label encoding (Encoding 정보 file 저장)'
# Y(target)을- label로 변환
encoder = LabelEncoder()
labeled_Y = encoder.fit_transform(Y)
label = encoder.classes_
print(label)

# encoding mapping 정보를 저장
with open('../datasets/CAT4U_category_encoder.pickle', 'wb') as f:
  pickle.dump(encoder, f)

print(labeled_Y)


'b) Label encoding을 Onehot-Encoding으로 변환'
# label을 onehotencoding 으로 변환
onehot_Y = to_categorical(labeled_Y)
print(onehot_Y)


'4. Train Data 처리'
'4.a) 형태소 분리'
okt = Okt()
# print(type(X))
# okt_X = okt.morphs(X[0])
# print(X[0])
# print(okt_X)

# 하나의 문장을 형태소로 분할
for i in range(len(X)):
  X[i] = okt.morphs(X[i])
print(X)

'4.b) 불용어 제거 및 재조합'
# 접속사 조사 감탄사 등 문장분류에 도움이 되지 않는 단어들을 제거
# stopwords.csv 파일 load
stopwords = pd.read_csv('../datasets/stopwords.csv')
# print(stopwords)

# 불용어 제거 후 형태소로 이루어진 문장으로 재조합
for i in range(len(X)) :
  result = []
  for j in range(len(X[i])):
    if len(X[i][j]) > 1:
      if X[i][j] not in list(stopwords['stopword']):
        result.append(X[i][j])
  X[i] = ' '.join(result)
print(X)

'4.c) 단어 Tokenizing : 각 단어별로 숫자를 배정 (Tokenization mapping file 저장)'
token = Tokenizer()
token.fit_on_texts(X) # 형태소에 어떤 숫자를 배정할 것인지 배정
tokened_X = token.texts_to_sequences(X) # 토큰에 저장된 정보를 바탕으로 문장을 변환
print(tokened_X[0])
print(type(tokened_X))

# tokenizing mapping 정보를 저장
with open('../datasets/CAT4U_category_token.pickle', 'wb') as f:
  pickle.dump(token, f)
# 이후 다른 문장을 tokenization 수행시 이전에 token화 되지 않은 단어 데이터는 제외시킴

'4.d) 형태소 개수 및 최장 문장길이 확인(학습모델 설정정보)'
# 형태소 개수 확인
wordsize = len(token.word_index) +1   # +1 의미 : index 0를 padding 으로 추가 예정으로 1개의 단어자리가 필요
print(token.word_index)

print(wordsize)

# 모델에 제공하는 데이터 길이를 맞춰주기 위한 작업 수행
# 모든 문장 중 가장 긴 문장 기준으로 맞춤
# 문장의 앞자리에 0을 padding함으로써 문장 앞부분의 영향을 미미하도록 유지

# 가장 긴 문자의 길이 확인
max = 0
for i in range(len(tokened_X)):
  if max < len(tokened_X[i]):
    max = len(tokened_X[i])
print(max)

'4.e) Padding'
# padding ; 모든 문장 길이를 가장 긴 문장 길이와 맞추고 비어있는 앞쪽 자리를 0으로 채움
X_pad = pad_sequences(tokened_X, max)
print(X_pad[:10])


'5. Train / Test Dataset Split'
X_train, X_test, Y_train, Y_test = train_test_split(X_pad, onehot_Y, test_size=0.1)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


'Dataset save'
xy = X_train, X_test, Y_train, Y_test
np.save('../datasets/CAT4U__dataset_max_{}_size_{}'.format(max, wordsize), xy)

