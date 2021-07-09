"""
    Pre-processing  - Yes24 웹 베스트샐러 Crawling data pre-processing

    1. Setting

    2. Needless Columns Processing
    3. Space Processing
    4. Missing Value Processing
    5. Data Split : Train Data / Target Data (for pre-processing)

    6. Target Data Processing
     a. Label Encoding and Data Save
     b. OneHot_Encoding (for Training)

    7. Train Data Processing
     a. Morphs separation
     b. stopwords removal
     c. Tokenization and Data save
     d. Data info Check (for Modeling)
     e. Padding

    8. Data Split : Train / test Dataset (for model training)

    9. Train / Test Dataset Save
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

'1. Setting'
# pandas display set
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.max_columns', 6)

# Raw data Load
file_path = './datasets/book_raw_data.csv'
data = pd.read_csv(file_path)
print('initial', data.info())


'2. Needless Columns Processing'
# Na Drop
data.drop(['Unnamed: 0', 'Small_category', 'cnt'], axis=1, inplace=True)
data = data.dropna()
print('after dropna',data.info())


'3. Space Processing'
for i in range(len(data)):
  for j in range(5, 1, -1):
      data.iloc[i,2] = data.iloc[i,2].replace(' '*j, ' ')
print('space remove',data.info())


'4. Missing Value Processing'
# # 결측처리 및 column 지정 n
data.dropna(subset=['Medium_category'], axis=0, inplace=True)
data.drop_duplicates(subset=['Introduction'], inplace=True)
data.reset_index(drop=True, inplace=True)
print(data.info())


'5. Data Split : Train Data / Target Data '
X = data['Introduction']
Y = data['Medium_category']


'6. Target Data Processing'
'6.a. Label Encoding and Data Save'
encoder = LabelEncoder()
labeled_Y = encoder.fit_transform(Y)
label = encoder.classes_
print(label)

# Encode mapping data save
with open('../datasets/category_encoder.pickle', 'wb') as f:
  pickle.dump(encoder, f)
print(labeled_Y)

'6.b. OneHot_Encoding (for Training)'
# label을 onehotencoding 으로 변환
onehot_Y = to_categorical(labeled_Y)
print(onehot_Y)
# print(X[309])
# print(type(X))


'7. Train Data Processing'

'7.a. Morphs separation'
okt = Okt()
print('형태소 분리')

for i in range(len(X)):
  X[i] = okt.morphs(X[i])

  #progress display
  if (i % 250 == 0) and (i>1):
    print('.', end='')
  if i % 5000 == 0:
    print('{} / {}'.format(i, len(X)))
print(X)

# Check point 1: Data Save
X.to_csv('../datasets/morphs_sep_X.csv', encoding='utf-8-sig')


'7.b. stopwords removal'
print('stopwords 제거')
stopwords = pd.read_csv('./datasets/stopwords.csv')

for i in range(len(X)) :
  result = []
  for j in range(len(X[i])):
    if len(X[i][j]) > 1:
      if X[i][j] not in list(stopwords['stopword']):
        result.append(X[i][j])
  X[i] = ' '.join(result)

  #progress display
  if (i % 250 == 0) and (i>1):
    print('.', end='')
  if i % 5000 == 0:
    print('{} / {}'.format(i, len(X)))
print(X)
# Check point 2: Data Save
X.to_csv('./datasets/stopword_removed_X.csv')


'7.c. Tokenization and Data save'
# 단어 tokenization : 각 단어별로 숫자를 배정
token = Tokenizer()
token.fit_on_texts(X) # 형태소에 어떤 숫자를 배정할 것인지 배정
tokened_X = token.texts_to_sequences(X) # 토큰에 저장된 정보를 바탕으로 문장을 변환
print(tokened_X[0])

# Check point 3 : Data Save
tokened_X.to_csv('./datasets/tokened_X.csv')

# Tokeniation Mapping data save
with open('./datasets/news_token.pickle', 'wb') as f:
  pickle.dump(token, f)


'7.d. Data info Check (for Modeling)'
# Word_size Check : 형태소 개수 확인
wordsize = len(token.word_index) +1
print('word index : ', token.word_index)
print('wordsize is : ', wordsize)

# Max_Length Check : 가장 긴 문자의 길이 확인
max = 0
for i in range(len(tokened_X)):
  if max < len(tokened_X[i]):
    max = len(tokened_X[i])
print('max is : ', max)


'7.e. Padding'
# 각 문장의 길이를 Max_Length로 맞추고 길이 이하의 문장의 앞은 0으로 Padding
X_pad = pad_sequences(tokened_X, max)
# Check point 4 : Final Data Save
X_pad.to_csv('./datasets/padded_X.csv')
print(X_pad[:10])


'8. Split : Train / test Dataset'
X_train, X_test, Y_train, Y_test = train_test_split(X_pad, onehot_Y, test_size=0.2)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


'9. Train / Test Dataset Save'
xy = X_train, X_test, Y_train, Y_test
np.save('./datasets/book_data_max_{}_size_{}'.format(max, wordsize), xy)