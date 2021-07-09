"""
    TF-IDF Vectorization

        ; 문장간 유사도 점수 Matrix 생성


    1. Setting

    2. TF-IDF Initialization

    3. TF-IDF Matrix Fitting

    4. TF-IDF Object Information Save

    5. TF-IDF Matrix Save

"""


import pandas as pd
# vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# Matrix Market Save / Load
from scipy.io import mmwrite, mmread
# mm : matrix market(TF-IDF score table)
import pickle


'1. Setting'
# data load
df_review_1stcs = pd.read_csv('./crawling/one_sentences_review_2018~2021.csv', index_col=0)
print(df_review_1stcs.info())
print(df_review_1stcs.head())


'2. TF-IDF Initialization'
# TF-IDF 객체 생성
tfidf = TfidfVectorizer(sublinear_tf=True)


'3. TF-IDF Matrix Fitting'
# TF-IDF matrix 생성
tfidf_matrix = tfidf.fit_transform(df_review_1stcs['reviews'])


'4. TF-IDF Object Information Save'
# 차후 데이터 추가 시 matrix update 를 위해 tfidf 객체정보를 저장해둬야 함
with open('./models/tfidf.pickle', 'wb') as f:
    pickle.dump(tfidf, f)


'5. TF-IDF Matrix Save'
# TF-IDF matrix 저장
mmwrite('./models/tfidf_movie_review.mtx', tfidf_matrix)


