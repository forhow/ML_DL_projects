"""
    Movie Recommendation Program

        - 입력된 keyword에 대한 유사도 높은 영화를 선정, 추천


    1. Setting

    2. Function Definition

    3. Entry of Keyword

    4. Creation Sentence for similarity Comparison

    5. Movie Recommendation

    - Test : load 한 movie review
"""

import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from gensim.models import Word2Vec
from scipy.io import mmread
import pickle


'1. Setting'
# Data Load
df_review_1stcs = pd.read_csv('./crawling/one_sentences_review_2018~2021.csv', index_col=0)
# print(df_review_1stcs.info())
# print(df_review_1stcs.head())

# TF-IDF matrix load / TF-IDF Load
tfidf_matrix = mmread('./models/tfidf_movie_review.mtx').tocsr()

# TF-IDF Information Load
with open('./models/tfidf.pickle', 'rb') as f:
    tfidf = pickle.load(f)


'2. Function Definition'
def getRecommendation(cosine_sim):
    simScore = list(enumerate(cosine_sim[-1]))
    simScore = sorted(simScore, key=lambda x: x[1], reverse=True)
    simScore = simScore[1:10]  # 0은 self
    movieidx = [i[0] for i in simScore]
    recMovieList = df_review_1stcs.iloc[movieidx, 0]  # (Row, Title) Indexing
    return recMovieList


'3. Entry of Keyword'
print('=' * 100, '\n추천받고자 하는 영화에 대한 키워드를 입력하세요!')
key_word = input('키워드 : ')
print('=' * 100)
embedding_model = Word2Vec.load('./models/word2VecModel_2018_2021.model')

print('입력한 키워드 {}와 관련된 영화 정보'.format(key_word))


'4. Creation Sentence for similarity Comparison'
# keyword 반복입력 - TF 값을 증가
sentence = [key_word] * 10

# keyword 외 관련도 높은 단어를 함께 문장으로 구성
sim_word = embedding_model.wv.most_similar(key_word, topn=10)
labels = []
for label, _ in sim_word:
    labels.append(label)
print('유사단어 : ', labels)

for i, word in enumerate(labels):
    sentence += [word] * (9-i)

sentence = ' '.join(sentence)
# print(sentence)


'5. Movie Recommendation'
sentence_vec = tfidf.transform([sentence])
cosine_sim = linear_kernel(sentence_vec, tfidf_matrix)
recommendation = getRecommendation(cosine_sim)
print(recommendation)


'''
    비지도 학습의 특성상 모델에 대한 객관적인 평가지표가 없음(Acc / Loss) 
    모델이나 서비스 적용 이전과 이후의 매출 차이, 서비스 만족도로 평가를 가늠해 볼 수 있음
'''


'''
    TEST
'''
# # method 1. 영화의 index 탐색
# movie_idx = df_review_1stcs[df_review_1stcs['titles']=='기생충 (PARASITE)'].index[0]
#
# # method 2. index를 직접 지정해서 사용 가능
# # movie_idx = 300
# # 영화 제목 확인
# print(df_review_1stcs.iloc[movie_idx, 0])
#
# # linear Kernel :
# # - 특정 영화의 tfidf 와 전체 영화의 tfidf 를 비교해서 cosine 유사도 값을 구함
# cosine_sim = linear_kernel(tfidf_matrix[movie_idx], tfidf_matrix)
#
# # 추천 영화 목록 추출함수 실행
# recommendation = getRcommendation(cosine_sim)
# # 결과 출력
# print(recommendation)