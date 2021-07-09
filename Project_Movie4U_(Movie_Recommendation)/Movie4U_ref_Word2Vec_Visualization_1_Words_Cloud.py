"""
    Word Cloud Visualization

        - 영화 리뷰에 등장하는 단어를 이용한 Word Cloud 생성
        - 전처리 상태 및 영화별 리뷰 내용의 성향 확인

    1. Setting

    2. Entry for Movie title

    3. Information Print

    4. Words, Frequency Dictionary Creation

    5. Words Cloud Creation
"""

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import collections
import matplotlib as mpl
from matplotlib import font_manager, rc


'1. Setting'
# Font Information for MPL
fontpath = 'malgun.ttf'
font_name = font_manager.FontProperties(fname=fontpath).get_name()
rc('font', family=font_name)
mpl.font_manager._rebuild()

# Data load
df = pd.read_csv('./crawling/one_sentences_review_2018~2021.csv', index_col=0)

# Stopwords
stopwords = ['관객', '작품', '받다', '촬영', '크다', '메다', '리뷰', '나오다', '그렇다',
             '개봉', '스크린', '출연', '극장', '평가', '출연', '평점', '보다']

# Missing value processing
df.dropna(inplace=True)
# print(df.info())
print('Movie List')

for idx, title in enumerate(df.titles.unique()):
    print(title, end=', ')
    if idx % 10 == 0:
        print()


'2. Entry for Movie title'
print('=================================================================================== \n\n')
title = input('Enter Movi title for Word Cloud (from above list) : ')


'3. Information Print'
# 해당하는 조건의 index를 반환
movie_index = df[df['titles'] == title].index[0]
print('Movie Number : ', movie_index)

# 리뷰를 구성하는 단어를 확인
print('Words in Review', df.reviews[movie_index])
words = df.reviews[movie_index].split()
# print(words)


'4. Words, Frequency Dictionary Creation'
word_dict = collections.Counter(words)  # Counter Object Type
# print(word_dict)
word_dict = dict(word_dict)  # Dictionary Type
# print(word_dict)


'5. Words Cloud Creation'
'''
method_1 : generate
- 구성된 단어를 기준으로 그림 ; 단어 리스트로 전달
- stopword 지정함으로써 추가적인 전처리 과정 수행 가능
'''
word_cloud_img = WordCloud(background_color='white',  # Backgraoud Color
                           max_words=200,  # Number of words to use
                           stopwords=stopwords,
                           font_path=fontpath
                           ).generate(df.reviews[movie_index])

'''
method_2 : generate_from_frequencies
- 단어 출현 빈도에 따라 그림 ; {단어:count} 형태의 dictionary로 값 전달
- stopword 추가 제거 불가
'''
# word_cloud_img = WordCloud(background_color='white',
#                            max_words=200,
#                            font_path=fontpath
#                            ).generate_from_frequencies(word_dict)


plt.figure(figsize=(8,8))
plt.imshow(word_cloud_img, interpolation='bilinear')
plt.title(df.titles[movie_index])
plt.axis('off')
plt.show()
