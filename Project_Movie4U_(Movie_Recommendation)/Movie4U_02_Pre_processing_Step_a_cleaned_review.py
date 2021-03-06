'''
    Pre-processing - Step. a
    
        ; 형태소 분석, 불용어 제거 등 리뷰 문장에 대한 가공
    
    1. Setting / Initialization
    
    2. Entry of Objective File Name(year)
    
    3. Raw data(Crawling) File Load (.csv)
    
    4. Preprocessing
     a. alphabet, number, symbol Removal
     b. Tokenization
     c. Noun, Verb, Adjective Extraction
     d. Stopwords Removal
     e. Concatenation for split words (cleaned_sentences)
     
    5. Column and Dataframe Creation
    
    6. File Save
    
'''


import pandas as pd
import numpy as np
import re
from konlpy.tag import Okt
# 다른 형태소 분리기 사용시 참고
# from konlpy.tag import Komoran
# from konlpy.tag import Kkma
# from konlpy.tag import Hannanum
# from konlpy.tag import Mecab


'1. Setting / Initialization'
# 형태소 분리 객체 생성
okt = Okt()

# progress counter 용
count = 0

# 불용어 제거 후 단어 병합 저장용 리스트
cleaned_sentences = []

# 불용어 리스트(set)
stopwords = pd.read_csv('./crawling/stopwords.csv', index_col=0)
movie_stopwords = ['관객', '작품', '받다', '촬영', '크다',
                   '리뷰', '개봉', '스크린', '출연', '개봉',
                   '극장', '평가', '출연', '평점', '영화',
                   '작품', '배우', '주인공', '주연', '조연',
                   '감독', '연출', '극본', '시나리오', '관객',
                   '개인', '관람', '제작', '기록', '제작비',
                   '투입', '수익', '예고편', ]
stopwords_list = list(stopwords.stopword) + movie_stopwords
# stopwords_list = set(stopwords_list)
# print(len(stopwords_list))
# print(type(stopwords), len(stopwords), stopwords)


'2. Entry of Objective file(.csv)'
print('전처리 수행할 파일 이름과 년도 확인')
print('ex) reviews_2021.csv -> 2021')
year = int(input('대상 파일의 연도 입력 : ' ))


'3. Raw data(Crawling) File Load (.csv)'
df = pd.read_csv('./crawling/reviews_{}.csv'.format(year), index_col=0 )
print(df.info())
print(df.head())


'4. Preprocessing - step.a'
for sentence in df.reviews:
    # progress counter
    count += 1
    if count % 10 == 0:
        print('.', end='')
    if count % 300 == 0:
        print(' {}'.format(count))

    '4.a. alphabet, number, symbol Removal'
    sentence = re.sub('[^가-힣 ]', '', sentence)

    '4.b. Tokenization'
    token = okt.pos(sentence, stem=True)
    df_token = pd.DataFrame(token, columns=['word', 'class'])

    '4.c. Noun, Verb, Adjective Extraction'
    df_cleaned_token = df_token[(df_token['class'] == 'Noun') |
                            (df_token['class'] == 'Verb') |
                            (df_token['class'] == 'Adjective')]
    words = []

    '4.d. Stopwords Removal'
    for word in df_cleaned_token['word']:
        if len(word) > 1:
            if word not in stopwords_list:
                words.append(word)

    '4.e. Concatenation for split words (cleaned_sentences)'
    cleaned_sentence = ' '.join(words)
    cleaned_sentences.append(cleaned_sentence)


'5. Column and Dataframe Creation'
df['cleaned_sentences'] = cleaned_sentences
# print(df.head())
df = df[['titles', 'cleaned_sentences']]
print(df.info())


'6. File Save'
df.to_csv('./crawling/cleaned_review_{}.csv'.format(year))
