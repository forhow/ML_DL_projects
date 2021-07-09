"""
    Embedding Modeling : Word2Vec

    - NLP embedding Model ; Words Vectorization


    1. Setting
     a. data load
     b. Review part Separation
     c. Variable Initialization

    2. Tokenization

    3. Embedding

    4. Model Save

"""

import pandas as pd
from gensim.models import Word2Vec


'1. Setting'

'1.a. data load'
review_word = pd.read_csv('./crawling/cleaned_review_2018_2021.csv', index_col=0)
print(review_word.info())
print(review_word.head())

'1.b. Review part Separation'
cleaned_token_review = list(review_word['cleaned_reviews'])
print(len(cleaned_token_review))

'1.c. Variable Initialization'
cleaned_tokens = []
count = 0


'2. Tokenization'
for sentence in cleaned_token_review:
    token = sentence.split()
    cleaned_tokens.append(token)
print(len(cleaned_tokens))


'3. Embedding'
# 벡터공간의 근처에 배치된 단어는 유사하다는 전체로 학습
embedding_model = Word2Vec(cleaned_tokens,
                           vector_size=100,  # output dim : 차원축소
                           window=4,  # 커널 사이즈 (문장의 길이)
                           min_count=20,  # 출현빈도가 20이상인 단어만 사용
                           workers=4,  # 사용할 cpu core 개수
                           epochs=100,  # 학습 진행 횟수
                           sg=1)  # algorithm 선택

'4. Model Save'
embedding_model.save('./models/word2VecModel_2018_2021.model')


# tokenization 된 상태 확인  -> deprecated
# print(embedding_model.wv.vocab.keys())
# print(len(embedding_model.wv.vocab.keys()))
