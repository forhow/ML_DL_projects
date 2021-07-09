"""
    Words Vectorization 후 단어들의 배치 관계 시각화

        - 입력한 키워드와 관련된 단어 10개 확인
        - Vectorized space 내 키워드와 관련단어의 배치 시각화 확인


    1. Setting

    2. Entry of Keyword

    3. Similar Words Extraction from Model

    4. Create Dataframe with Similar Words and its Vector

    5. Dimension Reduction

    6. Visualization

"""



import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from sklearn.manifold import TSNE
from matplotlib import font_manager, rc
import matplotlib as mpl


'1. Setting'
# Font 지정
font_path = './malgun.ttf'
font_name = font_manager.FontProperties(fname=font_path).get_name()
mpl.rcParams['axes.unicode_minus']=False
rc('font', family=font_name)

# Model Load
embedding_model = Word2Vec.load('./models/word2VecModel_2018_2021.model')


'2. Entry of Keyword'
# key_word = input('키워드를 입력하세요 : ')
key_word = '기다림'


'3. Similar Words Extraction from Model'
# Keyword와 비슷한 Word Vector 10개 Load
sim_word = embedding_model.wv.most_similar(key_word, topn=10)
# (단어, Vector) List로 반환
print(sim_word)

'4. Create Dataframe with Similar Words and its Vector'
# 벡터공간 차원 및 단어 배치 확인
vectors = []
labels = []

for label, _ in sim_word:
    labels.append(label)
    vectors.append(embedding_model.wv[label])

df_vectors = pd.DataFrame(vectors)
print(df_vectors.iloc[0,12])


'5. Dimension Reduction'
# t-SNE 모델 활용, 2차원으로 축소
tsne_model = TSNE(perplexity=40, n_components=2,
                  init='pca', n_iter=2500, random_state=23)

new_values = tsne_model.fit_transform(df_vectors)

df_xy = pd.DataFrame({'word':labels,
                      'x':new_values[:,0],
                      'y':new_values[:,1]})
print(df_xy.head())
print(df_xy.shape)

# 새로운 row에 keyword와 좌표(0,0) 추가
df_xy.loc[df_xy.shape[0]] = (key_word, 0, 0)


'6. Visualization'
# Scatter with Base point
plt.figure(figsize=(8,8))
# Center(origin) point
plt.scatter(0, 0, s=1500, marker='*')

for i in range(len(df_xy.x)):
    a = df_xy.loc[[i,10], :]
    # print('X :\n', a.x)  # vector 좌표 확인
    # print('Y :\n', a.y)
    plt.plot(a.x, a.y, '-D', linewidth=2)
    # 주석 작성 - 연결선과 내용
    plt.annotate(df_xy.word[i],
                 xytext=(5,5),  # 주석내용의 우측하부 시작위치
                 xy=(df_xy.x[i], df_xy.y[i]),  # 점의 위치
                 textcoords='offset points',
                 ha='right',  # Horizontal align
                 va='bottom')  # Vertical align    글자크기 다를 때 정렬 기준위치
    # base line : 알파벳에서 기준선 설정 (j, y 등)
plt.show()







