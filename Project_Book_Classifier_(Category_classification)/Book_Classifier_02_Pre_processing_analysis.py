"""
    Visualization - Words Analysis

    1. Setting : Data load
    2. Words count in Each Sentence
    3. Visualization
     a. 문장 당 단어수의 분포 확인
     b. 단어수 동일 문장의 분포 확인

"""

import pandas as pd
import matplotlib.pyplot as plt

'1. Setting : Data load'
# Pre-processing step. 7.b에서 불용어 처리까지 끝난 데이터로 분석
data = pd.read_csv('./pre_process_data/stopword_removed_X.csv', index_col=0)
print(data.info())
print(type(data))
data = data.Introduction
# print(data[0].split())
# print(len(data[0].split()))
# print(len(data))


'2. Words count in Each Sentence'
Y = []
for i in range(len(data)):
    snt = str(data[i]).split()
    Y.append(len(snt))
print(Y[:20])


'3. Visualization'
# X pos = 0
X = [0 for i in range(len(Y))]


'3.a. 문장 당 단어수의 분포 확인'
plt.title('Word Count')
plt.scatter(Y, X, s=2**2)
plt.ylim([-0.1, 0.1])
plt.yticks([-1,0,1])
plt.xlabel('Count of Words')
plt.show()


'3.b. 단어수 동일 문장의 분포 확인'
plt.title('Sentence Count')
plt.hist(Y, bins= 1500, facecolor='g', alpha=0.75)
plt.xlim([0, 1500])
plt.grid(True)
plt.xlabel('Count of Words')
plt.ylabel('Count of Sentences')
plt.xticks([i for i in range(0, 1500, 100)])
plt.show()

