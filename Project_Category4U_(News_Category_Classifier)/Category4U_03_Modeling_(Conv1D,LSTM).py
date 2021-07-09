"""
    Modeling

    1. Dataset Load
    2. Modeling (Embedding, Conv1D, MaxPooling, LSTM, Dense)
    3. Compiling
    4. Training
    5. Evaluation
    6. Model Save
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

'1. Dataset Load'
# data upload 및 load
X_train, X_test, Y_train, Y_test = np.load('../datasets/news_data_max_27_size_24151.npy', allow_pickle=True)
# X_train, X_test, Y_train, Y_test = np.load('/content/datasets/news_data_max_16_size_984.npy', allow_pickle=True)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


'2. Modeling (Embedding, Conv1D, MaxPooling, LSTM, Dense)'

model = Sequential()
# vectorizing : Embedding
model.add(Embedding(24151, 300, input_length=27))

# 단어간의 관계를 해석하기 위한 도구로 conv1D 사용
model.add(Conv1D(32, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPool1D(pool_size=1))
model.add(LSTM(128, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64, activation='tanh', return_sequences=False))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(6, activation='softmax'))

print(model.summary())

'3. Compiling'
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

'4. Training'
fit_hist = model.fit(X_train, Y_train, 
                     epochs=8, 
                     batch_size=100, 
                     validation_data=(X_test, Y_test))

# visualization
plt.plot(fit_hist.history['loss'])
plt.show()

'5. Evaluation'
score = model.evaluate(X_test, Y_test, verbose=1)
print(score[1])

'6. Model Save'
model.save('/content/models/news_classification_{}.h5'.format(score[1]))