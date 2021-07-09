import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.callbacks import EarlyStopping

# 데이터셋 불러오기
X_train, X_test, Y_train, Y_test = np.load('../datasets/binary_image_data.npy', allow_pickle=True)
# D:/workspace/stepBYstep/datasets/human_horse_dataset.npy
print('X_train shape : ', X_train.shape)
print('X_test shape : ', X_test.shape)
print('Y_train shape : ', Y_train.shape)
print('Y_test shape : ', Y_test.shape)

# modeling
model = Sequential()
model.add(Conv2D(64,  # output 개수
                 kernel_size=(3, 3),  # filter size
                 input_shape=(64, 64, 3),  # input shape ; Conv layer는 다차원 전달가능
                 padding='same',  # conv' 수행 후 출력이미지가 입력이미지 사이즈와 같도록 padding 조절
                 activation='relu'))  # activation 함수
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())

# compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# early stopping
early_stop = EarlyStopping(monitor='val_accuracy', patience=7)

# Training
fit_hist = model.fit(X_train, Y_train,
                     epochs=100,
                     validation_split=0.2,
                     batch_size=128,
                     callbacks=[early_stop], )
# model save
model.save('../model/catNdog_binary_classification.h5')

# model evaluation
score = model.evaluate(X_test, Y_test, verbose=1)
print('Loss :', score[0])
print('Accuaray :', score[1])

# visualiztion
import matplotlib.pyplot as plt

plt.plot(fit_hist.history['loss'], label='loss')
plt.plot(fit_hist.history['val_loss'], label='val_loss')
plt.legend()
plt.title('lOSS')
plt.show()

plt.plot(fit_hist.history['accuracy'], label='Acc')
plt.plot(fit_hist.history['val_accuracy'], label='val_Acc')
plt.legend()
plt.title('Accuracy')
plt.show()

'''
Loss : 0.4043552875518799
Accuaray : 0.847599983215332
'''
