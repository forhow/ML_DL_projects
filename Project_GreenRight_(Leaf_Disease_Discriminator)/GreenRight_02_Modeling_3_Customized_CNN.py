"""
    Project : GreenRight
        - 식물(잎) 이미지를 통한 식물 분류 및 병충해 판별기 구축

    Modeling, Training, Evaluation : Customized CNN

    1. Setting

    2. Modeling
     a. Model_01 Customized CNN
     b. Model_02 Customized CNN (train : 0.9338  test : 0.8923)
     c. Model_03 Customized CNN (train : 0.9338  test : 0.8923)
     d. Model_04 Customized CNN (train : 0.9848  test : 0.8852)
     e. Model_05 Customized CNN (train : 0.9895  test : 0.9310)

    3. Model Training

    4. Model Evaluation

    5. Training Result Visualization
     a. Evaluation Result
     b. Accuracy & Validation Accuracy
     c. Loss & Validation Loss

"""

from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dropout, Activation, Dense
from keras.layers import Flatten, MaxPooling2D
from keras.layers import Conv2D
from keras.models import load_model
from keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
import os
import numpy as np

'1. Setting'
# Model Save path
model_dir = './model'
if not os.path.exists("./model/"):
    os.mkdir('./model/')

# Dataset Load
X_train, X_test, Y_train, Y_test = np.load('./img_data.npy', allow_pickle=True)

# Category Setting
categories = list(str(i) for i in range(20))

# Training Hyper parameter setting
EPOCHS = 100 
BS =32
INIT_LR = 1e-3
n_classes = len(categories)


'2. Modeling'

'2.a Model_01 Customized CNN'
model = Sequential()
model.add(Conv2D(32, (3,3), padding="same", input_shape=X_train.shape[1:], activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


'2.b Model_02 Customized CNN'
#  - train: 0.9338 test:0.8923
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(64,64, 3)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128))
model.add(Dropout(0.5))
model.add(Activation("relu"))

model.add(Dense(64))
model.add(Dropout(0.5))
model.add(Activation("relu"))

model.add(Dense(20, activation="softmax"))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


'2.c Model_03 Customized CNN'
# - train: 0.9848 test: 0.8852
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=X_train.shape[1:], padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(n_classes))
model.add(Activation('softmax'))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
print(model.summary())


'2.d Model_04 Customized CNN'
#  - train: 0.9950 test: 0.9010
model = Sequential()
model.add(Conv2D(32, (3,3), padding="same", input_shape=X_train.shape[1:], activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
    

'2.e Model_05 Customized CNN'
#  - train 0.9895 test:9310
model = Sequential()
model.add(Conv2D(32, (3,3), padding="same", input_shape=X_train.shape[1:], activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3,3), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3,3), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(n_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


'3. Model Training'
# Training Option
model_path = model_dir + '/multi_img_classification.model'
checkpoint = ModelCheckpoint(filepath=model_path , monitor='val_loss', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=6)

# Training
history = model.fit(X_train,
                    Y_train,
                    batch_size=BS,
                    validation_data=(X_test, Y_test),
                    steps_per_epoch=len(X_train) // BS,
                    epochs=EPOCHS, verbose=1,
                    callbacks=[checkpoint,early_stopping])


'4. Model Evaluation'
print("Train 정확도 : %.4f" % (model.evaluate(X_train, Y_train)[1]))
print("Test 정확도 : %.4f" % (model.evaluate(X_test, Y_test)[1]))


'5. Training Result Visualization'
import tensorflow as tf
import matplotlib.pyplot as plt

'5.a. Evaluation Result'
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

'5.b. Accuracy & Validation Accuracy'
plt.rc('font',family='Malgun Gothic')
plt.plot(epochs, acc, 'b', label='Training accurarcy')
plt.plot(epochs, val_acc, 'r', label='Testing accurarcy')
plt.title('학습과 훈련 정확도')
plt.legend()
plt.figure()

'5.c. Loss & Validation Loss'
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Testing loss')
plt.title('학습과 훈련 손실')
plt.legend()
plt.show()







