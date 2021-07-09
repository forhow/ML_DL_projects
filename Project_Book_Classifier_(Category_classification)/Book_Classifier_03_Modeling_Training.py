"""
    Modeling and Training

    1. Dataset Load
    2. Modeling
    3. Compiling
    4. Training
    5. Evaluation
    6. Visualization
    7. Graph and Model Save

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras import regularizers


'1. Dataset Load'
X_train, X_test, Y_train, Y_test = np.load('../datasets/book_data_cat6_max_196_wordsize_218318.npy', allow_pickle=True)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


'2. Modeling'
model = Sequential()
model.add(Embedding(218318, 100, input_length=196)) 
model.add(Conv1D(32, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPool1D(pool_size=1))
model.add(Dropout(0.7))
model.add(LSTM(32, activation='tanh', return_sequences=True))  
model.add(Dropout(0.7))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(6, activation='softmax'))
print(model.summary())

# Option : Eearly Stoping
# early_stopping = tf.keras.callbacks.EarlyStopping(moniter='val_accuracy', patience=5)


'3. Compiling'
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


'4. Training'
fit_hist = model.fit(X_train, Y_train, batch_size=32, epochs=10, validation_data=(X_test, Y_test))


'5. Evaluation'
score = model.evaluate(X_test, Y_test, verbose=1)
print('Loss : ', score[0])
print('Accuracy : ', score[1])

no = round(score[1], 4)


'6. Visualization'
plt.subplot(121)
plt.plot(fit_hist.history['val_loss'], label='val_loss')
plt.plot(fit_hist.history['loss'], label='loss')
plt.legend()
plt.title('Loss & Val_loss')

plt.subplot(122)
plt.plot(fit_hist.history['val_accuracy'], label='val_acc')
plt.plot(fit_hist.history['accuracy'], label='acc')
plt.legend()
plt.title('Acc & Val_acc')


'7. Graph and Model Save'
import os
os.mkdir('./model/model_{}'.format(no))
plt.savefig('./model/model_{}/model_{}_graph.png'.format(no, no))
model.save('./model/model_{}/books_model_CM1+LD1+D4_ACC_{}.h5'.format(no, no))



'''
Ref. 1.>    Model Summary

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, 196, 100)          21831800  
_________________________________________________________________
conv1d (Conv1D)              (None, 196, 32)           16032     
_________________________________________________________________
max_pooling1d (MaxPooling1D) (None, 196, 32)           0         
_________________________________________________________________
dropout (Dropout)            (None, 196, 32)           0         
_________________________________________________________________
lstm (LSTM)                  (None, 196, 32)           8320      
_________________________________________________________________
dropout_1 (Dropout)          (None, 196, 32)           0         
_________________________________________________________________
flatten (Flatten)            (None, 6272)              0         
_________________________________________________________________
dense (Dense)                (None, 32)                200736    
_________________________________________________________________
dropout_2 (Dropout)          (None, 32)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 6)                 198       
=================================================================
Total params: 22,057,086
Trainable params: 22,057,086
Non-trainable params: 0
_________________________________________________________________


Ref. 2.>  Loss and Accuracy        
Evaluation loss : 0.7426667352172697
Evaluation accuracy : 0.8470359
'''