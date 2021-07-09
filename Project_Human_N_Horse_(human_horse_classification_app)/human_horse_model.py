import numpy as np

# data load
X_train, X_test, Y_train, Y_test = np.load('../datasets/human_horse_dataset.npy', allow_pickle=True)
# print(X_train.shape)
# print(X_test.shape)
# print(Y_train.shape)
# print(Y_test.shape)


# modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten

model = Sequential()
model.add(Conv2D(128, kernel_size=(3,3), strides=1, padding='same',
                 input_shape=(64,64,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.1))
model.add(Conv2D(64, kernel_size=(3,3), strides=1, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.1))
model.add(Conv2D(16, kernel_size=(3,3), strides=1, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())


# compile
model.compile(loss= 'mse', optimizer='adam', metrics=['binary_accuracy'])

# training
fit_hist = model.fit(X_train, Y_train,
                     batch_size=32,
                     epochs=10,
                     verbose=1,
                     validation_split=0.2,
                     )
# model save
model.save('../model/human_horse_classification.h5')

# model evaluation
score = model.evaluate(X_test, Y_test, verbose=1)
print('Loss :', score[0])
print('Accuaray :', score[1])


# visualiztion
import matplotlib.pyplot as plt
plt.plot(fit_hist.history['loss'], label='loss')
plt.plot(fit_hist.history['val_loss'], label='val_loss')
plt.title('loss')
plt.legend()
plt.show()

plt.plot(fit_hist.history['binary_accuracy'], label='acc')
plt.plot(fit_hist.history['val_binary_accuracy'], label='val_acc')
plt.title('Accs')
plt.legend()
plt.show()


'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 64, 64, 128)       3584      
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 32, 32, 128)       0         
_________________________________________________________________
dropout (Dropout)            (None, 32, 32, 128)       0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 32, 32, 64)        73792     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 16, 16, 64)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 16, 16, 64)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 16, 16, 16)        9232      
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 8, 8, 16)          0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 8, 8, 16)          0         
_________________________________________________________________
flatten (Flatten)            (None, 1024)              0         
_________________________________________________________________
dense (Dense)                (None, 128)               131200    
_________________________________________________________________
dropout_3 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 64)                8256      
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 65        
=================================================================
Total params: 226,129
Trainable params: 226,129
Non-trainable params: 0
_________________________________________________________________


Loss : 0.005427143070846796
Accuaray : 0.9870967864990234


'''