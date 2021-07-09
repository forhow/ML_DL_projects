"""
    Project : GreenRight
        - 식물(잎) 이미지를 통한 식물 분류 및 병충해 판별기 구축

    Modeling, Training, Evaluation : Alex Net

    1. Setting

    2. Function Definition - AlexNet Model

    3. Model Training

    4. Model Evaluation

    5. Training Result Visualization
     a. Evaluation Result
     b. Accuracy & Validation Accuracy
     c. Loss & Validation Loss

"""


import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.layers import Flatten, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import Model
from keras.optimizers import Adam
import numpy as np
import os
from matplotlib import pyplot as plt


'1. Setting'
plant_label = ["Apple", "Blueberry", "Cherry_(including_sour)", "Corn_(maize)",
              "Grape", "Orange", "Peach", "Pepper,_bell", "Potato", "Raspberry",
              "Soybean", "Squash", "Strawberry", "Tomato"]

disease_label = ["Apple_scab", "Bacterial_spot", "Black_rot", "Cedar_apple_rust",
                 "Cercospora_leaf_spot_Gray_leaf_spot", "Common_rust_",
                 "Early_blight", "Esca_(Black_Measles)", "Haunglongbing_(Citrus_greening)",
                 "Late_blight", "Leaf_Mold", "Leaf_blight_(lsariopsis_Leaf_Spot)",
                 "Leaf_scorch", "Northem_Leaf_Blight", "Powdery_mildew", "Septoria_leaf_spot",
                 "Spider_mites_Two-spotted_spider_mite", "Target_Spot",
                 "Tomato_Yellow_Leaf_Curl_Virus", "Tomato_mosaic_virus", "healthy"]

combined_labels = ["Corn_(maize)_Common_rust_", "Corn_(maize)_healthy", "Grape_Black_rot",
                   "Grape_Esca_(Black_Measles)", "Grape_Leaf_blight_(lsariopsis_Leaf_Spot)",
                   "Orange_Haunglongbing_(Citrus_greening)", "Pepper,_bell_Bacterial_spot",
                   "Pepper,_bell_healthy", "Potato_Early_blight", "Potato_Late_blight",
                   "Soybean_healthy", "Squash_Powdery_mildew", "Tomato_Bacterial_spot",
                   "Tomato_Early_blight", "Tomato_Late_blight", "Tomato_Septoria_leaf_spot",
                   "Tomato_Spider_mites_Two-spotted_spider_mite", "Tomato_Target_Spot",
                   "Tomato_Tomato_Yellow_Leaf_Curl_Virus", "Tomato_healthy"]

p_classes = len(plant_label)
d_classes = len(disease_label)
com_classes = len(combined_labels)

# Training Hyper parameter setting
tf.compat.v1.enable_eager_execution()
learning_rate = 1e-3
epchs = 60
BS = 128
display_step = 20

# Optimizer
optimizer = Adam(learning_rate=learning_rate)

# Model Save Path
model_path = './model/Dense_test/AlexNet_Dense20481024_BS128_E50.h5'

# Label Dictionary
list0 = list(str(i) for i in range(com_classes))
dict0 = dict(zip(list0, combined_labels))

# Dataset Load
X_train, X_test, Y_train, Y_test = np.load('./dataset/image_data.npy', allow_pickle=True)
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
Y_train = Y_train.astype(np.float32)
Y_test = Y_test.astype(np.float32)


'2. Function Definition - AlexNet Model'
def Alexnet_model():
    inputs = Input(shape=(64, 64, 3))

    conv1 = Conv2D(filters=48, kernel_size=(10, 10), strides=1, padding="valid", activation='relu')(inputs)
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=2, padding="valid")(conv1)
    nor1 = tf.nn.local_response_normalization(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75)

    conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding="same", groups=2, activation='relu')(nor1)
    pool2 = MaxPooling2D(pool_size=(3, 3), strides=2, padding="valid")(conv2)
    nor2 = tf.nn.local_response_normalization(pool2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    conv3 = Conv2D(filters=192, kernel_size=(3, 3), strides=1, padding="same", activation='relu')(nor2)
    conv4 = Conv2D(filters=192, kernel_size=(3, 3), strides=1, padding="same", activation='relu')(conv3)
    conv5 = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding="same", activation='relu')(conv4)
    pool3 = MaxPooling2D(pool_size=(3, 3), strides=2, padding="valid")(conv5)
    drop1 = Dropout(0.5)(pool3)
    nor3 = tf.nn.local_response_normalization(drop1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    flat = Flatten()(nor3)
    dense1 = Dense(units=2048, activation='relu')(flat)
    dense2 = Dense(units=1024, activation='relu')(dense1)
    logits = Dense(units=20, activation='softmax')(dense2)

    return Model(inputs=inputs, outputs=logits)

model = Alexnet_model()
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=["accuracy"])
print(model.summary())


'3. Model Training'
if os.path.exists(model_path):
    model.load_weights(model_path)
else:
    # Training Option
    # checkpoint = ModelCheckpoint(filepath=model_path , monitor='val_loss', verbose=1, save_best_only=True)
    # early_stopping = EarlyStopping(monitor='val_loss', patience=6)

    history = model.fit(X_train,
                        Y_train,
                        batch_size=BS,
                        validation_data=(X_test, Y_test),
                        epochs=50, verbose=1,)
    model.save_weights(model_path)


'4. Model Evaluation'
print("Train 정확도 : %.4f" % (model.evaluate(X_train, Y_train)[1]))
print("Test 정확도 : %.4f" % (model.evaluate(X_test, Y_test)[1]))


'5. Training Result Visualization'

'5.a. Evaluation Result'
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

'5.b. Accuracy & Validation Accuracy'
plt.rc('font', family='Malgun Gothic')
plt.plot(epochs, acc, 'b', label='Training accurarcy')
plt.plot(epochs, val_acc, 'r', label='Testing accurarcy')
plt.title('학습과 훈련 정확도')
plt.legend()
plt.savefig('./model/Dense_test/AlexNet_Dense20481024_BS128_E30_accuracy.png')
plt.figure()

'5.c. Loss & Validation Loss'
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Testing loss')
plt.title('학습과 훈련 손실')
plt.legend()
plt.savefig('./model/Dense_test/AlexNet_Dense20481024_BS128_E30_loss.png')
plt.show()




