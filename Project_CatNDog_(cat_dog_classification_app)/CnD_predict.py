import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import glob

'''
    경고 내용 숨김처리 solutions - no help
    
# export TF_CPP_MIN_LOG_LEVEL=2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.error)
'''



# 모델 불러오기 및 확인
model = load_model('../model/catNdog_binary_classification.h5')
print(model.summary())


# 테스트할 이미지 선택 및 전처리
img_dir = '../datasets/train/'
image_w = 64
image_h = 64
# dog / cat 파일 리스트 작성
dog_files = glob.glob(img_dir + 'dog*.jpg')
cat_files = glob.glob(img_dir + 'cat*.jpg')

# 개 이미지 sampling 및 파일경로 반환
dog_sample = np.random.randint(len(dog_files))
dog_sample_path = dog_files[dog_sample]

# 고양이 이미지 sampling 및 파일경로 반환
cat_sample = np.random.randint(len(cat_files))
cat_sample_path = cat_files[cat_sample]

print(dog_sample_path)
print(cat_sample_path)

try:
    # dog 파일 로드 및 전처리
    img = Image.open(dog_sample_path)
    img.show()
    img = img.convert('RGB')
    img = img.resize((image_w, image_h))
    data = np.asarray(img)
    data = data / 255
    dog_data = data.reshape(1,64,64,3)

    # cat 파일 로드 및 전처리
    img = Image.open(cat_sample_path)
    img.show()
    img = img.convert('RGB')
    img = img.resize((image_w, image_h))
    data = np.asarray(img)
    data = data / 255
    cat_data = data.reshape(1, 64, 64, 3)
except:
    print('error')
# print(dog_data.shape)
# print(cat_data.shape)

def classifier(num):
    if num == 1:
        return 'Dog'
    elif num ==0:
        return 'Cat'

# 모델 예측
print('dog data : ', classifier(model.predict(dog_data).round()))
print('cat data : ', classifier(model.predict(cat_data).round()))
