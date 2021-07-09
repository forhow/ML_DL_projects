"""
    Project : GreenRight
        - 식물(잎) 이미지를 통한 식물 분류 및 병충해 판별기 구축

    Model Test

    1. Test Setting

    2. Test Function Definition

    3. Model Test
     a. Model Test with Random Images - 20 Classes, 50 Cases
     b. Test Result on a Class
     c. Test Result of Total Cases

"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import random
import os


'1. Test Setting'
combined_labels = ["Corn_(maize)_Common_rust_", "Corn_(maize)_healthy", "Grape_Black_rot",
                   "Grape_Esca_(Black_Measles)", "Grape_Leaf_blight_(lsariopsis_Leaf_Spot)",
                   "Orange_Haunglongbing_(Citrus_greening)", "Pepper,_bell_Bacterial_spot",
                   "Pepper,_bell_healthy", "Potato_Early_blight", "Potato_Late_blight",
                   "Soybean_healthy", "Squash_Powdery_mildew", "Tomato_Bacterial_spot",
                   "Tomato_Early_blight", "Tomato_Late_blight", "Tomato_Septoria_leaf_spot",
                   "Tomato_Spider_mites_Two-spotted_spider_mite", "Tomato_Target_Spot",
                   "Tomato_Tomato_Yellow_Leaf_Curl_Virus", "Tomato_healthy"]

# Label Dictionary
t_list = list(str(i) for i in range(20))
t_dict = dict(zip(t_list, combined_labels))

# Test data List Load
tests = './tests'
tests_list = os.listdir(tests)

# Model Load
model = load_model('CNN_model4.h5')
sum_count = 0


'2. Test Function Definition'

'2.a. convert_image_to_array'
def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None:
            image = cv2.resize(image, dsize=(64, 64))
            return img_to_array(image)
        else:
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None

'2.b. predict_disease'
def predict_disease(image_path, num):
    global count
    global sum_count
    image_array = convert_image_to_array(image_path)
    np_image = np.array(image_array, dtype=np.float32) / 225.0
    np_image = np.expand_dims(np_image, 0)
    result = model.predict_classes(np_image)
    c = result.astype(str)[0]
    if c == num:
        count += 1
        sum_count += 1
    return count, sum_count


'3. Model Test'

'3.a. Model Test with Random Images - 20 Classes, 50 Cases'
for i in range(20):
    num = str(i)
    tests_file = os.listdir(f'tests/{num}')
    count = 0
    max = len(tests_file)

    for j in range(50):
        ran_num = random.randint(0, max)  # 임의의 숫자 추출
        tests_path = f'tests/{num}/' + os.listdir(f'./tests/{num}')[ran_num]
        predict_disease(tests_path, num)

    '3.b. Test Result on a Class'
    print(f'###### 테스트 데이터 {t_dict.get(num)} 의 정확도 입니다 #######')
    print('accuracy: {:0.5f}'.format(count / 50))

'3.c. Test Result of Total Cases'
print("----------------------------------------------------------------------------------")
print('total_accuracy: {:0.5f}'.format(sum_count / 1000))
print('테스트 완료')
