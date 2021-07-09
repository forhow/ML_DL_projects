"""
    Project : GreenRight
        - 식물(잎) 이미지를 통한 식물 분류 및 병충해 판별기 구축

    Pre-processing : Dataset Creation

    1. Setting
    2. Function Definition
     a. Dataset Creation - mk_dataset
     b. Directory Categorization & Image Copy Function - copy_image
    3. Function Execution
"""

import cv2
from sklearn.model_selection import train_test_split
import numpy as np
import shutil, glob, os

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


'2. Function Definition'

'2.a. Dataset Creation Function'
def mk_dataset(label_name, label_classes, image_w, image_h):
    if not os.path.exists("./dataset/"):
        os.mkdir('./dataset/')
    X = []
    Y = []
    for index, p_label in enumerate(label_name):
        label = [0 for i in range(label_classes)]
        label[index] = 1
        image_dir = f'../train_division/{label_name[index]}/'
        for top, dir, f in os.walk(image_dir):
            for filename in f:
                print(image_dir + filename)
                img = cv2.imread(image_dir + filename)
                img = cv2.resize(img, dsize=(image_w, image_h))
                # 컬러 변경 참고 : https://076923.github.io/posts/Python-opencv-10/
                # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)# RGB를 흑백으로
                # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) # 흑백을 RGB로
                X.append(img/255)
                Y.append(label)
    X = np.array(X)
    Y = np.array(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
    # 데이터셋 저장시 return 기능 주석하시고 사용하시면 됩니다. xy, save 기능 활성화 시키시면 됩니다.
    # 데이터셋을 저장하지 않고 불러들이는 것으로 하는 경우 xy, np.save 기능 주석하시고 return 활성화 시키시면 됩니다.
    xy = (X_train, X_test, Y_train, Y_test)
    np.save('../datasets/image_data.npy', xy)
    # return X_train, X_test, Y_train, Y_test

'2.b. Directory Categorization & Image Copy Function'
def copy_image(label_name, new_dir_name=''):
    path = "../train"
    dir_name = '../{}/'.format(new_dir_name)
    file_list = os.listdir(path)
    if not os.path.exists(dir_name):
        print('폴더 생성중')
        os.mkdir(dir_name)
        for i in range(len(label_name)):
            os.mkdir(dir_name + label_name[i])
            print(dir_name + label_name + '폴더 생성 완료.')
        print('새로운 폴더 모두 생성 완료.')

    for i in range(len(os.listdir(path))):
        cl_list = file_list[i][:-4].split('_')  # 이름을 나누기 plant + disease + 번호
        folder_name = plant_label[int(cl_list[0])]+'_'+disease_label[int(cl_list[1])]
        if file_list[i] not in path:
            shutil.copy(f'{path}/{file_list[i]}', f'{dir_name}{folder_name}')
        else:
            print('복사완료')
            break


p_classes = len(plant_label)
d_classes = len(disease_label)
com_classes = len(combined_labels)


'3. Function Execution'
mk_dataset(combined_labels, com_classes, 64, 64) # 가로, 세로 64의 픽셀값을 가진 사진으로 데이터셋 생성
# copy_image(label_name, new_dir_name='') # 라벨 이름, 생성할 디렉토리 이름

