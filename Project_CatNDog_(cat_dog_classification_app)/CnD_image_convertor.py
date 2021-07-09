from PIL import Image
import glob
import numpy as np
from sklearn.model_selection import train_test_split

img_dir = '../datasets/train/'
categories = 'cat dog'.split()

# 모델에서 학습할 크기로 모두 동일하게 지정
image_w = 64
image_h = 64
pixel = image_h * image_w * 3

X = []
Y = []
files = None
# 0 : cat, 1 : dog
for idx, category in enumerate(categories):
    # glob : directory 내 파일 일름 읽어옴
    # 파일의 경로가 리스트로 저장
    files = glob.glob(img_dir + category + '*.jpg')
    # index와 file 이름 iter
    for i, f in enumerate(files):
        try:
            # image file open
            img = Image.open(f)
            # 이미지를 rgb로 바꿈
            img = img.convert('RGB')
            # file 사이즈는 tuple로 전달
            img = img.resize((image_w, image_h))
            # image 데이터를 array로 전달해야 함,
            data = np.asarray(img)
            X.append(data)
            Y.append(idx)
            # 300 개 마다 한번씩 print
            if i % 300 == 0:
                print(category, ':', f)
        except:
            # 예외처리 출력
            print(category, i, ' 번째에서 에러')

X = np.array(X)
Y = np.array(Y)

X = X / 255

print(X[0])
print(Y[0:5])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

xy = (X_train, X_test, Y_train, Y_test)

# np.save : 변수들의 원래형태로 저장해서 차후 로드할 때 변수들 그대로 불러옴
# 다른 파일형식으로 저장해서 로드하면 문자열로 인식함
np.save('../datasets/binary_image_data.npy', xy)