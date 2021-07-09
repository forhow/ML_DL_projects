import numpy as np
from PIL import Image
import glob
from sklearn.model_selection import train_test_split

root_dir = '../datas/horse-or-human/'
categories = 'humans horses'.split()
files=[]
X = []
Y = []
for idx, category in enumerate(categories):
    files = glob.glob(root_dir + category + '/*.png')
    # print(len(files))
    for ii, file in enumerate(files):
        img = Image.open(file)
        img = img.convert('RGB')
        img = img.resize((64,64))
        data = np.asarray(img)
        data = data / 255
        X.append(data)
        Y.append(idx)
        if ii % 100 == 0:
            print(category, ii, ' completed')

X = np.array(X)
Y = np.array(Y)

print(X.shape)
print(Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15)

xy = (X_train, X_test, Y_train, Y_test)

np.save('../datasets/human_horse_dataset.npy', xy)

