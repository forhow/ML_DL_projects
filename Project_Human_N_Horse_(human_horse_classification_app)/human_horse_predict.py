from tensorflow.keras.models import load_model
import glob
import numpy as np
from PIL import Image

model = load_model('../model/human_horse_classification.h5')
# print(model.summary())

root_dir = '../datas/horse-or-human/'
categories = 'humans horses'.split()
file_list = []
for categ in categories:
    files = glob.glob(root_dir + categ + '/*.png')
    for i in files:
        file_list.append(i)

numb = np.random.randint(len(file_list))
sample_file = file_list[numb]
# print(sample_file)

# file pre-processing
def img2data(path):
    img = Image.open(sample_file)
    img.show()
    img = img.convert('RGB')
    img = img.resize((64,64))
    data = np.asarray(img)
    data = data / 255
    data = data.reshape(1,64,64,3)
    print(data.shape)
    return data

def classifier(num):
    if num > 0.5:
        return 'Horse'
    else:
        return 'Human'


data = img2data(sample_file)

pred = model.predict(data)[0][0]
print(sample_file.replace(root_dir, ''))
print(classifier(pred))

