import os,sys
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions


def GetLabel(path):
    file = open(path)
    filenames = []
    labels = []
    for line in file:
        filename, label = line.split(' ')
        filenames.append(filename)
        labels.append(int(label))
    return filenames, labels


def getDecodes(labels):
    wage = max(labels)
    count = len(labels)
    lbl = np.zeros((count, wage+1))
    for i in range(len(labels)):
        lbl[i, labels[i]] = 1
    return lbl


def getImgRect(dirname, filenames):
    length = len(filenames)
    Images = np.zeros((length, 224, 224, 3))

    for i in range(len(filenames)):
        fileName = os.path.join(dirname, filenames[i])
        Image = ImageEncode(fileName)
        Images[i, :, :, :] = Image[0, :, :, :]

    return Images


def transClasses(filename):
    kinds = {}
    f = open(filename)
    for line in f:
        label, kind = line.split(' ')
        label = int(label)
        kind = kind.strip()
        kinds[label] = kind
    return kinds

def ImageEncode(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

if __name__ == '__main__':
    x = ImageEncode('D:\\guochuang\\Images\\kaggle猫狗大战数据集\\train\\train\\cat.0.jpg')
    print(x.shape)