import os
import utils
import numpy as np
import cv2

path = 'D:\\guochuang\\Images\\kaggle猫狗大战数据集\\train\\train'


def getpictures(mount):
    image = np.zeros((mount*2, 224, 224, 3))
    label = np.zeros((2*mount, 2))
    for i in range(mount):
        cat_file = 'cat.' + str(i) + '.jpg'
        full_path = os.path.join(path, cat_file)
        Img = utils.ImageEncode(full_path)
        image[i, :, :, :] = Img[0, :, :, :]
        label[i, 0] = 1
    for i in range(mount, 2*mount):
        cat_file = 'dog.' + str(i) + '.jpg'
        full_path = os.path.join(path, cat_file)
        Img = utils.ImageEncode(full_path)
        image[i, :, :, :] = Img[0, :, :, :]
        label[i, 1] = 1

    return image, label


if __name__ == '__main__':
    image, label = getpictures(2000)
    cv2.imshow('mat', np.array(image[3, :, :, :], dtype=np.uint8))
    cv2.waitKey(0)
    print(label[3, :])