from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import utils
import DogsCat as DC

font = cv2.FONT_HERSHEY_SIMPLEX

model = ResNet50(
    weights=None,
    classes=2
)

if __name__ == '__main__':

    # Images, labels = DC.getpictures(3000)
    #
    # model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    # model.fit(
    #     x=Images,
    #     y=labels,
    #     epochs=10,
    #     batch_size=5
    # )
    #
    #
    # model.save('my_model.h5')

    imgpath = "cat.jpg"

    model = load_model('dogcat.h5')

    code = utils.ImageEncode(imgpath)
    ret = model.predict(code)
    res1 = np.argmax(ret[0, :])

    img = cv2.imread(imgpath)
    if res1:
        cv2.putText(img, 'dog', (50, 100), font, 2, (255, 255, 255), 7)
        cv2.imshow('mat', img)
    else:
        cv2.putText(img, 'cat', (50, 100), font, 2, (255, 255, 255), 7)
        cv2.imshow('mat', img)

    cv2.waitKey(0)