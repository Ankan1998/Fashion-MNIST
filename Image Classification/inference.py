import numpy as np
import tensorflow as tf
from arch.basic_convnet import model
import cv2


def inference(image,model = model, model_path = 'tmp/ckpt'):

    map_class = {
        0: "T-shirt/top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot",
    }

    img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2GRAY)
    #print(img.shape)
    img_nmp = cv2.resize(img,(28,28))
    #print(img_nmp.shape)
    img_nmp_reshaped = img_nmp.reshape((1,28,28,1))
    model_inference = model
    model_inference.load_weights(model_path)

    y_pred = model_inference.predict(img_nmp_reshaped)
    # maximum value's index from the numpy array y_pred
    classified = np.argmax(y_pred, axis=1)
    #print(map_class[classified[0]])
    return map_class[classified[0]]


if __name__ == '__main__':
    image = r'C:\Users\Ankan\Desktop\Github\Fashion-MNIST\input_image\shirt.png'

    object = inference(image,model = model,model_path = 'tmp/ckpt')