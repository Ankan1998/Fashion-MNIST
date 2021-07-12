import tensorflow as tf
from arch.basic_convnet import model
from data_src.data import MNIST

def inference(x_test,y_test,model = model, model_path = '/tmp/ckpt'):

    model_inference = model
    model_inference.load_weights(model_path)

    y_pred = model_inference.evaluate(x_test)