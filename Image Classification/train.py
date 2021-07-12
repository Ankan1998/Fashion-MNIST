
import tensorflow as tf
from arch.basic_convnet import model
from data_src.data import MNIST

# Create a model compilar with optimizer as 'adam' and loss function as 'categorical_crossentropy' and metric as 'accuracy'
def training(x_train,
            y_train,
            x_val,
            y_val,
            x_test,
            y_test,
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
            batch_size=32,
            epochs=25,
            ckt_path='/tmp/ckpt'):

    model.compile(optimizer=optimizer,loss=loss,metrics=metrics)
    model.summary()

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=ckt_path,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

    model.fit(x_train,
            y_train, 
            batch_size = batch_size, 
            epochs=epochs,
            validation_data=(x_val,y_val),
            shuffle=True,
            callbacks=[model_checkpoint_callback])




if __name__ == '__main__':
    mnist = MNIST()
    x_train, y_train, x_val, y_val, x_test, y_test = mnist.data_loader('mnist.npz')
    training(x_train, y_train, x_val, y_val, x_test, y_test, epochs=1)
    


