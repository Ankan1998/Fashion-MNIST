import tensorflow as tf

model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(256, (3,3), activation='relu', input_shape=(28, 28, 1)), # Here 256 feature detector are used, normally 64 is enough
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu',input_shape=(28, 28, )),
  tf.keras.layers.Dense(10, activation='softmax')
])