from time import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard


digit = datasets.load_digits()
data = digit['data']
target = digit['target']
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=1776)

model = tf.keras.Sequential()
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer=tf.optimizers.SGD(0.01),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=10)
print(np.argmax(model.predict(x_test), axis=1))
print(y_test)
