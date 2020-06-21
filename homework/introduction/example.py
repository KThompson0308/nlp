import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler


boston = datasets.load_boston()
data = np.asarray(boston['data'])
target = np.asarray(boston['target'])

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train = scaler.fit_transform(data)

model = tf.keras.Sequential()
model.add(layers.Dense(100, activation='sigmoid'))
model.add(layers.Dense(10))

model.compile(optimizer='sgd',
              loss='mean_squared_error',
              metrics=['mean_squared_error'])

model.fit(scaled_train, target, epochs=20, batch_size=32)
