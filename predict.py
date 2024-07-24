import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv1D, AveragePooling1D, Dropout, MaxPooling1D, Flatten, Dense 
from sklearn.metrics import mean_absolute_error, mean_squared_error

X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

sample_shape = (18496,1)

model = Sequential()
model.add(Conv1D(21, kernel_size=4, activation='relu', kernel_initializer='he_uniform', input_shape=sample_shape))
model.add(AveragePooling1D(pool_size=6))
model.add(Dropout(0.01))
model.add(Conv1D(11, kernel_size=9, activation='relu', kernel_initializer='he_uniform', input_shape=sample_shape))
model.add(Dropout(0.01))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(9, kernel_size=14, activation='relu', kernel_initializer='he_uniform', input_shape=sample_shape))
model.add(Dropout(0.02))
model.add(Conv1D(9, kernel_size=3, activation='relu', kernel_initializer='he_uniform', input_shape=sample_shape))
model.add(Dropout(0.01))
model.add(Conv1D(9, kernel_size=1, activation='relu', kernel_initializer='he_uniform', input_shape=sample_shape))
model.add(Dropout(0.01))
model.add(Flatten())
model.add(Dense(1024, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(512, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(16, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(1, kernel_initializer='he_uniform'))

model.compile(loss='mse',
              optimizer=tf.keras.optimizers.Adam(lr=0.001),
              metrics=['MeanAbsoluteError'])

model.load_weights('callback/cp.ckpt')
model.predict(X_test)
