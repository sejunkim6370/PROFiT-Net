import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv1D, AveragePooling1D, Dropout, MaxPooling1D, Flatten, Dense 
import os

X_train = np.load('X_train.npy')
X_val = np.load('X_val.npy')
y_train = np.load('y_train.npy')
y_val = np.load('y_val.npy')

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

checkpoint_path = 'callback/cp.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 mode='min',
                                                 verbose=0,
                                                 monitor='val_mean_absolute_error')

model.fit(X_train,
          y_train,
          batch_size=512,
          epochs=500,
          verbose=1,
          validation_data=(X_val, y_val),
          callbacks=[cp_callback])
