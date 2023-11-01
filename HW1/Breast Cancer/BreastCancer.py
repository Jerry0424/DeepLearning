from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
d = load_breast_cancer()
dataUse_x = pd.DataFrame(d.data, columns=d.feature_names)
dataUse_y = pd.DataFrame(d.target, columns=['target'])

#print(d.data)
#print(dataUse_y)




train_x, test_x = np.split(dataUse_x.sample(frac=1, random_state=42),
                                   [int(0.9 * len(dataUse_x))])

print(train_x.shape, test_x.shape)

train_y, test_y = np.split(dataUse_y.sample(frac=1, random_state=42),
                                   [int(0.9 * len(dataUse_y))])

print(train_y.shape, test_y.shape)


X_train = train_x.values
X_test = test_x.values

y_train = train_y.values
y_test = test_y.values
print(X_train.shape,X_test.shape)

print(y_train.shape,y_test.shape)


scaler = MinMaxScaler()


X_trainScaled = scaler.fit_transform(X_train)
X_testScaled = scaler.fit_transform(X_test)
Y_trainScaled = scaler.fit_transform(y_train)
Y_testScaled = scaler.fit_transform(y_test)


myCallback = tf.keras.callbacks.EarlyStopping(patience=50,mode="auto")
lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.1,
    patience=5,
    verbose=0,
    mode='auto',
    min_delta=0.00000001)

device_name="/GPU:0"
with tf.device(device_name):
    model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=128, input_shape=[30], activation='relu'),
            tf.keras.layers.Dropout(0.8),
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dropout(0.8),
            tf.keras.layers.Dense(units=1, activation='sigmoid')
        ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=15)
    history = model.fit(X_trainScaled, Y_trainScaled,validation_data=(X_testScaled, Y_testScaled), epochs=100,callbacks=[early_stopping],verbose=1)



plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

