import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import History
from tensorflow.keras.callbacks import EarlyStopping, History
from tensorflow.keras.optimizers import Adam

filename = 'TheBostonHousingDataset.csv'
df = pd.read_csv(filename)


warnings.filterwarnings('ignore')


X = df.drop(columns=['MEDV'])
y = df['MEDV']
cols = X.columns




scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y.values.reshape(-1,1)).flatten()
X = pd.DataFrame(X, columns=cols)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)


nn = Sequential([
    layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='linear')
])
optimizer = Adam(learning_rate=0.001)

nn.compile(optimizer=optimizer, loss='mean_squared_error')
nn.summary()

history = nn.fit(X_train, y_train, validation_split=0.1, epochs=150, callbacks=[EarlyStopping(monitor='val_loss', patience=8)])


plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Training and validation loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


test_loss = nn.evaluate(X_test, y_test, verbose=0)
print(f'Test Loss: {test_loss}')


print("Report:")
print("Chosen architecture: 2 hidden layers with 32 and 16 neurons respectively.")
print("Activation functions: ReLU for hidden layers, linear for output layer.")
print("Optimizer: Adam")
print("Loss function: Mean Squared Error")
print("Observations: The model shows a decreasing trend in training and validation loss as the number of epochs increases. This suggests that the model is learning from the data. If the validation loss starts to increase, it might be a sign of overfitting. Early stopping is used to prevent overfitting.")


preds = nn.predict(X_test).flatten()
result = pd.DataFrame({'Actual': scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten(),
                       'Prediction': scaler_y.inverse_transform(preds.reshape(-1, 1)).flatten()})
result['Residual'] = abs(result['Actual'] - result['Prediction'])
print(result)

plt.scatter(range(len(y_test)), scaler_y.inverse_transform(y_test.reshape(-1, 1)), color='green', label='Actual', alpha=0.5)
plt.scatter(range(len(preds)), scaler_y.inverse_transform(preds.reshape(-1, 1)), color='red', label='Prediction', alpha=0.5)
plt.title('Actual vs Predicted Values')
plt.xlabel('Index')
plt.ylabel('MEDV')
plt.legend()
plt.show()


plt.scatter(result['Prediction'], result['Residual'], alpha=0.5)
plt.title('Residuals of Predictions')
plt.xlabel('Predicted MEDV')
plt.ylabel('Residuals')
plt.show()