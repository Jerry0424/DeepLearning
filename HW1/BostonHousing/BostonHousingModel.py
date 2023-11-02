import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

filename = 'TheBostonHousingDataset.csv'
df = pd.read_csv(filename)



X = df.drop(columns=['MEDV'])
y = df['MEDV']
cols = X.columns


scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y.values.reshape(-1,1)).flatten()
X = pd.DataFrame(X, columns=cols)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)


model = Sequential([
    layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='linear')
])
optimizer = Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss='mean_squared_error')
model.summary()

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',  # Changed to 'val_loss'
    factor=0.5,
    verbose=1,
    mode='auto',
    min_delta=0.0001,
    cooldown=0,
    min_lr=0
)
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(X_train, y_train, validation_split=0.1, epochs=100, callbacks=[early_stopping, reduce_lr])

# Plot the training history
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


test_loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}')


preds = model.predict(X_test).flatten()
result = pd.DataFrame({'Actual': scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten(),
                       'Prediction': scaler_y.inverse_transform(preds.reshape(-1, 1)).flatten()})
result['Residual'] = abs(result['Actual'] - result['Prediction'])
print(result)

# Plot the Actual vs Predicted
plt.scatter(range(len(y_test)), scaler_y.inverse_transform(y_test.reshape(-1, 1)), color='green', label='Actual', alpha=0.5)
plt.scatter(range(len(preds)), scaler_y.inverse_transform(preds.reshape(-1, 1)), color='red', label='Prediction', alpha=0.5)
plt.title('Actual vs Predicted Values')
plt.xlabel('Index')
plt.ylabel('MEDV')
plt.legend()
plt.show()

# Plot the Residuals
plt.scatter(result['Prediction'], result['Residual'], alpha=0.5)
plt.title('Residuals of Predictions')
plt.xlabel('Predicted MEDV')
plt.ylabel('Residuals')
plt.show()