from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load dataset
d = load_breast_cancer()
X = pd.DataFrame(d.data, columns=d.feature_names)
y = pd.DataFrame(d.target, columns=['target'])

# Split the dataset into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Scale the features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Only transform on the test set

# Define the model
model = Sequential([
    Dense(units=128, input_shape=[30], activation='relu'),
    Dropout(0.5),  # Reduced dropout rate
    Dense(units=64, activation='relu'),
    Dropout(0.5),
    Dense(units=1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=15)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',  # Changed to 'val_loss'
    factor=0.1,
    patience=10 ,
    verbose=1,
    mode='auto',
    min_delta=0.0001,
    cooldown=0,
    min_lr=0
)

# Train the model
history = model.fit(
    X_train_scaled,
    y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=100,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Evaluate the model
eval_results = model.evaluate(X_test_scaled, y_test)
print(f"\nTest Loss: {eval_results[0]}, Test Accuracy: {eval_results[1]}")

# Predictions for confusion matrix
y_pred = model.predict(X_test_scaled)
y_pred_classes = (y_pred > 0.5).astype("int32")


# Plot the training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()


conf_matrix = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Plot the training and validation loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()



'''
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
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
y = scaler_y.fit_transform(y.values.reshape(-1,1)).flatten()  # .flatten() 将 y 转换回一维数组以匹配预测输出


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


nn = Sequential([
    layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='linear')
])


optimizer = Adam(learning_rate=0.001)
nn.compile(optimizer=optimizer, loss='mean_squared_error')
nn.summary()


early_stopping = EarlyStopping(monitor='val_loss', patience=10)
history = nn.fit(X_train, y_train, validation_split=0.1, epochs=100, callbacks=[early_stopping])


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
print("Observations: ...")  # 根据模型的实际表现填写观察结果


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



'''