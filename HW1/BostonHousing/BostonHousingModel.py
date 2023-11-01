import pandas as pd
import numpy as np
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

# 分离特征和目标变量
X = df.drop(columns=['MEDV'])
y = df['MEDV']
cols = X.columns

# 标准化特征和目标变量
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y.values.reshape(-1,1)).flatten()  # .flatten() 将 y 转换回一维数组以匹配预测输出

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 建立模型
nn = Sequential([
    layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='linear')
])

# 设置优化器和编译模型
optimizer = Adam(learning_rate=0.001)
nn.compile(optimizer=optimizer, loss='mean_squared_error')
nn.summary()

# 使用EarlyStopping回调和训练模型
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
history = nn.fit(X_train, y_train, validation_split=0.1, epochs=100, callbacks=[early_stopping])

# 绘制训练和验证损失曲线
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Training and validation loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 在测试集上评估模型
test_loss = nn.evaluate(X_test, y_test, verbose=0)
print(f'Test Loss: {test_loss}')

# 输出报告
print("Report:")
print("Chosen architecture: 2 hidden layers with 32 and 16 neurons respectively.")
print("Activation functions: ReLU for hidden layers, linear for output layer.")
print("Optimizer: Adam")
print("Loss function: Mean Squared Error")
print("Observations: ...")  # 根据模型的实际表现填写观察结果

# 预测测试集
preds = nn.predict(X_test).flatten()
result = pd.DataFrame({'Actual': scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten(),
                       'Prediction': scaler_y.inverse_transform(preds.reshape(-1, 1)).flatten()})
result['Residual'] = abs(result['Actual'] - result['Prediction'])
print(result)

# 绘制实际值和预测值
plt.scatter(range(len(y_test)), scaler_y.inverse_transform(y_test.reshape(-1, 1)), color='green', label='Actual', alpha=0.5)
plt.scatter(range(len(preds)), scaler_y.inverse_transform(preds.reshape(-1, 1)), color='red', label='Prediction', alpha=0.5)
plt.title('Actual vs Predicted Values')
plt.xlabel('Index')
plt.ylabel('MEDV')
plt.legend()
plt.show()

# 绘制残差
plt.scatter(result['Prediction'], result['Residual'], alpha=0.5)
plt.title('Residuals of Predictions')
plt.xlabel('Predicted MEDV')
plt.ylabel('Residuals')
plt.show()
