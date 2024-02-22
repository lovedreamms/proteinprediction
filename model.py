import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import datetime
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization,LSTM, Dense
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.callbacks import LearningRateScheduler

def test():
    test_predict = model.predict(X_test)
    return (1-r_squared(y_test, test_predict))/1000

class TestModelCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(TestModelCallback, self).__init__()
        self.res = [0.4]

    def on_epoch_end(self, epoch, logs=None):
        # 在这里执行测试
        res = test()
        if np.array(self.res).max() < res :
            model_filename = 'model' + str(int(res*10000)) + '.h5'
            # 保存模型
            model.save(model_filename)
        print(f'res:{res}')
        self.res.append(res)
def scheduler(epoch, lr):
    if epoch % 10 == 0 and epoch > 0:
        return lr * 0.9
    else:
        return lr
def r_squared(y_true, y_pred):
    residual = y_true - y_pred
    ss_res = tf.reduce_sum(residual**2)
    ss_tot = tf.reduce_sum((y_true - tf.reduce_mean(y_true))**2)
    r2 = 1 - ss_res/ss_tot
    return (1-r2)*1000

# 读取数据集
data = pd.read_csv('data/1.csv')

# 划分特征和标签
X = data.iloc[:, 1:-1].values  # 特征
y = data.iloc[:, -1].values    # 标签
X = X.reshape(X.shape[0], X.shape[1], 1)
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print(np.array(X_train).shape,np.array(y_train).shape)

model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]),kernel_regularizer=l1(0.01),kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(LSTM(200, return_sequences=False,kernel_regularizer=l1(0.01),kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Dense(1024,kernel_regularizer=l1(0.01),kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Dense(512, kernel_regularizer=l1(0.01),kernel_initializer='he_uniform'))  
model.add(BatchNormalization())
model.add(Dense(64, kernel_regularizer=l1(0.01),kernel_initializer='he_uniform')) 
model.add(BatchNormalization())
model.add(Dense(1))
model.compile(optimizer=Adam(learning_rate=0.001), loss=r_squared)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=128, epochs=30, verbose=1,callbacks=[TestModelCallback(), LearningRateScheduler(scheduler)])
test_predict = model.predict(X_test)
print(test_predict)
# 绘制结果
plt.figure(figsize=(10,5))
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test, label='Original Test Data')
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), test_predict, label='Predicted Test Data')
plt.legend()
plt.show()
model_filename = 'model_' + 'last' + '.h5'
model.save(model_filename)

