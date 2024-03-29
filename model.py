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
from tensorflow.keras.optimizers import Adam,SGD
import keras_tuner as kt
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.callbacks import LearningRateScheduler

def test():
    test_predict = model.predict(X_test)
    num_features = 36
    test_predict_extended = np.zeros((len(test_predict), num_features))
    test_predict_extended[:, -1] = test_predict.ravel() 
    y_test_extended = np.zeros((len(y_test), num_features))
    y_test_extended[:, -1] = y_test.ravel()  # 确保y_test是正确的形状
    test_predict_inversed = scaler.inverse_transform(test_predict_extended)[:, -1]
    y_test_inversed = scaler.inverse_transform(y_test_extended)[:, -1]
    return 1-r_squared(y_test_inversed, test_predict_inversed)/10000

class TestModelCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(TestModelCallback, self).__init__()
        self.res = [0.38]

    def on_epoch_end(self, epoch, logs=None):
        # 在这里执行测试
        res = test()
        if np.array(self.res).max() < res :
            model_filename = 'model' + str(int(res*10000)) + '.h5'
            # 保存模型
            model.save(model_filename)
        print('epoch: %d, r2: %f, best r2: %f' % (epoch, res , np.array(self.res).max()))
        self.res.append(res)
def scheduler(epoch, lr):
    if epoch >250 :
        return lr
    if epoch % 10 == 0 and epoch > 0:
        return lr * 0.9
    else:
        return lr
def r_squared(y_true, y_pred):
    residual = y_true - y_pred
    ss_res = tf.reduce_sum(residual**2)
    ss_tot = tf.reduce_sum((y_true - tf.reduce_mean(y_true))**2)
    r2 = ss_res/ss_tot
    return 10000*r2

# 读取数据集
data = pd.read_csv('data/1.csv')

scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)
# 划分特征和标签
X = data[:, 1:-1]  # 特征
y = data[:, -1]   # 标签
X = X.reshape(X.shape[0], X.shape[1], 1)
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = Sequential()
model.add(LSTM(40, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]),kernel_regularizer=l2(0.1),kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(LSTM(80, return_sequences=False,kernel_regularizer=l2(0.01),kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Dense(160, kernel_regularizer=l2(0.01),kernel_initializer='he_uniform'))  
model.add(BatchNormalization())
model.add(Dense(16, kernel_regularizer=l2(0.01),kernel_initializer='he_uniform')) 
model.add(BatchNormalization())
model.add(Dense(1))
# 使用均方误差作为loss函数
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=256, epochs=500, verbose=1,callbacks=[TestModelCallback(), LearningRateScheduler(scheduler)], use_multiprocessing=True, workers=8)
train_predict = model.predict(X_train)
num_features = 36
train_predict_extended = np.zeros((len(train_predict), num_features))
train_predict_extended[:, -1] = train_predict.ravel() 
train_predict_inversed = scaler.inverse_transform(train_predict_extended)[:, -1]
y_train_extended = np.zeros((len(train_predict), num_features))
y_train_extended[:, -1] = y_train.ravel() 
y_train_inversed = scaler.inverse_transform(y_train_extended)[:, -1]
# 绘制结果
plt.figure(figsize=(10,5))
plt.plot(np.arange(0,100), y_train_inversed[0:100], label='Original Training Data')
plt.plot(np.arange(0,100), train_predict_inversed[0:100], label='Predicted Training Data')
plt.legend()
plt.show()
# plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test, label='Original Test Data')
# plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), test_predict, label='Predicted Test Data')
# plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_loss, label='Predicted Loss Data')

# plt.legend()
# plt.show()
model_filename = 'model_' + 'last' + '.h5'
model.save(model_filename)


