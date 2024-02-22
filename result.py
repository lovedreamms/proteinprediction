import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

def r_squared(y_true, y_pred):
    residual = y_true - y_pred
    ss_res = tf.reduce_sum(residual**2)
    ss_tot = tf.reduce_sum((y_true - tf.reduce_mean(y_true))**2)
    r2 = ss_res/ss_tot
    return 10000*r2
from tensorflow import keras

data = pd.read_csv('data/1.csv')
scaler1 = MinMaxScaler(feature_range=(0, 1))
scaler2 = MinMaxScaler(feature_range=(0, 1))
data = scaler1.fit_transform(data)
data = pd.read_csv('data/test.csv').values
data = scaler2.fit_transform(data)
X = data[:, 1:-1]  # 特征
y = data[:, -1]   # 标签
with keras.utils.custom_object_scope({'r_squared': r_squared}):
    model = keras.models.load_model('model3342.h5') 
y = model.predict(X) 
num_features = 36
test_predict_extended = np.zeros((len(y), num_features))
test_predict_extended[:, -1] = y.ravel() 
test_predict_inversed = scaler1.inverse_transform(test_predict_extended)[:, -1]
data = pd.read_excel('data/test.xlsx')
data['score\n( %)'] = test_predict_inversed
data.to_excel('data/result.xlsx', index=False)