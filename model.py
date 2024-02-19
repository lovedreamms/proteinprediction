import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

data = pd.read_csv('data/1.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1:].values

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 将numpy数组转换成torch.Tensor
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_units1, hidden_units2, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_units1)
        self.fc2 = nn.Linear(hidden_units1, hidden_units2)
        self.fc3 = nn.Linear(hidden_units2, hidden_units2)
        self.fc4 = nn.Linear(hidden_units2, output_dim)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.relu(self.fc3(out))
        out = self.dropout(out)
        out = self.fc4(out)
        return out

def r_squared(y_true, y_pred):
    residual = y_true - y_pred
    ss_res = torch.sum(residual**2)
    ss_tot = torch.sum((y_true - torch.mean(y_true))**2)
    r2 = 1 - ss_res/ss_tot
    return r2

def train(model, optimizer, criterion, X_train, y_train, epoch, lr_schedule):
    model.train()
    train_losses = []
    test_losses = []
    r2_scores = []
    for e in tqdm(range(epoch)):
        optimizer.param_groups[0]['lr'] = lr_schedule(e)
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = -r_squared(y_train, y_pred) # 将损失函数改为负的 R 平方值
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        if (e+1) % 1 == 0:
            test_loss, r2 = test(model, criterion, X_test, y_test)
            test_losses.append(test_loss)
            r2_scores.append(r2)
    print(np.array(test_losses).max(), np.array(r2_scores).max())

def test(model, criterion, X_test, y_test):
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        loss = criterion(y_pred, y_test)
        r2 = r_squared(y_test, y_pred)
    return loss.item(), r2.item()

# 初始化模型和优化器
input_dim = X_train.shape[1]
hidden_units1 = 128
hidden_units2 = 64
# hidden_units1 = 512
# hidden_units2 = 512
output_dim = 1
model = MLP(input_dim, hidden_units1, hidden_units2, output_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()    

# 定义学习率衰减策略
def lr_schedule(epoch):
    if epoch < 200:
        lr = 0.01
    elif epoch < 400:
        lr = 0.005
    elif epoch < 600:
        lr = 0.001
    else:
        lr = 0.00005
    return lr

# 训练模型
train(model, optimizer, criterion, X_train, y_train, epoch=10000, lr_schedule=lr_schedule)

# 在测试集上评估模型
test_loss, r2 = test(model, criterion, X_test, y_test)
print('测试集误差：', test_loss)
print('R平方值：', r2)
