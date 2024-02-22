import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel('data/test.xlsx')
print(df.head())
training_x = df['sequence']
training_y = df['score\n( %)']
# 将碱基序列转换为对应的数值表示（例如使用字典映射）
base_to_index = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
# # 转换数据为模型可接受的格式
indexed_data = []
for sample, efficiency in zip(training_x, training_y):
    indexed_sequence = [base_to_index[base] for base in sample]
    indexed_sequence.append(efficiency)
    indexed_data.append(indexed_sequence)
df = pd.DataFrame(indexed_data)
df.to_csv('data/test.csv')