import time
import pandas as pd
import numpy as np
import os
start = time.time()
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.width',1000)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
train_path = "D:/Bayesian inference/traindata1/train_data.csv"
test_path = "D:/Bayesian inference/testdata1/test_data.csv"
if os.path.exists(train_path):
    os.remove(train_path)
if os.path.exists(test_path):
    os.remove(test_path)
data_path = "D:/Bayesian inference/traindata1/data1/train_set.csv"
df = pd.read_csv(data_path,index_col=0)
# new_df = df.groupby('始发事件')
train_rate = 0.8
Refine = str(input("Have you further subdivided the initial event?(yes/no):"))
if Refine == 'yes':
    # num_tup = np.array([5216,782,261,2639,174,58,1000001,1472,53,13])  # 四类始发事件样本数
    num_tup = np.array([14,33,53,11,25,64,1000000,19,14,67])
    num_train_tup = np.array([(int)(round(i * train_rate)) for i in num_tup])  # round函数对数进行四舍五入处理
    num_test_tup = num_tup - num_train_tup
    # 定义分层抽样的字典，格式为：组名：数据个数
    typicalNDict_train = {'SLOCA': num_train_tup[0],'MLOCA': num_train_tup[1],'LLOCA': num_train_tup[2], 'SMSLB': num_train_tup[3],'MMSLB': num_train_tup[4],'LMSLB': num_train_tup[5], 'NORM': num_train_tup[6],
                          'SSGTR': num_train_tup[7],'MSGTR': num_train_tup[8],'LSGTR': num_train_tup[9]}  # 此处要根据不同的事件类型的总数设置抽样的数据
    typicalNDict_test = {'SLOCA': num_test_tup[0],'MLOCA': num_test_tup[1],'LLOCA': num_test_tup[2], 'SMSLB': num_test_tup[3],'MMSLB': num_test_tup[4],'LMSLB': num_test_tup[5], 'NORM': num_test_tup[6],
                          'SSGTR': num_test_tup[7],'MSGTR': num_test_tup[8],'LSGTR': num_test_tup[9]}  # 此处要根据不同的事件类型的总数设置抽样的数据
    print(typicalNDict_train,typicalNDict_test)
if Refine == 'no':
    num_tup = np.array([100, 100, 1, 100])   # 四类始发事件样本数
    num_train_tup = np.array([(int)(round(i*train_rate)) for i in num_tup])   # round函数对数进行四舍五入处理
    num_test_tup = num_tup - num_train_tup
    # 定义分层抽样的字典，格式为：组名：数据个数
    typicalNDict_train = {'LOCA': num_train_tup[0], 'MSLB': num_train_tup[1], 'NORM': num_train_tup[2], 'SGTR': num_train_tup[3]}  # 此处要根据不同的事件类型的总数设置抽样的数据
    typicalNDict_test = {'LOCA': num_test_tup[0], 'MSLB': num_test_tup[1], 'NORM': num_test_tup[2], 'SGTR': num_test_tup[3]}  # 此处要根据不同的事件类型的总数设置抽样的数据
#定义抽样函数
def typicalsamling(group, typicalNDict):
    name = group.name
    n = typicalNDict[name]
    return group.sample(n=n)
train_data = df.groupby('始发事件').apply(typicalsamling, typicalNDict_train)
# test_data = df.append(train_data).drop_duplicates(keep=False)
# train_data = train_data0.drop(train_data0.columns[0],axis=1,inplace=True)
# mul_index = pd.DataFrame(train_data.index)
# for i in range(mul_index.shape[0]):
#     train_data.rename(columns={mul_index.iloc[i,0]:mul_index.iloc[i,0][1]},inplace=True)
# print(pd.DataFrame(train_data.index))
train_data.drop(columns='始发事件',inplace=True)
train_data.reset_index(inplace=True)
train_data.set_index(train_data.columns[1],inplace=True)
# train_data.set_index(train_data.columns[0])
print(train_data)
# train_data = df.sample(frac=0.8,axis=0)
test_data = df[~df.index.isin(train_data.index)]
print(test_data)
train_data.to_csv(train_path,encoding='utf_8_sig')
# print(train_data)
test_data.to_csv(test_path,encoding='utf_8_sig')
# print(test_data.head(10))
# print(train_data.head(10))
end = time.time()
print('Running time: %s Seconds'%(end-start))