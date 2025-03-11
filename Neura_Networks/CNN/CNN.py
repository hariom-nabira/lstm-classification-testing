# import numpy as np
# import pandas as pd
# import torch
# from sklearn.preprocessing import LabelEncoder,MinMaxScaler
# from sklearn.model_selection import train_test_split
# from torch.utils.data import TensorDataset
# from torch.utils.data import DataLoader
# from torch import nn
# import torch.nn.functional as F
# import time
# def slicewindow(data, time_length):
#     '''
#     移动滑窗创建数据集
#     '''
#     X = []
#     for i in range(0, len(data) - time_length + 1, time_length):
#         end = i + time_length
#         oneX = data[i:end,:]
#         X.append(oneX)
#     return np.asarray(X,dtype='float64')
# def add_label():
#     '''
#        直接按比例添加
#        LOCA:11,31,58
#        MSLB:11,26,63
#        SGTR:14,32,54
#     '''
#     y = []
#     # for i in range(11):
#     #     y.append('0')
#     # for i in range(31):
#     #     y.append('1')
#     # for i in range(58):
#     #     y.append('2')
#     # for i in range(11):
#     #     y.append('3')
#     # for i in range(26):
#     #     y.append('4')
#     # for i in range(63):
#     #     y.append('5')
#     # for i in range(14):
#     #     y.append('6')
#     # for i in range(32):
#     #     y.append('7')
#     # for i in range(54):
#     #     y.append('8')
#     # for i in range(5):
#     #     y.append('9')
#     for i in range(100):
#         y.append('0')
#     for i in range(100):
#         y.append('1')
#     for i in range(100):
#         y.append('2')
#     for i in range(5):
#         y.append('3')
#     return np.asarray(y,dtype='int64')
#
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 5, kernel_size=(5,3))
#         self.conv2 = nn.Conv2d(5, 10, kernel_size=(4,3))
#         self.pooling = nn.MaxPool2d(2)
#         self.fc1 = nn.Linear(600, 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, 64)
#         self.fc4 = nn.Linear(64,4)
#     def forward(self,x):
#         batch_size = x.size(0)
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.pooling(x))
#         x = F.relu(self.conv2(x))
#         x = x.view(batch_size, -1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = self.fc4(x)
#         return x
#
#
# if __name__ == "__main__":
#     time_start = time.time()
#     time_sequence_length = 50
#     data_path = 'C:/Users/1/Desktop/dataset' + str(time_sequence_length) + '.csv'
#     data = pd.read_csv(data_path,index_col=0)
#     data = pd.DataFrame(data)
#     norm_data = data.loc[6000:6099]
#     # for i in range(19):
#     #     data = data.append(norm_data)
#     #归一化
#     min_max_scaler = MinMaxScaler()
#     min_max_scaler.fit(data)
#     data = min_max_scaler.transform(data)
#     X = slicewindow(data, time_sequence_length)
#     y = add_label()
#     encoder = LabelEncoder()
#     encoder.fit(y)
#     y = encoder.transform(y)
#     X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=6, stratify=y)
#     X_train, Y_train = torch.FloatTensor(X_train), torch.LongTensor(Y_train)
#     X_test, Y_test = torch.FloatTensor(X_test), torch.LongTensor(Y_test)
#     train_dataset = TensorDataset(X_train, Y_train)
#     test_dataset = TensorDataset(X_test, Y_test)
#     train_loader = DataLoader(dataset=train_dataset, batch_size=len(X_train), shuffle=True, num_workers=6)
#     test_loader = DataLoader(dataset=test_dataset, batch_size=len(X_test), shuffle=True, num_workers=6)
#
#     cnn = CNN()
#     print(cnn)
#     device = torch.device("cuda:0"if torch.cuda.is_available() else "cpu")
#     cnn.to(device)
#     optimizer = torch.optim.Adam(cnn.parameters(), lr=0.01)
#     loss_func = nn.CrossEntropyLoss()
#     accuracy_list = []
#     for epoch in range(5000):
#         for step, (b_x, b_y) in enumerate(train_loader):
#             b_x = b_x.view(244,-1, time_sequence_length, 12)
#             b_x , b_y = b_x.to(device), b_y.to(device)
#             output = cnn(b_x)
#             loss = loss_func(output, b_y)
#             loss.requires_grad_()
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             if step % 10 ==0:
#                 X_test = X_test.view(61,-1, time_sequence_length, 12)
#                 X_test = X_test.to(device)
#                 Y_test = Y_test.to(device)
#                 test_out = cnn(X_test)
#                 pred_y = torch.max(test_out, 1)[1].data
#                 accuracy = pred_y.eq(Y_test.data.view_as(pred_y)).cpu().sum() / len(Y_test)
#                 accuracy_list.append(accuracy)
#                 print('Epoch:',epoch,'| train loss: %.4f' % loss.data.numpy(),'| test accuracy: %.2f' % accuracy)
#     print('max accuracy:{}'.format(max(accuracy_list)))
#     test_output = cnn(X_test[:10].view(10,-1, time_sequence_length, 12))
#     pred_y = torch.max(test_output, 1)[1].data.numpy()
#     print(pred_y, 'pred number')
#     print(Y_test[:10], 'real number')
#     time_end = time.time()
#     print('total cost', time_end - time_start)
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
import time


def slicewindow(data, time_length):
    '''
    移动滑窗创建数据集
    '''
    X = []
    for i in range(0, len(data) - time_length + 1, time_length):
        end = i + time_length
        oneX = data[i:end, :]
        X.append(oneX)
    return np.asarray(X, dtype='float64')

'''
50s的卷积神经网络
'''


class CNN50(nn.Module):
    def __init__(self):
        super(CNN50, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=(5, 3))
        self.conv2 = nn.Conv2d(5, 10, kernel_size=(4, 3))
        self.pooling = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(600, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 4)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.pooling(x))
        x = F.relu(self.conv2(x))
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


'''
20s的卷积神经网络
'''
class CNN20(nn.Module):
    def __init__(self):
        super(CNN20, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(5, 10, kernel_size=(3, 3))
        self.pooling = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(320, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2 (x)
        return x
if __name__ == "__main__":
    time_start = time.time()
    time_sequence_length = 20
    interpolation_number = 3
    total_dataset_path = 'C:/Users/1/Desktop/total_dataset' + str(time_sequence_length) + str(interpolation_number) + '.csv'
    total_labelset_path = 'C:/Users/1/Desktop/total_labelset' + str(time_sequence_length) + str(interpolation_number) + '.csv'
    dataset = pd.read_csv(total_dataset_path, index_col=0)
    labelset = pd.read_csv(total_labelset_path, index_col=0)
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(dataset)
    data = min_max_scaler.transform(dataset)
    X = slicewindow(data, time_sequence_length)
    y = np.array(labelset['0'])
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=6, stratify=y)
    X_train, Y_train = torch.FloatTensor(X_train), torch.LongTensor(Y_train)
    print(X_train.size())
    X_test, Y_test = torch.FloatTensor(X_test), torch.LongTensor(Y_test)
    train_dataset = TensorDataset(X_train, Y_train)
    test_dataset = TensorDataset(X_test, Y_test)
    train_loader = DataLoader(dataset=train_dataset, batch_size=len(X_train), shuffle=True, num_workers=6)
    test_loader = DataLoader(dataset=test_dataset, batch_size=len(X_test), shuffle=True, num_workers=6)

    cnn = CNN20()
    print(cnn)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cnn.to(device)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.01)
    loss_func = nn.CrossEntropyLoss()
    accuracy_list = []
    for epoch in range(5000):
        for step, (b_x, b_y) in enumerate(train_loader):
            b_x = b_x.view(len(X_train), -1, time_sequence_length, 12)
            b_x, b_y = b_x.to(device), b_y.to(device)
            output = cnn(b_x)
            loss = loss_func(output, b_y)
            loss.requires_grad_()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 10 == 0:
                X_test = X_test.view(len(X_test), -1, time_sequence_length, 12)
                X_test = X_test.to(device)
                Y_test = Y_test.to(device)
                test_out = cnn(X_test)
                pred_y = torch.max(test_out, 1)[1].data
                accuracy = pred_y.eq(Y_test.data.view_as(pred_y)).cpu().sum() / len(Y_test)
                accuracy_list.append(accuracy)
                print('Epoch:', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
    print('max accuracy:{}'.format(max(accuracy_list)))
    test_output = cnn(X_test[:20].view(20, -1, time_sequence_length, 12))
    pred_y = torch.max(test_output, 1)[1].data.numpy()
    print(pred_y, 'pred number')
    print(Y_test[:10], 'real number')
    time_end = time.time()
    print('total cost', time_end - time_start)
