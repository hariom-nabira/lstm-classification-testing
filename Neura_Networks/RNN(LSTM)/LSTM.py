# # # import torch
# # # from torch import nn
# # # from torch.autograd import Variable
# # # import torchvision.datasets as dsets
# # # import torchvision.transforms as transforms
# # # import matplotlib.pyplot as plt
# # #
# # # # Hyper Parameters
# # # EPOCH = 1
# # # BATCH_SIZE = 64
# # # TIME_SIZE = 28
# # # INPUT_SIZE = 28
# # # LR = 0.01
# # # DOWNLOAD_MINST = True
# # #
# # # train_data = dsets.MNIST(root='./mnist/', train=True, transform=transforms.ToTensor(), download=DOWNLOAD_MINST)
# # # train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
# # #
# # # test_data = dsets.MNIST(root='./mnist/', train=False, transform=transforms.ToTensor())
# # # test_x = test_data.data.type(torch.FloatTensor)[:2000]/255  # shape (2000, 28, 28) value in range(0,1)
# # # test_y = test_data.targets.numpy()[:2000]
# # #
# # # class RNN(nn.Module):
# # #     def __init__(self):
# # #         super(RNN, self).__init__()
# # #
# # #         self.rnn = nn.LSTM(
# # #             input_size=INPUT_SIZE,
# # #             hidden_size=64,
# # #             num_layers=1,
# # #             batch_first=True
# # #         )
# # #         self.out = nn.Linear(64,10)
# # #
# # #     def forward(self,x):
# # #         r_out, (h_n, h_c) = self.rnn(x, None)
# # #         out = self.out(r_out[:, -1, :])
# # #         return out
# # #
# # # rnn = RNN()
# # # optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
# # # loss_func = nn.CrossEntropyLoss()
# # #
# # # for epoch in range(EPOCH):
# # #     for step, (b_x, b_y) in enumerate(train_loader):
# # #         b_x = b_x.view(-1, 28, 28)
# # #         output = rnn(b_x)
# # #         loss = loss_func(output, b_y)
# # #         optimizer.zero_grad()
# # #         loss.backward()
# # #         optimizer.step()
# # #
# # #         if step % 50 ==0:
# # #             test_out = rnn(test_x)
# # #             pred_y = torch.max(test_out, 1)[1].data.numpy()
# # #             accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
# # #             print('Epoch:',epoch,'| train loss: %.4f' % loss.data.numpy(),'| test accuracy: %.2f' % accuracy)
# # # test_output = rnn(test_x[:10].view(-1, 28, 28))
# # # pred_y = torch.max(test_output, 1)[1].data.numpy()
# # # print(pred_y, 'pred number')
# # # print(test_y[:10], 'real number')
# #
# # import numpy as np
# # import pandas as pd
# # import torch
# # from sklearn.preprocessing import LabelEncoder,MinMaxScaler
# # from sklearn.model_selection import train_test_split
# # from torch.utils.data import TensorDataset
# # from torch.utils.data import DataLoader
# # from torch import nn
# # import time
# # def slicewindow(data, time_length):
# #     '''
# #     移动滑窗创建数据集
# #     '''
# #     X = []
# #     for i in range(0, len(data) - time_length + 1, time_length):
# #         end = i + time_length
# #         oneX = data[i:end,:]
# #         X.append(oneX)
# #     return np.asarray(X,dtype='float64')
# # def add_label():
# #     '''
# #        直接按比例添加
# #        LOCA:11,31,58
# #        MSLB:11,26,63
# #        SGTR:14,32,54
# #     '''
# #     y = []
# #     # for i in range(11):
# #     #     y.append('0')
# #     # for i in range(31):
# #     #     y.append('1')
# #     # for i in range(58):
# #     #     y.append('2')
# #     # for i in range(11):
# #     #     y.append('3')
# #     # for i in range(26):
# #     #     y.append('4')
# #     # for i in range(63):
# #     #     y.append('5')
# #     # for i in range(14):
# #     #     y.append('6')
# #     # for i in range(32):
# #     #     y.append('7')
# #     # for i in range(54):
# #     #     y.append('8')
# #     # for i in range(5):
# #     #     y.append('9')
# #     for i in range(100):
# #         y.append('0')
# #     for i in range(100):
# #         y.append('1')
# #     for i in range(100):
# #         y.append('2')
# #     for i in range(5):
# #         y.append('3')
# #     return np.asarray(y,dtype='int64')
# #
# # class RNN(nn.Module):
# #     def __init__(self):
# #         super(RNN, self).__init__()
# #         self.rnn = nn.LSTM(
# #             input_size=12,
# #             hidden_size=20,
# #             num_layers=1,
# #             batch_first=True
# #         )
# #         self.out = nn.Linear(20 ,4)
# #     def forward(self,x):
# #         r_out, (h_n, h_c) = self.rnn(x, None)
# #         out = self.out(r_out[:, -1, :])
# #         # out = F.log_softmax(out, dim=1)
# #         return out
# # if __name__ == "__main__":
# #     time_start = time.time()
# #     time_lengh = 50
# #     data_path = 'D:/deeplearning/sequence_data/dataset' + str(time_lengh) + '.csv'
# #     data = pd.read_csv(data_path,index_col=0)
# #     data = pd.DataFrame(data)
# #     # norm_data = data.loc[6000:6099]
# #     # for i in range(19):
# #     #     data = data.append(norm_data)
# #     #归一化
# #     min_max_scaler = MinMaxScaler()
# #     min_max_scaler.fit(data)
# #     data = min_max_scaler.transform(data)
# #     X = slicewindow(data, time_lengh)
# #     y = add_label()
# #     encoder = LabelEncoder()
# #     encoder.fit(y)
# #     y = encoder.transform(y)
# #     X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=6, stratify=y)
# #     X_train, Y_train = torch.FloatTensor(X_train), torch.LongTensor(Y_train)
# #     X_test, Y_test = torch.FloatTensor(X_test), torch.LongTensor(Y_test)
# #     train_dataset = TensorDataset(X_train, Y_train)
# #     test_dataset = TensorDataset(X_test, Y_test)
# #     train_loader = DataLoader(dataset=train_dataset, batch_size=len(X_train), shuffle=True, num_workers=12)
# #     test_loader = DataLoader(dataset=test_dataset, batch_size=len(X_test), shuffle=True, num_workers=12)
# #
# #     rnn = RNN()
# #     optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)
# #     loss_func = nn.CrossEntropyLoss()
# #     accuracy_list = []
# #     for epoch in range(5000):
# #         for step, (b_x, b_y) in enumerate(train_loader):
# #             b_x = b_x.view(-1, time_lengh, 12)
# #             output = rnn(b_x)
# #             loss = loss_func(output, b_y)
# #             loss.requires_grad_()
# #             optimizer.zero_grad()
# #             loss.backward()
# #             optimizer.step()
# #             if step % 10 ==0:
# #                 test_out = rnn(X_test.view(-1, time_lengh, 12))
# #                 pred_y = torch.max(test_out, 1)[1].data
# #                 accuracy = pred_y.eq(Y_test.data.view_as(pred_y)).cpu().sum() / len(Y_test)
# #                 accuracy_list.append(accuracy)
# #                 print('Epoch:',epoch,'| train loss: %.4f' % loss.data.numpy(),'| test accuracy: %.2f' % accuracy)
# #     print('max accuracy:{}'.format(max(accuracy_list)))
# #     test_output = rnn(X_test[:10].view(-1, time_lengh, 12))
# #     pred_y = torch.max(test_output, 1)[1].data.numpy()
# #     print(pred_y, 'pred number')
# #     print(Y_test[:10], 'real number')
# #     time_end = time.time()
# #     print('total cost', time_end - time_start)
# #
# #
# # import torch
# # from torch import nn
# # from torch.autograd import Variable
# # import torchvision.datasets as dsets
# # import torchvision.transforms as transforms
# # import matplotlib.pyplot as plt
# #
# # # Hyper Parameters
# # EPOCH = 1
# # BATCH_SIZE = 64
# # TIME_SIZE = 28
# # INPUT_SIZE = 28
# # LR = 0.01
# # DOWNLOAD_MINST = True
# #
# # train_data = dsets.MNIST(root='./mnist/', train=True, transform=transforms.ToTensor(), download=DOWNLOAD_MINST)
# # train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
# #
# # test_data = dsets.MNIST(root='./mnist/', train=False, transform=transforms.ToTensor())
# # test_x = test_data.data.type(torch.FloatTensor)[:2000]/255  # shape (2000, 28, 28) value in range(0,1)
# # test_y = test_data.targets.numpy()[:2000]
# #
# # class RNN(nn.Module):
# #     def __init__(self):
# #         super(RNN, self).__init__()
# #
# #         self.rnn = nn.LSTM(
# #             input_size=INPUT_SIZE,
# #             hidden_size=64,
# #             num_layers=1,
# #             batch_first=True
# #         )
# #         self.out = nn.Linear(64,10)
# #
# #     def forward(self,x):
# #         r_out, (h_n, h_c) = self.rnn(x, None)
# #         out = self.out(r_out[:, -1, :])
# #         return out
# #
# # rnn = RNN()
# # optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
# # loss_func = nn.CrossEntropyLoss()
# #
# # for epoch in range(EPOCH):
# #     for step, (b_x, b_y) in enumerate(train_loader):
# #         b_x = b_x.view(-1, 28, 28)
# #         output = rnn(b_x)
# #         loss = loss_func(output, b_y)
# #         optimizer.zero_grad()
# #         loss.backward()
# #         optimizer.step()
# #
# #         if step % 50 ==0:
# #             test_out = rnn(test_x)
# #             pred_y = torch.max(test_out, 1)[1].data.numpy()
# #             accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
# #             print('Epoch:',epoch,'| train loss: %.4f' % loss.data.numpy(),'| test accuracy: %.2f' % accuracy)
# # test_output = rnn(test_x[:10].view(-1, 28, 28))
# # pred_y = torch.max(test_output, 1)[1].data.numpy()
# # print(pred_y, 'pred number')
# # print(test_y[:10], 'real number')



# #repository cloned at this point

# import numpy as np
# import pandas as pd
# import torch
# from sklearn.preprocessing import LabelEncoder,MinMaxScaler
# from sklearn.model_selection import train_test_split
# from torch.utils.data import TensorDataset
# from torch.utils.data import DataLoader
# from torch import nn
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

# class RNN(nn.Module):
#     def __init__(self):
#         super(RNN, self).__init__()
#         self.rnn = nn.LSTM(
#             input_size=12,
#             hidden_size=20,
#             num_layers=1,
#             batch_first=True
#         )
#         self.out = nn.Linear(20 ,10)
#     def forward(self,x):
#         r_out, (h_n, h_c) = self.rnn(x, None)
#         out = self.out(r_out[:, -1, :])
#         # out = F.log_softmax(out, dim=1)
#         return out
# if __name__ == "__main__":
#     time_start = time.time()
#     time_lengh = 20
#     interpolation_number = 3
#     total_dataset_path = 'D:/deeplearning/interpolation_data/20s/total_dataset' + str(time_lengh) + str(interpolation_number) + '.csv'
#     total_labelset_path = 'D:/deeplearning/interpolation_data/20s/total_labelset' + str(time_lengh) +  str(interpolation_number) + '.csv'
#     dataset = pd.read_csv(total_dataset_path, index_col=0)
#     labelset = pd.read_csv(total_labelset_path, index_col=0)
#     min_max_scaler = MinMaxScaler()
#     min_max_scaler.fit(dataset)
#     data = min_max_scaler.transform(dataset)
#     X = slicewindow(data, time_lengh)
#     y = np.array(labelset['0'])
#     encoder = LabelEncoder()
#     encoder.fit(y)
#     y = encoder.transform(y)
#     X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=6, stratify=y)
#     X_train, Y_train = torch.FloatTensor(X_train), torch.LongTensor(Y_train)
#     X_test, Y_test = torch.FloatTensor(X_test), torch.LongTensor(Y_test)
#     train_dataset = TensorDataset(X_train, Y_train)
#     test_dataset = TensorDataset(X_test, Y_test)
#     train_loader = DataLoader(dataset=train_dataset, batch_size=len(X_train), shuffle=True, num_workers=12)
#     test_loader = DataLoader(dataset=test_dataset, batch_size=len(X_test), shuffle=True, num_workers=12)

#     rnn = RNN()
#     print(rnn)
#     optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)
#     loss_func = nn.CrossEntropyLoss()
#     accuracy_list = []
#     for epoch in range(5000):
#         for step, (b_x, b_y) in enumerate(train_loader):
#             b_x = b_x.view(-1, time_lengh, 12)
#             output = rnn(b_x)
#             loss = loss_func(output, b_y)
#             loss.requires_grad_()
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             if step % 10 ==0:
#                 test_out = rnn(X_test.view(-1, time_lengh, 12))
#                 pred_y = torch.max(test_out, 1)[1].data
#                 accuracy = pred_y.eq(Y_test.data.view_as(pred_y)).cpu().sum() / len(Y_test)
#                 accuracy_list.append(accuracy)
#                 print('Epoch:',epoch,'| train loss: %.4f' % loss.data.numpy(),'| test accuracy: %.2f' % accuracy)
#     print('max accuracy:{}'.format(max(accuracy_list)))
#     test_output = rnn(X_test[:20].view(-1, time_lengh, 12))
#     pred_y = torch.max(test_output, 1)[1].data.numpy()
#     print(pred_y, 'pred number')
#     print(Y_test[:20], 'real number')
#     time_end = time.time()
#     print('total cost', time_end - time_start)



#first gpt edit

#!/usr/bin/env python3
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import time

def slicewindow(data, time_length):
    """
    Create dataset with a sliding window.
    """
    X = []
    for i in range(0, len(data) - time_length + 1, time_length):
        end = i + time_length
        oneX = data[i:end, :]
        X.append(oneX)
    return np.asarray(X, dtype='float64')

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).init()
        self.rnn = nn.LSTM(
            input_size=12,
            hidden_size=20,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Linear(20, 10)
def forward(self, x):
    r_out, (h_n, h_c) = self.rnn(x, None)
    out = self.out(r_out[:, -1, :])
    # out = F.log_softmax(out, dim=1)
    return out

if __name__ == "__main__":
    time_start = time.time()
    time_length = 20
    interpolation_number = 3

    # Use relative paths for processed data (make sure the preprocessing scripts have run)
    total_dataset_path = './processed_data/total_dataset' + str(time_length) + str(interpolation_number) + '.csv'
    total_labelset_path = './processed_data/total_labelset' + str(time_length) + str(interpolation_number) + '.csv'

    dataset = pd.read_csv(total_dataset_path, index_col=0)
    labelset = pd.read_csv(total_labelset_path, index_col=0)

    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(dataset)
    data = min_max_scaler.transform(dataset)
    X = slicewindow(data, time_length)

    y = np.array(labelset['0'])
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=6, stratify=y)
    X_train, Y_train = torch.FloatTensor(X_train), torch.LongTensor(Y_train)
    X_test, Y_test = torch.FloatTensor(X_test), torch.LongTensor(Y_test)

    train_dataset = TensorDataset(X_train, Y_train)
    test_dataset = TensorDataset(X_test, Y_test)

    # You may wish to adjust num_workers according to your system
    train_loader = DataLoader(dataset=train_dataset, batch_size=len(X_train), 
                            shuffle=True, num_workers=12)
    test_loader = DataLoader(dataset=test_dataset, batch_size=len(X_test), 
                            shuffle=True, num_workers=12)

    rnn = RNN()
    print(rnn)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)
    loss_func = nn.CrossEntropyLoss()
    accuracy_list = []

    for epoch in range(5000):
        for step, (b_x, b_y) in enumerate(train_loader):
            b_x = b_x.view(-1, time_length, 12)
            output = rnn(b_x)
            loss = loss_func(output, b_y)
            loss.requires_grad_()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 10 == 0:
                test_out = rnn(X_test.view(-1, time_length, 12))
                pred_y = torch.max(test_out, 1)[1].data
                accuracy = pred_y.eq(Y_test.data.view_as(pred_y)).cpu().sum() / len(Y_test)
                accuracy_list.append(accuracy)
                print('Epoch:', epoch, '| train loss: %.4f' % loss.data.numpy(), 
                    '| test accuracy: %.2f' % accuracy)
    print('Max accuracy:{}'.format(max(accuracy_list)))
    test_output = rnn(X_test[:20].view(-1, time_length, 12))
    pred_y = torch.max(test_output, 1)[1].data.numpy()
    print(pred_y, 'predicted labels')
    print(Y_test[:20], 'true labels')
    time_end = time.time()
    print('Total time cost:', time_end - time_start)