#数据集划分
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split
class pro_data:
    def __init__(self, data_path, test_size, seed, train_batch_size, test_batch_size, num_workers):
        self.data_path = data_path
        self.test_size = test_size
        self.seed = seed
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.X_train = []
        self.X_test = []
        self.Y_train = []
        self.Y_test = []
    def data_dividing(self):
        data = pd.read_csv(self.data_path, index_col=0)
        data = data.fillna(axis=0,method='ffill')
        data = data.values
        X = data[:, :-1].astype(float)
        Y = data[:, -1]
        encoder = LabelEncoder()
        encoder.fit(Y)
        Y = encoder.transform(Y)
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=0.2, random_state=self.seed, stratify=Y)
        #归一化
        min_max_scaler = MinMaxScaler()
        min_max_scaler.fit(self.X_train)
        self.X_train = min_max_scaler.transform(self.X_train)
        min_max_scaler.fit(self.X_test)
        self.X_test = min_max_scaler.transform(self.X_test)
    def data_load(self):
        X_train, Y_train = torch.FloatTensor(self.X_train), torch.LongTensor(self.Y_train)
        X_test, Y_test = torch.FloatTensor(self.X_test), torch.LongTensor(self.Y_test)
        train_dataset = TensorDataset(X_train, Y_train)
        test_dataset = TensorDataset(X_test, Y_test)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=self.num_workers)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.test_batch_size, shuffle=True, num_workers=self.num_workers)
        print('长度：{}'.format(len(test_loader.dataset)))
        return train_loader, test_loader
#模型训练
#模型评估
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import shutil
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(12, 15)
        self.fc2 = nn.Linear(15, 10)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x),dim=1)
        return x


def train(epoch):
    model.train()
    total_loss = 0
    print(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        # if torch.cuda.is_available():
        #     data, traget = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step() #更新所有参数
        total_loss += loss.item()
        # print(batch_idx)
        # if batch_idx % 10 == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100.* batch_idx / len(train_loader), loss.item()))
    writer.add_scalars('train/loss', {'train_loss': total_loss/24659}, epoch)
def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    pre_cor_result = []
    sub_accuracy = {}
    with torch.no_grad():
        for data, target in test_loader:
            # if torch.cuda.is_available():
            #     data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1] #只返回最大值的每个索引
            Pred = pred.view_as(target)
            for i, j in enumerate(Pred==target):
                if j:
                    pre_cor_result.append(int(target[i]))
            correct += pred.eq(target.data.view_as(pred)).cpu().sum() #对预测正确的数据个数进行累加
    pre_cor_result = np.asarray(pre_cor_result)
    print(pre_cor_result)
    print(Pred)
    print(target)
    for i in target.unique():
        target = np.asarray(target)
        i = int(i)
        sub_total_number = np.sum(target==i)
        sub_cor_number = np.sum(pre_cor_result==i)
        sub_accuracy[str(i)] = str(100.*sub_cor_number/sub_total_number) + '%' + '(' + str(sub_cor_number) + '/' + str(sub_total_number) + ')'
    print(sub_accuracy)
    test_loss /= len(test_loader.dataset)
    writer.add_scalars('test/loss', {'test_loss':test_loss}, epoch)
    writer.add_scalars('test/accuracy', {'Accuracy':100.* correct / len(test_loader.dataset)}, epoch)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100.* correct / len(test_loader.dataset)))
    return 100.* correct / len(test_loader.dataset),test_loss
if __name__ == '__main__':
    import time
    from torch.utils.tensorboard import SummaryWriter
    import pandas as pd
    import numpy as np
    time_start = time.time()
    dataset_path = 'D:/deeplearning/dataset3'
    accuraccy = []
    # for subdataset in os.listdir(dataset_path):
    subdataset = '10.csv'
    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=0.2, momentum=0.5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.6, mode='min', patience=20, verbose=True)
    visual_path = ('./' + subdataset).replace('.csv','')
    model_path = ('./' + subdataset).replace('csv','pth')
    if os.path.exists(visual_path):
        shutil.rmtree(visual_path)
    writer = SummaryWriter(visual_path)
    subdataset_path = 'D:/deeplearning/dataset3/' + subdataset
    Pro_data = pro_data(subdataset_path, 0.3, 1, 5, 30824, 14)
    Pro_data.data_dividing()
    train_loader,test_loader = Pro_data.data_load()
    Acc = 0
    for epoch in range(1, 1500):
        print(subdataset)
        train(epoch)
        acc, test_loss = test(epoch)
        scheduler.step(test_loss)
        if acc > Acc:
            Acc = acc
            print('save model')
            torch.save(model.state_dict(),model_path)
    print(Acc)
    accuraccy.append(Acc)
    writer.close()
    accuraccy = pd.DataFrame(accuraccy)
    accuraccy.to_csv('./accuracy.csv')
    time_end = time.time()
    print('total coss:{}s'.format(time_end - time_start))