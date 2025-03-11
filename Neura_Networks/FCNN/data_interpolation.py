import pandas as pd
import os
import shutil
import math
import time
class Interpolation:
    def __init__(self,path,interpalation_times):
        self.path = path
        self.data = pd.read_csv(self.path, index_col=0).fillna(axis=0, method='ffill')
        self.data.drop('12',axis=1,inplace=True)
        self.interpolation_times = interpalation_times
        self.LOCA_data = self.data[0:100]
        self.MSLB_data = self.data[100:200]
        self.SGTR_data = self.data[200:300]
        self.NORM_data = self.data[300:305]
    @staticmethod
    def interpolation(data):
        data = data.copy()
        data.index = range(data.shape[0])
        data_copy = data[1:data.shape[0]]
        data_copy.loc[data_copy.shape[0] + 1] = 0
        data_copy.index = range(data_copy.shape[0])
        data_copy = (data + data_copy)/2
        data_copy = data_copy[0:data_copy.shape[0]-1]
        data['order'] = [2*i for i in range(data.shape[0])]
        data_copy['order'] = [2*i + 1 for i in range(data_copy.shape[0])]
        data = pd.concat([data, data_copy], axis=0, ignore_index=True)
        data.sort_values('order', inplace=True)
        data.drop('order', axis=1, inplace=True)
        data = data.reset_index(drop=True)
        return data
    def cyclic_interpolation(self):
        for i in range(self.interpolation_times):
            self.LOCA_data = self.interpolation(self.LOCA_data)
            self.MSLB_data = self.interpolation(self.MSLB_data)
            self.SGTR_data = self.interpolation(self.SGTR_data)
            self.NORM_data = self.interpolation(self.NORM_data)
    @staticmethod
    def accident_label(small_number, medium_number, large_number, label_number):
        label = []
        for i in range(small_number):
            label.append(str(label_number))
        for i in range(small_number, medium_number):
            label.append(str(label_number + 1))
        for i in range(medium_number, large_number):
            label.append(str(label_number + 2))
        return label
    def add_label(self):
        '''
        直接按比例添加
        LOCA:11,31,58
        MSLB:11,26,63
        SGTR:14,32,54
        '''
        LOCA_num = self.LOCA_data.shape[0]
        MSLB_num = self.MSLB_data.shape[0]
        SGTR_num = self.SGTR_data.shape[0]
        NORM_num = self.NORM_data.shape[0]
        SLOCA = math.ceil(LOCA_num * 0.11)
        MLOCA = math.ceil(LOCA_num * 0.31)
        LLOCA = LOCA_num
        LOCA_label = self.accident_label(SLOCA, MLOCA, LLOCA, 0)
        self.LOCA_data['label'] = LOCA_label
        SMSLB = math.ceil(MSLB_num * 0.11)
        MMSLB = math.ceil(MSLB_num * 0.26)
        LMSLB = MSLB_num
        MSLB_label = self.accident_label(SMSLB, MMSLB, LMSLB, 3)
        self.MSLB_data['label'] = MSLB_label
        SSGTR = math.ceil(SGTR_num * 0.14)
        MSGTR = math.ceil(SGTR_num * 0.32)
        LSGTR = SGTR_num
        SGTR_label = self.accident_label(SSGTR, MSGTR, LSGTR, 6)
        self.SGTR_data['label'] = SGTR_label
        #添加正常工况标签
        NORM_label = []
        for i in range(NORM_num):
            NORM_label.append('9')
        self.NORM_data['label'] = NORM_label
        self.data = pd.concat([self.LOCA_data, self.MSLB_data, self.SGTR_data, self.NORM_data], axis=0, ignore_index=False)
        self.data = self.data.reset_index(drop=True)
        return self.data
if __name__ == "__main__":
    time_start = time.time()
    if os.path.exists('D:\\deeplearning\\dataset3'):
        shutil.rmtree('D:\\deeplearning\\dataset3')
    os.mkdir('D:\\deeplearning\\dataset3')
    while True:
        interpolation_number = input('请输入插值次数（正整数）：')
        if not (interpolation_number.isdigit()):
            print('输入错误请重新输入！')
        else:
            interpolation_number = int(interpolation_number)
            samples_number = ((2**interpolation_number - 1)*99 + 100)*3 + ((2**interpolation_number - 1)*4 + 5)
            samples_growth = samples_number/305
            print('样本数将达到{}个，增长了{:.2%}'.format(samples_number, samples_growth))
            go_on = input('是否继续(Y/N):')
            if (go_on == 'Y'):
                break
            else:
                continue
    for time_dataset in os.listdir('D:\\deeplearning\\dataset1'):
        time_dataset_path = 'D:\\deeplearning\\dataset1\\' + time_dataset
        chazhi = Interpolation(time_dataset_path, interpolation_number)
        chazhi.cyclic_interpolation()
        chazhi = chazhi.add_label()
        new_time_dataset_path = 'D:\\deeplearning\\dataset3\\' + time_dataset
        chazhi.to_csv(new_time_dataset_path)
        print(time_dataset)
    time_end = time.time()
    print('总共用时：', time_end - time_start)