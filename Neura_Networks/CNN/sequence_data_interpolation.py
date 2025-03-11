import pandas as pd
import math
import time
import numpy as np
class Interpolation:
    def __init__(self,path,interpalation_times,time_length):
        self.path = path
        self.data = pd.read_csv(self.path, index_col=0).fillna(axis=0, method='ffill')
        # self.data.drop('12',axis=1,inplace=True)
        self.interpolation_times = interpalation_times
        self.LOCA_data = self.data[0:2000]
        self.MSLB_data = self.data[2000:4000]
        self.SGTR_data = self.data[4000:6000]
        self.NORM_data = self.data[6000:6100]
        self.time_length = time_length
        self.total_label = []
    @staticmethod
    def interpolation(data, time_length):
        data = data.copy()
        data_1 = data[time_length:data.shape[0]] #方便插值，去掉尺寸为0.01的样本
        data_1.index = range(data_1.shape[0])
        data_2 = data[0:(data.shape[0] - time_length)] #方便插值，去掉尺寸为1的样本
        data_2.index = range(data_2.shape[0])
        data_interpolation = (data_1 + data_2) / 2
        data_interpolation = data_interpolation.assign(order = np.arange(1, data_interpolation.shape[0], 2).repeat(time_length)[:data_interpolation.shape[0]]) #添加order方便将数据插入到指定位置
        data = data.assign(order = np.arange(0, data.shape[0], 2).repeat(time_length)[:data.shape[0]])
        data_out = pd.concat([data, data_interpolation], axis=0, ignore_index=True)
        data_out = data_out.assign(sub_order = np.arange(0,data_out.shape[0]))
        # data_out.to_csv('C:/Users/1/Desktop/no.csv')
        data_out.sort_values(['order','sub_order'], ascending=[True, True], inplace=True)
        # data_out.to_csv('C:/Users/1/Desktop/yes.csv')
        data_out.drop(['order','sub_order'], axis=1, inplace=True)
        data_out = data_out.reset_index(drop=True)
        return data_out
    def cyclic_interpolation(self):
        for i in range(self.interpolation_times):
            self.LOCA_data = self.interpolation(self.LOCA_data, self.time_length)
            self.MSLB_data = self.interpolation(self.MSLB_data, self.time_length)
            self.SGTR_data = self.interpolation(self.SGTR_data, self.time_length)
            self.NORM_data = self.interpolation(self.NORM_data, self.time_length)
        total_data = pd.concat([self.LOCA_data, self.MSLB_data, self.SGTR_data, self.NORM_data], axis=0, ignore_index=False)
        total_data = total_data.reset_index(drop=True)
        return total_data
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
        LOCA_num = int(self.LOCA_data.shape[0] / self.time_length)
        MSLB_num = int(self.MSLB_data.shape[0] / self.time_length)
        SGTR_num = int(self.SGTR_data.shape[0] / self.time_length)
        NORM_num = int(self.NORM_data.shape[0] / self.time_length)
        SLOCA = math.ceil(LOCA_num * 0.11)
        MLOCA = math.ceil(LOCA_num * 0.31)
        LLOCA = LOCA_num
        LOCA_label = self.accident_label(SLOCA, MLOCA, LLOCA, 0)
        # self.LOCA_data['label'] = LOCA_label
        SMSLB = math.ceil(MSLB_num * 0.11)
        MMSLB = math.ceil(MSLB_num * 0.26)
        LMSLB = MSLB_num
        MSLB_label = self.accident_label(SMSLB, MMSLB, LMSLB, 3)
        # self.MSLB_data['label'] = MSLB_label
        SSGTR = math.ceil(SGTR_num * 0.14)
        MSGTR = math.ceil(SGTR_num * 0.32)
        LSGTR = SGTR_num
        SGTR_label = self.accident_label(SSGTR, MSGTR, LSGTR, 6)
        # self.SGTR_data['label'] = SGTR_label
        #添加正常工况标签
        NORM_label = []
        for i in range(NORM_num):
            NORM_label.append('9')
        # self.NORM_data['label'] = NORM_label
        self.total_label = LOCA_label + MSLB_label + SGTR_label + NORM_label
        self.total_label = pd.DataFrame(self.total_label)
        return self.total_label
if __name__ == "__main__":
    time_start = time.time()
    # if os.path.exists('D:\\deeplearning\\dataset3'):
    #     shutil.rmtree('D:\\deeplearning\\dataset3')
    # os.mkdir('D:\\deeplearning\\dataset3')
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
    # for time_dataset in os.listdir('D:\\deeplearning\\dataset1'):
    time_length = 20
    time_dataset_path = 'C:/Users/1/Desktop/dataset' + str(time_length) + '.csv'
    chazhi = Interpolation(time_dataset_path, interpolation_number, time_length)
    total_dataset = chazhi.cyclic_interpolation()
    total_labelset = chazhi.add_label()
    total_dataset_path = 'C:/Users/1/Desktop/total_dataset' + str(time_length)  + str(interpolation_number) + '.csv'
    total_labelset_path = 'C:/Users/1/Desktop/total_labelset' + str(time_length)  + str(interpolation_number) + '.csv'
    total_dataset.to_csv(total_dataset_path)
    total_labelset.to_csv(total_labelset_path)
    time_end = time.time()
    print('总共用时：', time_end - time_start)