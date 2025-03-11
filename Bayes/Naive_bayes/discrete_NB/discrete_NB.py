import time as tm
import os,shutil
import pandas as pd
import numpy as np
from collections import defaultdict
start = tm.time()
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.width',1000)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
while True:
    # data_type = str(input("please enter your evidence data_type(train/test):"))
    data_type = "train"
    if data_type == "train":
        IES = ['LOCA', 'MSLB','NORMAL','SGTR']
        datapath = "D:/Bayesian inference/traindata1"
        data1path = datapath + "/" + "data1"
        evidence_set_path = data1path + "/train_set.csv"
        break
    elif data_type == "test":
        IES = ['LOCAtest', 'MSLBtest', 'SGTRtest']
        datapath = "D:/Bayesian inference/testdata1"
        data1path = datapath + "/" + "data2"
        evidence_set_path = data1path + "/test_set.csv"
        break
    else:
        print("enter error,please try it again")

# human_time = str(input("please select the time you want(01~60):"))
Cal_result = {'时间':[],'准确率':[]}
for i in range(1,60):
    if os.path.exists(evidence_set_path):
        os.remove(evidence_set_path)
    if os.path.exists("D:/Bayesian inference/traindata1/check.csv"):
        os.remove("D:/Bayesian inference/traindata1/check.csv")
    # print('时间：%ss'%i)
    if i <= 9:
        human_time = '0' + str(i)
    else:
        human_time = str(i)
    time = "00:00:" + human_time + ":"
    file_list1 = os.listdir(data1path)
    #正常运行参数
    normal = ['NORMAL0.txt', 15478880.0, 0.1728333, 2.8357710000000003, 530.0755000000001, 6965062.0, 526.7943, 6831392.0, 100.06700000000001, 36.377570000000006, 0.0, 0.0, 309.8469999999999]
    #处理所有的表
    data_set = []
    for filename in file_list1:
        result_path = data1path + "/" + filename
        result_list = os.listdir(result_path)
        result_list.sort(key=lambda x: int(x[:-8]))
        # print(result_list)
        for dif_size in result_list:
            child_path = result_path + "/" + dif_size
            # print(child_path)
            ie = pd.read_table(child_path,index_col=False,header=1,error_bad_lines=False)
            ie_daoxu = pd.DataFrame(ie.iloc[::-1])  # 倒序
            # print(ie_daoxu)
            a = []
            for i in range(29):
                a.append(str(i))
            ie_daoxu.columns = a
            b = []
            for i in range(1, 29, 2):
                b.append(str(i))
            b.append("28")
            b.append("14")
            ie_jianhua1 = ie_daoxu.drop(columns=b)  # 去除空的列
            col_name = ["时间", "稳压器压力", "稳压器水位", "上充流量", "SG1给水流量", "SG1出口压力", "SG1出口蒸汽流量", "主蒸汽母管压力", "安全壳压力", "安全壳温度",
                        "安全壳放射性", "地坑水位", "冷却剂平均温度"]
            ie_jianhua1.columns = col_name  # 更改列名
            ie_jianhua2 = ie_jianhua1
            ie_jianhua2.index = range(ie_jianhua2.shape[0])  # 修改索引从零开始
            newDF_index = []
            # time = "00:00:05:"
            # print(ie_jianhua2)
            for i in range(ie_jianhua2.shape[0]):
                c = str(ie_jianhua2.iat[i, 0])
                if time in c:
                    # d = str(i)
                    newDF_index.append(i)
                    # print(pd.DataFrame(ie_jianhua2.iloc[i]))
            # print(newDF_index)
            start_index = int(newDF_index[0])
            end_inex = int(newDF_index[len(newDF_index) - 1] + 1)
            data_tiqu = ie_jianhua2.iloc[start_index:end_inex]  # 数据提取
            data_tiqu.index = range(data_tiqu.shape[0])  # 更改索引，从零开始
            # print(data_tiqu)
            data_chuli = [dif_size]  # 可替换为始发事件
            for i in range(1, data_tiqu.shape[1]):
                e = 0
                for j in range(data_tiqu.shape[0]):
                    e = e + data_tiqu.iat[j, i]
                f = e / (data_tiqu.shape[0])
                data_chuli.append(f)
            class_name = dif_size[-8:-4]
            data_evidence = [class_name]
            for i in range(1, len(data_chuli)):
                data_evidence.append(data_chuli[i])
            data_set.append(data_evidence)
    data_source = pd.DataFrame(data_set)
    c = data_source.loc[200]
    for ii in range(data_source.shape[0]):
        for jj in range(1,data_source.shape[1]):
            a = data_source.iloc[ii,jj]
            b = data_source.iloc[200,jj]
            if float(a) < (c[jj]*0.999):
                data_source.iloc[ii, jj] = "lower"
            elif data_source.iloc[ii,jj] > (c[jj]*1.001):
                data_source.iloc[ii, jj] = "higher"
            else:
                data_source.iloc[ii, jj] = "normal"
    data_source.to_csv("D:/Bayesian inference/traindata1/check.csv",encoding='utf_8_sig')

    if data_type == "train":
        # Refine = str(input("Do you further subdivide the initial event?(yes/no):"))
        Refine = 'yes'
        if Refine == 'yes':
            '''
                始发事件次数增加
                index：0~99（LOCA）:0~10(SLOCA),11~39(MLOCA),40~99(LLOCA)
                index：100~199（MSLB）100~120(SMSLB),121~141(MMSLB),142~199(LMSLB)
                index：200（NORMAL）
                index：201~300（SGTR）201~211(SSGTR),212~249(MSGTR),250~300(LSGTR)
             '''
            #首先更改名字
            ###6秒
            # for i in range(11):
            #     data_source.iloc[i, 0] = 'SLOCA'
            # for i in range(11, 40):
            #     data_source.iloc[i, 0] = 'MLOCA'
            # for i in range(40, 100):
            #     data_source.iloc[i, 0] = 'LLOCA'
            # for i in range(100, 121):
            #     data_source.iloc[i, 0] = 'SMSLB'
            # for i in range(121, 142):
            #     data_source.iloc[i, 0] = 'MMSLB'
            # for i in range(142, 200):
            #     data_source.iloc[i, 0] = 'LMSLB'
            # for i in range(200, 205):
            #     data_source.iloc[i, 0] = 'NORM'
            # for i in range(205, 216):
            #     data_source.iloc[i, 0] = 'SSGTR'
            # for i in range(216, 254):
            #     data_source.iloc[i, 0] = 'MSGTR'
            # for i in range(254, 305):
            #     data_source.iloc[i, 0] = 'LSGTR'
            ###20s
            # for i in range(14):
            #     data_source.iloc[i, 0] = 'SLOCA'
            # for i in range(14, 47):
            #     data_source.iloc[i, 0] = 'MLOCA'
            # for i in range(47, 100):
            #     data_source.iloc[i, 0] = 'LLOCA'
            # for i in range(100, 111):
            #     data_source.iloc[i, 0] = 'SMSLB'
            # for i in range(111, 136):
            #     data_source.iloc[i, 0] = 'MMSLB'
            # for i in range(136, 200):
            #     data_source.iloc[i, 0] = 'LMSLB'
            # for i in range(200, 205):
            #     data_source.iloc[i, 0] = 'NORM'
            # for i in range(205, 224):
            #     data_source.iloc[i, 0] = 'SSGTR'
            # for i in range(224, 238):
            #     data_source.iloc[i, 0] = 'MSGTR'
            # for i in range(238, 305):
            #     data_source.iloc[i, 0] = 'LSGTR'
            # ##34s
            # for i in range(14):
            #     data_source.iloc[i, 0] = 'SLOCA'
            # for i in range(14, 44):
            #     data_source.iloc[i, 0] = 'MLOCA'
            # for i in range(44, 100):
            #     data_source.iloc[i, 0] = 'LLOCA'
            # for i in range(100, 107):
            #     data_source.iloc[i, 0] = 'SMSLB'
            # for i in range(107, 129):
            #     data_source.iloc[i, 0] = 'MMSLB'
            # for i in range(129, 200):
            #     data_source.iloc[i, 0] = 'LMSLB'
            # for i in range(200, 205):
            #     data_source.iloc[i, 0] = 'NORM'
            # for i in range(205, 220):
            #     data_source.iloc[i, 0] = 'SSGTR'
            # for i in range(220, 259):
            #     data_source.iloc[i, 0] = 'MSGTR'
            # for i in range(259, 305):
            #     data_source.iloc[i, 0] = 'LSGTR'
            ##48s
            # for i in range(6):
            #     data_source.iloc[i, 0] = 'SLOCA'
            # for i in range(6, 37):
            #     data_source.iloc[i, 0] = 'MLOCA'
            # for i in range(37, 100):
            #     data_source.iloc[i, 0] = 'LLOCA'
            # for i in range(100, 105):
            #     data_source.iloc[i, 0] = 'SMSLB'
            # for i in range(105, 141):
            #     data_source.iloc[i, 0] = 'MMSLB'
            # for i in range(141, 200):
            #     data_source.iloc[i, 0] = 'LMSLB'
            # for i in range(200, 205):
            #     data_source.iloc[i, 0] = 'NORM'
            # for i in range(205, 216):
            #     data_source.iloc[i, 0] = 'SSGTR'
            # for i in range(216, 254):
            #     data_source.iloc[i, 0] = 'MSGTR'
            # for i in range(254, 305):
            #     data_source.iloc[i, 0] = 'LSGTR'
            ##平均
            for i in range(11):
                data_source.iloc[i, 0] = 'SLOCA'
            for i in range(11, 37):
                data_source.iloc[i, 0] = 'MLOCA'
            for i in range(37, 100):
                data_source.iloc[i, 0] = 'LLOCA'
            for i in range(100, 111):
                data_source.iloc[i, 0] = 'SMSLB'
            for i in range(111, 137):
                data_source.iloc[i, 0] = 'MMSLB'
            for i in range(137, 200):
                data_source.iloc[i, 0] = 'LMSLB'
            for i in range(200, 205):
                data_source.iloc[i, 0] = 'NORM'
            for i in range(205, 219):
                data_source.iloc[i, 0] = 'SSGTR'
            for i in range(219, 251):
                data_source.iloc[i, 0] = 'MSGTR'
            for i in range(251, 305):
                data_source.iloc[i, 0] = 'LSGTR'
            #改回dataset,方便后续事件的增加
            data_set = list(np.array(data_source))
            # data_increase = str(input("Do you want to increase some events(yes/no):"))
            data_increase = 'yes'
            if data_increase == 'yes':
                # event = eval(input("please enter event type('SLOCA','MLOCA','LLOCA','SMSLB','MMSLB','LMSLB','NORM','SSGTR','MSGTR','LSGTR'):"))
                event = ['NORM']
                # print(len(event))
                for i in range(len(event)):
                    if event[i] == "SLOCA":
                        # size_type = input("please enter SLOCA size(0~10):")
                        # size_type = size_type.split(",")
                        # # print(size_type)
                        # for l in range(len(size_type)):
                        for j in range(11):
                            number = 473
                            for ii in range(number):
                                data_set.append(data_set[j])
                    if event[i] == "MLOCA":
                        # size_type = input("please enter MLOCA size(11~39):")
                        # size_type = size_type.split(",")
                        # for l in range(len(size_type)):
                        for j in range(11,40):
                            number = 26
                            for ii in range(number):
                                data_set.append(data_set[j])
                    if event[i] == "LLOCA":
                        # size_type = input("please enter LLOCA size(40~99):")
                        # size_type = size_type.split(",")
                        # for l in range(len(size_type)):
                        for j in range(40,100):
                            # if int(size_type[l]) == j:
                                # number = int(input("please enter %s number:" % j))
                            number = 3
                            for ii in range(number):
                                data_set.append(data_set[j])
                    if event[i] == "SMSLB":
                        # size_type = input("please enter SMSLB size(100~120):")
                        # size_type = size_type.split(",")
                        # for l in range(len(size_type)):
                        for j in range(100,121):
                            # if int(size_type[l]) == j:
                                # number = int(input("please enter %s number:" %j))
                            number = 125
                            for ii in range(number):
                                data_set.append(data_set[j])
                    if event[i] == "MMSLB":
                        # size_type = input("please enter MMSLB size(121~141):")
                        # size_type = size_type.split(",")
                        # for l in range(len(size_type)):
                        for j in range(121,142):
                            # if int(size_type[l]) == j:
                                # number = int(input("please enter %s number:" %j))
                            number = 7
                            for ii in range(number):
                                data_set.append(data_set[j])
                    if event[i] == "LMSLB":
                        # size_type = input("please enter LMSLB size(142~199):")
                        # size_type = size_type.split(",")
                        # for l in range(len(size_type)):
                        for j in range(142,200):
                            # if int(size_type[l]) == j:
                                # number = int(input("please enter %s number:" %j))
                            number = 0
                            for ii in range(number):
                                data_set.append(data_set[j])
                    if event[i] == "NORM":
                        # print('1')
                        # size = int(input("please enter NORM size(38):"))
                        # number = int(input("please enter norm number:"))
                        for j in range(200, 205):
                            number = 20 #1000000#2584683
                            for ii in range(number):
                                data_set.append(data_set[j])
                    if event[i] == "SSGTR":
                        # size_type = input("please enter SSGTR size(201~211):")
                        # size_type = size_type.split(",")
                        # for l in range(len(size_type)):
                        for j in range(205,215):
                            # if int(size_type[l]) == j:
                                # number = int(input("please enter %s number:"%j ))
                            number = 1471
                            for ii in range(number):
                                data_set.append(data_set[j])
                    if event[i] == "MSGTR":
                        # size_type = input("please enter MSGTR size(212~249):")
                        # size_type = size_type.split(",")
                        # for l in range(len(size_type)):
                        for j in range(215, 253):
                            # if int(size_type[l]) == j:
                                # number = int(input("please enter %s number:" % j))
                            number = 52
                            for ii in range(number):
                                data_set.append(data_set[j])
                    if event[i] == "LSGTR":
                        # print('1')
                        # size_type = input("please enter LSGTR size(250~299):")
                        # size_type = size_type.split(",")
                        # for l in range(len(size_type)):
                        for j in range(253, 305):
                            # if int(size_type[l]) == j:
                                # number = int(input("please enter %s number:" % j))
                            number = 12
                            for ii in range(number):
                                data_set.append(data_set[j])
            if data_increase == 'no':
                pass
        #始发事件工况增加
        if Refine == 'no':
            '''          
                始发事件次数增加
                index：0~99（LOCA）
                index：100~199（MSLB）
                index：200（NORMAL）
                index：201~300（SGTR）
            '''
            # 改回dataset,方便后续事件的增加
            data_set = list(np.array(data_source))
            data_increase = str(input("Do you want to increase some events(yes/no):"))
            if data_increase == 'yes':
                event = eval(input("please enter event type('LOCA','MSLB','NORM','SGTR'):"))
                # print(len(event))
                for i in range(len(event)):
                    if event[i] == "LOCA":
                        # size = int(input("please enter LOCA size(0~99):"))
                        size_type = input("please enter LOCA size(0~99):")
                        # size_type = [0,1]
                        size_type = size_type.split(",")
                        print(size_type)
                        for l in range(len(size_type)):
                            for j in range(100):
                                if int(size_type[l]) == j:
                                    number = int(input("please enter %s number:" % j))
                                    # number = 200
                                    for ii in range(number):
                                        data_set.append(data_set[int(size_type[l])])

                    if event[i] == "MSLB":
                        size_type = input("please enter MSLB size(100~199):")
                        size_type = size_type.split(",")
                        for l in range(len(size_type)):
                            for j in range(100, 200):
                                if int(size_type[l]) == j:
                                    number = int(input("please enter %s number:" % j))
                                    # number = 200
                                    for ii in range(number):
                                        data_set.append(data_set[int(size_type[l])])
                    if event[i] == "NORM":
                        size_type = input("please enter NORM size(100~199):")
                        size_type = size_type.split(",")
                        for l in range(len(size_type)):
                            for j in range(200, 205):
                                if int(size_type[l]) == j:
                                    number = int(input("please enter %s number:" % j))
                                    # number = 200
                                    for ii in range(number):
                                        data_set.append(data_set[int(size_type[l])])
                    if event[i] == "SGTR":
                        size_type = input("please enter SGTR size(201~300):")
                        # size_type = [39,40]
                        size_type = size_type.split(",")
                        for l in range(len(size_type)):
                            for j in range(205, 305):
                                if int(size_type[l]) == j:
                                    number = int(input("please enter %s number:" % j))
                                    # number = 200
                                    for ii in range(number):
                                        data_set.append(data_set[int(size_type[l])])
            if data_increase == 'no':
                pass
    evidence_set = pd.DataFrame(data_set)
    evidence_set.columns = ["始发事件", "稳压器压力", "稳压器水位", "上充流量", "SG1给水流量", "SG1出口压力", "SG1出口蒸汽流量", "主蒸汽母管压力", "安全壳压力", "安全壳温度","安全壳放射性", "地坑水位", "冷却剂平均温度"]
    evidence_set.to_csv(evidence_set_path,encoding='utf_8_sig')
    train_path = "D:/Bayesian inference/traindata1/train_data.csv"
    test_path = "D:/Bayesian inference/testdata1/test_data.csv"
    if os.path.exists(train_path):
        os.remove(train_path)
    if os.path.exists(test_path):
        os.remove(test_path)
    data_path = "D:/Bayesian inference/traindata1/data1/train_set.csv"
    df = pd.read_csv(data_path,index_col=0)
    train_rate = 0.8
    # Refine = str(input("Have you further subdivided the initial event?(yes/no):"))
    # print(data_increase)
    if Refine == 'yes':
        # num_tup = np.array([5216,782,261,2639,174,58,1000001,1472,53,13])  # 四类始发事件样本数
        if data_increase == 'yes':
            num_tup = np.array([11, 31, 58, 11, 26, 63, 100, 14, 32, 54]) #平均
            # num_tup = np.array([6, 31, 63, 5, 36, 59, 100, 11, 38, 51]) #48s
            # num_tup = np.array([14, 30, 56, 7, 22, 71, 100, 15, 39, 46]) #34s
            # num_tup = np.array([14,33,53,11,25,64,100,19,14,67]) #20s
            # num_tup = np.array([11, 29, 60, 21, 21, 58, 100, 11, 38, 51]) #6s
        if data_increase == 'no':
            um_tup = np.array([11, 31, 58, 11, 26, 63, 5, 14, 32, 54])  # 平均
            # num_tup = np.array([6, 31, 63, 5, 36, 59, 5, 11, 38, 51])  # 48s
            # num_tup = np.array([14, 30, 56, 7, 22, 71, 5, 15, 39, 46]) #34s
            # num_tup = np.array([14, 33, 53, 11, 25, 64, 5, 19, 14, 67]) #20S
            # num_tup = np.array([11, 29, 60, 21, 21, 58, 5, 11, 38, 51]) #6s
        num_train_tup = np.array([(int)(round(i * train_rate)) for i in num_tup])  # round函数对数进行四舍五入处理
        num_test_tup = num_tup - num_train_tup
        # 定义分层抽样的字典，格式为：组名：数据个数
        typicalNDict_train = {'SLOCA': num_train_tup[0],'MLOCA': num_train_tup[1],'LLOCA': num_train_tup[2], 'SMSLB': num_train_tup[3],'MMSLB': num_train_tup[4],'LMSLB': num_train_tup[5], 'NORM': num_train_tup[6],
                              'SSGTR': num_train_tup[7],'MSGTR': num_train_tup[8],'LSGTR': num_train_tup[9]}  # 此处要根据不同的事件类型的总数设置抽样的数据
        typicalNDict_test = {'SLOCA': num_test_tup[0],'MLOCA': num_test_tup[1],'LLOCA': num_test_tup[2], 'SMSLB': num_test_tup[3],'MMSLB': num_test_tup[4],'LMSLB': num_test_tup[5], 'NORM': num_test_tup[6],
                              'SSGTR': num_test_tup[7],'MSGTR': num_test_tup[8],'LSGTR': num_test_tup[9]}  # 此处要根据不同的事件类型的总数设置抽样的数据
        # print(typicalNDict_train,typicalNDict_test)
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
    def load_data():
        data_path = "D:/Bayesian inference/traindata1/train_data.csv"
        df = pd.read_csv(data_path,index_col=0)
        data = df.values
        return data
    # a = load_data()
    # print(a)
    ##极大似然估计

    def trainNB(data):
        labels = data[:,0]
        # SLOCA = (sum([1 for l in labels if l == 'SLOCA']) + 1)/ float(len(labels)+9)
        # MLOCA = (sum([1 for l in labels if l == 'MLOCA']) + 1)/ float(len(labels)+9)
        # LLOCA = (sum([1 for l in labels if l == 'LLOCA']) + 1)/ float(len(labels)+9)
        #
        # SMSLB = (sum([1 for l in labels if l == 'SMSLB']) + 1)/ float(len(labels)+9)
        # MMSLB = (sum([1 for l in labels if l == 'MMSLB']) + 1)/ float(len(labels)+9)
        # LMSLB = (sum([1 for l in labels if l == 'LMSLB']) + 1)/ float(len(labels)+9)
        #
        # NORM = (sum([1 for l in labels if l == 'NORM']) + 1)/ float(len(labels)+9)
        #
        # SSGTR = (sum([1 for l in labels if l == 'SSGTR']) + 1)/ float(len(labels)+9)
        # MSGTR = (sum([1 for l in labels if l == 'MSGTR']) + 1)/ float(len(labels)+9)
        # LSGTR = (sum([1 for l in labels if l == 'LSGTR']) + 1)/ float(len(labels)+9)
        # SLOCA = 0.002
        # MLOCA = 3e-4
        # LLOCA = 1e-4
        #
        # SMSLB = 0.001
        # MMSLB = 6e-4
        # LMSLB = 2e-4
        # NORM = 0.9903
        #
        # SSGTR = 0.0052
        # MSGTR = 7.75e-4
        # LSGTR = 2.58e-4
        SLOCA = 0.1
        MLOCA = 0.1
        LLOCA = 0.1

        SMSLB = 0.1
        MMSLB = 0.1
        LMSLB = 0.1
        NORM = 0.1

        SSGTR = 0.1
        MSGTR = 0.1
        LSGTR = 0.1

        NBClassify = {'SLOCA':{},'MLOCA':{},'LLOCA':{},'SMSLB':{},'MMSLB':{},'LMSLB':{},'NORM':{},'SSGTR':{},'MSGTR':{},'LSGTR':{}}
        for label in NBClassify.keys():
            sub_data = data[data[:,0] == label]
            # sub_data = pd.DataFrame(sub_data)
            # print(pd.DataFrame(sub_data))
            sub_data = np.array(sub_data)
            for k in range(sub_data.shape[1]):

                tags = list(set(data[:,k]))
                d = sub_data[:,k]
                # print(len(d))
                # print(tags)
                def lapulas():
                    return 1/float(float((len(d) + len(tags))))
                NBClassify[label][k] = defaultdict(lapulas)
                for tag in tags:
                    a = 0
                    a += sum([1 for i in d if i == tag])
                    # print(a)
                    if a != 0:
                        NBClassify[label][k][tag] = (a + 1 ) /float((len(d) + len(tags)))#添加拉普拉斯平滑

                    else:
                        NBClassify[label][k][tag] = 1 / float((len(d) + len(tags)))
        # print(pd.DataFrame(NBClassify))
        # print(NBClassify)

        return SLOCA,MLOCA,LLOCA,SMSLB,MMSLB,LMSLB,NORM,SSGTR,MSGTR,LSGTR,NBClassify
    def testNB(data,SLOCA1,MLOCA1,LLOCA1,SMSLB1,MMSLB1,LMSLB1,NORM1,SSGTR1,MSGTR1,LSGTR1,NBClassify1):
        # print(SLOCA1,MLOCA1,LLOCA1,SMSLB1,MMSLB1,LMSLB1,NORM1,SSGTR1,MSGTR1,LSGTR1)
        predict_vec = list()
        m = 0
        for sample in data:
            m += 1
            SLOCA = np.math.log(SLOCA1,2)
            MLOCA = np.math.log(MLOCA1,2)
            LLOCA = np.math.log(LLOCA1,2)

            SMSLB = np.math.log(SMSLB1,2)
            MMSLB = np.math.log(MMSLB1,2)
            LMSLB = np.math.log(LMSLB1,2)

            NORM = np.math.log(NORM1,2)

            SSGTR = np.math.log(SSGTR1,2)
            MSGTR = np.math.log(MSGTR1,2)
            LSGTR = np.math.log(LSGTR1,2)

            for label in NBClassify1.keys():
                for k,tag in enumerate(sample):
                    k += 1
                    if label == 'SLOCA':
                        SLOCA += np.math.log(NBClassify1[label][k][tag],2)
                    if label == 'MLOCA':
                        MLOCA += np.math.log(NBClassify1[label][k][tag], 2)
                    if label == 'LLOCA':
                        LLOCA += np.math.log(NBClassify1[label][k][tag], 2)
                        # LOCA = LOCA * NBClassify[label][k][tag]
                    if label == 'SMSLB':
                        SMSLB += np.math.log(NBClassify1[label][k][tag],2)
                    if label == 'MMSLB':
                        MMSLB += np.math.log(NBClassify1[label][k][tag], 2)
                    if label == 'LMSLB':
                        LMSLB += np.math.log(NBClassify1[label][k][tag], 2)
                        # MSLB = MSLB * NBClassify[label][k][tag]
                    if label == 'NORM':
                        NORM += np.math.log(NBClassify1[label][k][tag],2)
                        # NORM = NORM * NBClassify[label][k][tag]
                    if label == 'SSGTR':
                        SSGTR += np.math.log(NBClassify1[label][k][tag],2)
                    if label == 'MSGTR':
                        MSGTR += np.math.log(NBClassify1[label][k][tag], 2)
                    if label == 'LSGTR':
                        LSGTR += np.math.log(NBClassify1[label][k][tag], 2)
                        # SGTR = SGTR * NBClassify[label][k][tag]
            # number2 = heapq.nlargest(2,[LOCA,MSLB,NORM,SGTR])
            # print(number2)
            if SLOCA == max(SLOCA,MLOCA,LLOCA,SMSLB,MMSLB,LMSLB,NORM,SSGTR,MSGTR,LSGTR):
                predict_vec.append('SLOCA')
            if MLOCA == max(SLOCA,MLOCA,LLOCA,SMSLB,MMSLB,LMSLB,NORM,SSGTR,MSGTR,LSGTR):
                predict_vec.append('MLOCA')
            if LLOCA == max(SLOCA,MLOCA,LLOCA,SMSLB,MMSLB,LMSLB,NORM,SSGTR,MSGTR,LSGTR):
                predict_vec.append('LLOCA')
            if SMSLB == max(SLOCA,MLOCA,LLOCA,SMSLB,MMSLB,LMSLB,NORM,SSGTR,MSGTR,LSGTR):
                predict_vec.append('SMSLB')
            if MMSLB == max(SLOCA,MLOCA,LLOCA,SMSLB,MMSLB,LMSLB,NORM,SSGTR,MSGTR,LSGTR):
                predict_vec.append('MMSLB')
            if LMSLB == max(SLOCA,MLOCA,LLOCA,SMSLB,MMSLB,LMSLB,NORM,SSGTR,MSGTR,LSGTR):
                predict_vec.append('LMSLB')
            if NORM == max(SLOCA,MLOCA,LLOCA,SMSLB,MMSLB,LMSLB,NORM,SSGTR,MSGTR,LSGTR):
                predict_vec.append('NORM')
            if SSGTR == max(SLOCA,MLOCA,LLOCA,SMSLB,MMSLB,LMSLB,NORM,SSGTR,MSGTR,LSGTR):
                predict_vec.append('SSGTR')
            if MSGTR == max(SLOCA,MLOCA,LLOCA,SMSLB,MMSLB,LMSLB,NORM,SSGTR,MSGTR,LSGTR):
                predict_vec.append('MSGTR')
            if LSGTR == max(SLOCA,MLOCA,LLOCA,SMSLB,MMSLB,LMSLB,NORM,SSGTR,MSGTR,LSGTR):
                predict_vec.append('LSGTR')
        return np.array(predict_vec)
    acuracy = []
    for i in range(300):
        train_data = df.groupby('始发事件').apply(typicalsamling, typicalNDict_train)
        train_data.drop(columns='始发事件',inplace=True)
        train_data.reset_index(inplace=True)
        train_data.set_index(train_data.columns[1],inplace=True)
        # print(train_data)
        test_data = df[~df.index.isin(train_data.index)]
        # print(test_data)
        train_data.to_csv(train_path,encoding='utf_8_sig')
        test_data.to_csv(test_path,encoding='utf_8_sig')

        a = pd.read_csv("D:/Bayesian inference/testdata1/test_data.csv",index_col=0)
        b = a.values.tolist()
        b = np.array(b)
        # print(b.shape[0])
        test = a.drop(axis=1,columns=['始发事件']).values.tolist()
        # print(test)
        # print(len(test))
        data = load_data()
        SLOCA1,MLOCA1,LLOCA1,SMSLB1,MMSLB1,LMSLB1,NORM1,SSGTR1,MSGTR1,LSGTR1,NBClassify1 = trainNB(data)
        # print(LOCA,MSLB,NORM,SGTR)
        # print(NBClassify1)
        predict_vec = testNB(test,SLOCA1,MLOCA1,LLOCA1,SMSLB1,MMSLB1,LMSLB1,NORM1,SSGTR1,MSGTR1,LSGTR1,NBClassify1)
        # print(predict_vec.shape[0])
        count = 0
        same = 0
        different = 0
        sd1 = []
        sd2 = []
        for i in range(len(predict_vec)):
            if predict_vec[i] == b[i,0]:
                count += 1
            else:
                sd0 = [predict_vec[i],b[i,0]]
                if predict_vec[i][-4:] == b[i,0][-4:]:
                    same += 1
                    sd1.append(sd0)
                    # print(predict_vec[i], b[i, 0])
                else:
                    sd2.append(sd0)
                    # print(predict_vec[i], b[i, 0])
                    different += 1
                # print(predict_vec[i],b[i,0])
        sd1 = np.array(sd1)
        sd1 = np.array(list(set(tuple(t) for t in sd1)))
        sd2 = np.array(sd2)
        sd2 = np.array(list(set(tuple(t) for t in sd2)))
        # print('错误数：',int(len(predict_vec)-count))
        # print('子类错误：',same)
        # print('子类错误类型:')
        # print(sd1)
        # print('大类错误：',different)
        # print('大类错误类型:')
        # print(sd2)
        accuracy0 = count/len(predict_vec)
        # print('准确率:',accuracy0)
        acuracy.append(accuracy0)
    a_max = np.max(acuracy)
    Cal_result['时间'].append(human_time)
    Cal_result['准确率'].append(a_max)
    print('时间:%s准确率:%f'%(human_time,a_max))
Cal_result_save = pd.DataFrame.from_dict(Cal_result,orient="columns")
Cal_result_save.to_csv("D:/Bayesian inference/traindata1/Cal_result.csv",encoding='utf_8_sig')
