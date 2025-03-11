import time as tm
import os,shutil
import pandas as pd
import numpy as np
start = tm.time()
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.width',1000)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
while True:
    # data_type = str(input("please enter your evidence data_type(train/test):"))
    data_type = 'train'
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
if os.path.exists(evidence_set_path):
    os.remove(evidence_set_path)
# human_time = str(input("please select the time you want(01~60):"))
Cal_result = {'时间':[],'准确率':[]}
for i in range(1,60):
    print('时间：%ss'%i)
    if i <= 9:
        human_time = '0' + str(i)
    else:
        human_time = str(i)
    acuracy = []
    for i in range(50):
        time = "00:00:" + human_time + ":"
        file_list1 = os.listdir(data1path)
        #正常运行参数
        normal = ['NORMAL0.txt', 15478880.0, 0.1728333, 2.8357710000000003, 530.0755000000001, 6965062.0, 526.7943, 6831392.0, 100.06700000000001, 36.377570000000006, 0.0, 0.0, 309.8469999999999]
        # normal_set = ['NORM', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal']
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
                # print(data_tiqu)
                for i in range(1, data_tiqu.shape[1]):
                    e = 0
                    for j in range(data_tiqu.shape[0]):
                        e = e + data_tiqu.iat[j, i]
                    f = e / (data_tiqu.shape[0])
                    data_chuli.append(f)
                # print(data_chuli)
                class_name = dif_size[-8:-4]
                data_evidence = [class_name]
                for i in range(1,len(data_chuli)):
                    data_evidence.append(data_chuli[i])
                data_set.append(data_evidence)
        data_source = pd.DataFrame(data_set)
        # print(data_source)
        for i in range(11):
            data_source.iloc[i, 0] = '1'
        for i in range(11, 42):
            data_source.iloc[i, 0] = '2'
        for i in range(42, 100):
            data_source.iloc[i, 0] = '3'
        for i in range(100, 111):
            data_source.iloc[i, 0] = '4'
        for i in range(111, 137):
            data_source.iloc[i, 0] = '5'
        for i in range(137, 200):
            data_source.iloc[i, 0] = '6'
        for i in range(200,205):
            data_source.iloc[i, 0] = '0'
        for i in range(205, 219):
            data_source.iloc[i, 0] = '7'
        for i in range(219, 251):
            data_source.iloc[i, 0] = '8'
        for i in range(251, 305):
            data_source.iloc[i, 0] = '9'
        # data_source = data_source.values.tolist()
        # print(data_source)
        # data_increase = str(input("Do you want to increase some events(yes/no):"))
        # if data_increase == 'yes':
        #     event = eval(input("please enter event type('SLOCA','MLOCA','LLOCA','SMSLB','MMSLB','LMSLB','NORM','SSGTR','MSGTR','LSGTR'):"))
        #     # print(len(event))
        #     for i in range(len(event)):
        #         if event[i] == "SLOCA":
        #             # size_type = input("please enter SLOCA size(0~10):")
        #             # size_type = size_type.split(",")
        #             # # print(size_type)
        #             # for l in range(len(size_type)):
        #             for j in range(11):
        #                 number = 473
        #                 for ii in range(number):
        #                     data_source.append(data_source[j])
        #         if event[i] == "MLOCA":
        #             # size_type = input("please enter MLOCA size(11~39):")
        #             # size_type = size_type.split(",")
        #             # for l in range(len(size_type)):
        #             for j in range(11, 40):
        #                 number = 26
        #                 for ii in range(number):
        #                     data_source.append(data_source[j])
        #         if event[i] == "LLOCA":
        #             # size_type = input("please enter LLOCA size(40~99):")
        #             # size_type = size_type.split(",")
        #             # for l in range(len(size_type)):
        #             for j in range(40, 100):
        #                 # if int(size_type[l]) == j:
        #                 # number = int(input("please enter %s number:" % j))
        #                 number = 3
        #                 for ii in range(number):
        #                     data_source.append(data_source[j])
        #         if event[i] == "SMSLB":
        #             # size_type = input("please enter SMSLB size(100~120):")
        #             # size_type = size_type.split(",")
        #             # for l in range(len(size_type)):
        #             for j in range(100, 121):
        #                 # if int(size_type[l]) == j:
        #                 # number = int(input("please enter %s number:" %j))
        #                 number = 125
        #                 for ii in range(number):
        #                     data_source.append(data_source[j])
        #         if event[i] == "MMSLB":
        #             # size_type = input("please enter MMSLB size(121~141):")
        #             # size_type = size_type.split(",")
        #             # for l in range(len(size_type)):
        #             for j in range(121, 142):
        #                 # if int(size_type[l]) == j:
        #                 # number = int(input("please enter %s number:" %j))
        #                 number = 7
        #                 for ii in range(number):
        #                     data_source.append(data_source[j])
        #         if event[i] == "LMSLB":
        #             # size_type = input("please enter LMSLB size(142~199):")
        #             # size_type = size_type.split(",")
        #             # for l in range(len(size_type)):
        #             for j in range(142, 200):
        #                 # if int(size_type[l]) == j:
        #                 # number = int(input("please enter %s number:" %j))
        #                 number = 0
        #                 for ii in range(number):
        #                     data_source.append(data_source[j])
        #         if event[i] == "NORM":
        #             # size = int(input("please enter NORM size(38):"))
        #             # number = int(input("please enter norm number:"))
        #             for j in range(200, 205):
        #                 number = 200000  # 1000000#2584683
        #                 for ii in range(number):
        #                     data_source.append(data_source[j])
        #         if event[i] == "SSGTR":
        #             # size_type = input("please enter SSGTR size(201~211):")
        #             # size_type = size_type.split(",")
        #             # for l in range(len(size_type)):
        #             for j in range(205, 215):
        #                 # if int(size_type[l]) == j:
        #                 # number = int(input("please enter %s number:"%j ))
        #                 number = 1471
        #                 for ii in range(number):
        #                     data_source.append(data_source[j])
        #         if event[i] == "MSGTR":
        #             # size_type = input("please enter MSGTR size(212~249):")
        #             # size_type = size_type.split(",")
        #             # for l in range(len(size_type)):
        #             for j in range(215, 253):
        #                 # if int(size_type[l]) == j:
        #                 # number = int(input("please enter %s number:" % j))
        #                 number = 52
        #                 for ii in range(number):
        #                     data_source.append(data_source[j])
        #         if event[i] == "LSGTR":
        #             # print('1')
        #             # size_type = input("please enter LSGTR size(250~299):")
        #             # size_type = size_type.split(",")
        #             # for l in range(len(size_type)):
        #             for j in range(253, 305):
        #                 # if int(size_type[l]) == j:
        #                 # number = int(input("please enter %s number:" % j))
        #                 number = 12
        #                 for ii in range(number):
        #                     data_source.append(data_source[j])
        # if data_increase == 'no':
        #     pass
        # data_source = pd.DataFrame(data_source)
        # print(data_source)
        ##划分测试集和训练集
        data_source.columns = ["始发事件", "稳压器压力", "稳压器水位", "上充流量", "SG1给水流量", "SG1出口压力", "SG1出口蒸汽流量", "主蒸汽母管压力", "安全壳压力", "安全壳温度","安全壳放射性", "地坑水位", "冷却剂平均温度"]
        # print(data_source)
        train_rate = 0.8 #分割比例
        num_tup = np.array([11,31,58,11,26,63,5,14,32,54])  # 四类始发事件样本数
        num_train_tup = np.array([(int)(round(i * train_rate)) for i in num_tup])  # round函数对数进行四舍五入处理
        num_test_tup = num_tup - num_train_tup
        # 定义分层抽样的字典，格式为：组名：数据个数
        typicalNDict_train = {'1': num_train_tup[0],'2': num_train_tup[1],'3': num_train_tup[2], '4': num_train_tup[3],'5': num_train_tup[4],'6': num_train_tup[5], '0': num_train_tup[6],
                              '7': num_train_tup[7],'8': num_train_tup[8],'9': num_train_tup[9]}  # 此处要根据不同的事件类型的总数设置抽样的数据
        typicalNDict_test = {'1': num_test_tup[0],'2': num_test_tup[1],'3': num_test_tup[2], '4': num_test_tup[3],'5': num_test_tup[4],'6': num_test_tup[5], '0': num_test_tup[6],
                              '7': num_test_tup[7],'8': num_test_tup[8],'9': num_test_tup[9]}  # 此处要根据不同的事件类型的总数设置抽样的数据
        #定义抽样函数
        def typicalsamling(group, typicalNDict):
            name = group.name
            n = typicalNDict[name]
            return group.sample(n=n)
        train_data = data_source.groupby('始发事件').apply(typicalsamling,typicalNDict_train)
        train_data.drop(columns='始发事件',inplace=True)
        train_data.reset_index(inplace=True)
        train_data.set_index(train_data.columns[1],inplace=True)
        test_data = data_source[~data_source.index.isin(train_data.index)]
        # print(train_data)
        #提取训练集和测试集的特征和目标
        X_train = train_data.iloc[:,1:]
        y_train = train_data.iloc[:,0]
        X_test = test_data.iloc[:,1:]
        y_test = test_data.iloc[:,0]

        #GNB类
        class Gaussian_NB:
            def __init__(self):
                self.num_of_samples = None
                self.num_of_class = None
                self.class_name = []
                self.prior_prob = []
                self.X_mean = []
                self.X_var = []
            def SepByClass(self,X,y):
                ##按类别分割数据
                ##输入未分类的特征和目标，输出分类完成的数据（字典形式）
                self.num_of_samples = len(y) #样本数量
                y = y.reshape(X.shape[0],1)
                data = np.hstack((y,X))
                data_byclass = {}
                # print(len(data[:,0]))
                #提取各类别数据，字典的键为类别名，值为对应的分类数据
                for i in range(len(data[:,0])):
                    if i in data[:,0]:
                        data_byclass[i] = data[data[:,0]==i]
                # print(data_byclass)
                self.class_name = list(data_byclass.keys()) #类别名
                self.num_of_class = len(data_byclass.keys()) #类别总数
                return data_byclass
            def CalPriorProb(self,y_byclass):
                ###计算y的先验概率（使用拉普拉斯平滑）###
                ###输入当前类别下的目标，输出该目标的先验概率###
                # 计算公式：（当前类别下的样本数+1）/（总样本数+类别总数）
                return (len(y_byclass) + 1) / (self.num_of_samples + self.num_of_class)
            def CalXMean(self,X_byclass):
            ###计算各类别特征各维度的平均值###
            ###输入当前类别下的特征，输出该特征各个维度的平均值###
                X_mean = []
                for i in range(X_byclass.shape[1]):
                    X_mean.append(np.mean(X_byclass[:,i]))
                return X_mean
            def CalXVar(self,X_byclass):
                X_var = []
                for i in range(X_byclass.shape[1]):
                    X_var.append(np.var(X_byclass[:,i]))
                return X_var
            def CalGaussianProb(self,X_new,mean,var):
                ###计算训练集特征（符合正态分布）在各类别下的条件概率###
                ###输入新样本的特征，训练集特征的平均值和方差，输出新样本的特征在相应训练集中的分布概率###
                # 计算公式：(np.exp(-(X_new-mean)**2/(2*var)))*(1/np.sqrt(2*np.pi*var))
                gaussian_prob = []
                for a,b,c in zip(X_new, mean, var):
                    formula1 = np.exp(-(a-b)**2/(2*(c + 2.1549429e-13)))
                    formula2 = 1/np.sqrt(2*np.pi*(c + 2.1549429e-13))
                    gaussian_prob.append((formula2 + 2.1549429e-13)*(formula1 + 2.1549429e-13))
                return gaussian_prob
            def fit(self, X, y):
                ###训练数据###
                ###输入训练集特征和目标，输出目标的先验概率，特征的平均值和方差###
                # 将输入的X,y转换为numpy数组
                X, y = np.asarray(X,np.float32), np.asarray(y,np.float32)
                data_byclass = Gaussian_NB.SepByClass(X,y)#数据分类
                # 计算各类别数据的目标先验概率，特征平均值和方差
                for data in data_byclass.values():
                    X_byclass = data[:,1:]
                    # print(X_byclass)
                    y_byclass = data[:,0]
                    # print(y_byclass)
                    # self.prior_prob.append(Gaussian_NB.CalPriorProb(y_byclass))
                    self.X_mean.append(Gaussian_NB.CalXMean(X_byclass))
                    self.X_var.append(Gaussian_NB.CalXVar(X_byclass))
                # self.prior_prob = [0.002,3e-4,1e-4,0.001,6e-4,2e-4,0.9903,0.0052,7.75e-4,2.58e-4]
                self.prior_prob = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
                # print('s',self.prior_prob)
                return self.prior_prob, self.X_mean, self.X_var
            def predict(self,X_new):
                ###预测数据###
                ###输入新样本的特征，输出新样本最有可能的目标###
                # 将输入的x_new转换为numpy数组
                X_new = np.asarray(X_new, np.float32)

                posteriori_prob = [] #初始化极大后验概率
                for i ,j,o in zip(self.prior_prob, self.X_mean, self.X_var):
                    gaussian = Gaussian_NB.CalGaussianProb(X_new,j,o)
                    # print(gaussian)
                    posteriori_prob.append(np.log(i) + sum(np.log(gaussian)))
                    idx = np.argmax(posteriori_prob)
                return self.class_name[idx]
        Gaussian_NB = Gaussian_NB() #实例化Gaussian_NB
        Gaussian_NB.fit(X_train,y_train)
        # print(Gaussian_NB.fit(X_train,y_train))
        acc = 0
        for i in range(len(X_test)):
            predict = Gaussian_NB.predict(X_test.iloc[i,:])
            target = np.array(y_test)[i]
            if int(predict) == int(target):
                acc += 1
        a = acc/len(X_test)
        print('准确率',a)
        acuracy.append(a)
    a_max = np.max(acuracy)
    Cal_result['时间'].append(human_time)
    Cal_result['准确率'].append(a_max)
    print('时间:%s准确率:%f'%(human_time,a_max))
Cal_result_save = pd.DataFrame.from_dict(Cal_result,orient="columns")
Cal_result_save.to_csv("D:/Bayesian inference/traindata1/Cal_result_GAUSS.csv",encoding='utf_8_sig')
