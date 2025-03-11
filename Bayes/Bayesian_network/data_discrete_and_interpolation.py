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
    data_type = str(input("please enter your evidence data_type(train/test):"))
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
human_time = str(input("please select the time you want(01~60):"))
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
# data_source.to_csv("D:/Bayesian inference/traindata1/check.csv",encoding='utf_8_sig')

if data_type == "train":
    Refine = str(input("Do you further subdivide the initial event?(yes/no):"))
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
        # for i in range(205, 215):
        #     data_source.iloc[i, 0] = 'SSGTR'
        # for i in range(215, 253):
        #     data_source.iloc[i, 0] = 'MSGTR'
        # for i in range(253, 305):
        #     data_source.iloc[i, 0] = 'LSGTR'
        ###20s
        for i in range(14):
            data_source.iloc[i, 0] = 'SLOCA'
        for i in range(14, 47):
            data_source.iloc[i, 0] = 'MLOCA'
        for i in range(47, 100):
            data_source.iloc[i, 0] = 'LLOCA'
        for i in range(100, 111):
            data_source.iloc[i, 0] = 'SMSLB'
        for i in range(111, 136):
            data_source.iloc[i, 0] = 'MMSLB'
        for i in range(136, 200):
            data_source.iloc[i, 0] = 'LMSLB'
        for i in range(200, 205):
            data_source.iloc[i, 0] = 'NORM'
        for i in range(205, 224):
            data_source.iloc[i, 0] = 'SSGTR'
        for i in range(224, 238):
            data_source.iloc[i, 0] = 'MSGTR'
        for i in range(238, 305):
            data_source.iloc[i, 0] = 'LSGTR'
        #改回dataset,方便后续事件的增加
        # print(data_source)
        data_set = list(np.array(data_source))
        data_increase = str(input("Do you want to increase some events(yes/no):"))
        if data_increase == 'yes':
            event = eval(input("please enter event type('SLOCA','MLOCA','LLOCA','SMSLB','MMSLB','LMSLB','NORM','SSGTR','MSGTR','LSGTR'):"))
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
                    # size = int(input("please enter NORM size(38):"))
                    # number = int(input("please enter norm number:"))
                    for j in range(200, 205):
                        number = 200000 #1000000#2584683
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
# print(evidence_set.drop(axis=1,columns=['始发事件']).values.tolist())
end = tm.time()
print('Running time: %s Seconds'%(end-start))




#处理单个表
# ie = pd.DataFrame(pd.read_table("D:/Bayesian inference/data/data1/LOCAresult/LOCA0.txt",index_col=False,header=1))
# ie_daoxu = pd.DataFrame(ie.iloc[::-1]) #倒序
# a = []
# for i in range(29):
#     a.append(str(i))
# ie_daoxu.columns = a
# b = []
# for i in range(1,29,2):
#     b.append(str(i))
# b.append("28")
# b.append("14")
# ie_jianhua1 = ie_daoxu.drop(columns=b)#去除空的列
# col_name = ["时间","稳压器压力","稳压器水位","上充流量","SG1给水流量","SG1出口压力","SG1出口蒸汽流量","主蒸汽母管压力","安全壳压力","安全壳温度","安全壳放射性","地坑水位","冷却剂平均温度"]
# ie_jianhua1.columns = col_name #更改列名
# ie_jianhua2 = ie_jianhua1
# ie_jianhua2.index = range(ie_jianhua2.shape[0])#修改索引从零开始
# newDF_index = []
# time = "00:00:05:"
# for i in range(ie_jianhua2.shape[0]):
#     c = str(ie_jianhua2.iat[i,0])
#     if time in c:
#         # d = str(i)
#         newDF_index.append(i)
#         # print(pd.DataFrame(ie_jianhua2.iloc[i]))
# start_index = int(newDF_index[0])
# end_inex = int(newDF_index[len(newDF_index) - 1] + 1)
# data_tiqu = ie_jianhua2.iloc[start_index:end_inex] #数据提取
# data_tiqu.index = range(data_tiqu.shape[0]) #更改索引，从零开始
# print(data_tiqu)
# data_chuli = [time] #可替换为事发事件
# for i in range(1,data_tiqu.shape[1]):
#     e = 0
#     for j in range(data_tiqu.shape[0]):
#         e = e + data_tiqu.iat[j,i]
#     f = e/(data_tiqu.shape[0])
#     data_chuli.append(f)
# print(data_chuli)
# data_evidence = ['loca0']
# for i in range(1,len(data_chuli)):
#     if data_chuli[i] == normal[i]:
#         data_evidence.append("normal")
#     elif data_chuli[i] > normal[i]:
#         data_evidence.append("higher")
#     else:
#         data_evidence.append("lower")
# one_evidence = pd.DataFrame(data_evidence).T
# w = pd.DataFrame(['1','2','3','4','5','6','7','8','9','10','11','12','13']).T
#
# print(all_evidence.append(one_evidence))



# print(ie_jianhua2.iat[0,0])
