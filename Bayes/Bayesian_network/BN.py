# -*- coding:utf-8 -*-
import re
from pgmpy.readwrite import BIFReader
import os,shutil
import numpy as np
from pgmpy.inference import VariableElimination
from prettytable import PrettyTable
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.width',1000)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
def main(list):
    a , b = (list[0],list[1]) if list[0] > list[1] else (list[1],list[0])
    for i in range(2,len(list)):
        if list[i] > list[0]:
            b = a
            a = list[i]
        elif list[i] > list[1]:
            b =list[i]
    return a,b

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
    VE_result = []
    LLOCA = []
    LMSLB = []
    LSGTR = []
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
            ##34s
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
        data_source.columns = ["始发事件", "稳压器压力", "稳压器水位", "上充流量", "SG1给水流量", "SG1出口压力", "SG1出口蒸汽流量", "主蒸汽母管压力", "安全壳压力",
                                "安全壳温度", "安全壳放射性", "地坑水位", "冷却剂平均温度"]
        data_source = data_source.drop(data_source[(data_source['始发事件']=="SLOCA")|(data_source['始发事件']=="MLOCA")|(data_source['始发事件']=="SMSLB")|
                                     (data_source['始发事件']=="MMSLB")|(data_source['始发事件']=="NORM")|(data_source['始发事件']=="SSGTR")|(data_source['始发事件']=="MSGTR")].index)
        for i in range(data_source.shape[0]):
            LLOCA.append('no')
            LMSLB.append('no')
            LSGTR.append('no')
            if data_source.iloc[i,0] == 'LLOCA':
                LLOCA[i] = 'yes'
            elif data_source.iloc[i,0] == 'LMSLB':
                LMSLB[i] = 'yes'
            else:
                LSGTR[i] = 'yes'
        data_source['LLOCA'] = LLOCA
        data_source['LMSLB'] = LMSLB
        data_source['LSGTR'] = LSGTR
        # print(data_source)
        reader = BIFReader('threeIE.bif')

        LLOCA_model = reader.get_model()
        LLOCA_infer = VariableElimination(LLOCA_model)
        predict_data = data_source.drop(columns=['始发事件','LLOCA','LMSLB','LSGTR'],axis=1)
        predict_data.columns = ['RP','RWL','UDF','SG1WF','SG1OP','SG1SF','MSHP','CP','CT','CR','PWL','AT']
        node = ['RP','RWL','UDF','SG1WF','SG1OP','SG1SF','MSHP','CP','CT','CR','PWL','AT']
        predict_data = predict_data.reset_index()
        predict_data = predict_data.drop(columns=['index'],axis=1)
        for i in range(predict_data.shape[0]):
            d = np.array(predict_data.loc[i])
            e = d.tolist()
            evd = dict(zip(node,e))
            f = LLOCA_infer.query(variables=['LLOCA', 'LMSLB', 'LSGTR'],evidence=evd, joint=False)
            # print(i)
            g = pd.DataFrame.from_dict(f,orient='index')
            order = list(g.index)
            p = []
            for i in range(g.shape[0]):
                h = (str(g.iloc[i,0])).strip()
                h1 = h.replace(' ','')
                h2 = h1.replace('\n','')
                # if i == 0:
                #     h2 = h2.replace('+------------+--------------+|LLOCA|phi(LLOCA)|+============+==============+|LLOCA(no)|','')
                #     h2 = h2.replace('|+------------+--------------+|LLOCA(yes)|','')
                #     h2 = h2.replace('|+------------+--------------+','')
                #     print(h2)
                # if i == 1:
                #     h2 = h2.replace('+------------+--------------+|L|phi(LLOCA)|+============+==============+|LLOCA(no)|','')
                #     h2 = h2.replace('|+------------+--------------+|LLOCA(yes)|','')
                #     h2 = h2.replace('|+------------+--------------+','')
                #     print(h2)
                # if i == 2:
                #     h2 = h2.replace('+------------+--------------+|LLOCA|phi(LLOCA)|+============+==============+|LLOCA(no)|','')
                #     h2 = h2.replace('|+------------+--------------+|LLOCA(yes)|','')
                #     h2 = h2.replace('|+------------+--------------+','')
                #     print(h2)
                print(h2)
                h3 = re.findall(r"\d+\.?\d*",h2)
                if h3 == []:
                    p.append('0')
                else:
                    p.append(h3[1])
            for i in range(len(p)):
                p[i] = float(p[i])
            j = main(p)
            if j[0] == j[1]:
                VE_result.append('error')
            else:
                suoyin = p.index(max(p))
                VE_result.append(order[suoyin])
        jishu = 0
        for i in range(data_source.shape[0]):
            if data_source.iloc[i,0] == VE_result[i]:
                jishu += 1
        accuracy = jishu/180
        Cal_result['时间'].append(human_time)
        Cal_result['准确率'].append(accuracy)
        print('第%ss准确率%f' %(human_time,accuracy))
        # print(pd.DataFrame(VE_result))
        # y_pred = LLOCA_model.predict(predict_data)
        # res_path = "D:/Bayesian inference/traindata1/data2/" + str(human_time) + "res.csv"
        # y_pred.to_csv(res_path)
Cal_result_save = pd.DataFrame.from_dict(Cal_result,orient="columns")
Cal_result_save.to_csv("D:/Bayesian inference/traindata1/data2/Cal_result1.csv",encoding='utf_8_sig')


