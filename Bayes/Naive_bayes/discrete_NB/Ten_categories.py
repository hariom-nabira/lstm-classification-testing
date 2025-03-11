import time
import pandas as pd
import numpy as np
from collections import defaultdict
# import ma
# import heapq
start = time.time()
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.width',1000)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)

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
    SLOCA = 0.002
    MLOCA = 3e-4
    LLOCA = 1e-4

    SMSLB = 0.001
    MMSLB = 6e-4
    LMSLB = 2e-4
    NORM = 0.9903

    SSGTR = 0.0052
    MSGTR = 7.75e-4
    LSGTR = 2.58e-4

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
            # print(NBClassify[label][k])
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
if __name__ == "__main__":
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
    print('错误数：',int(len(predict_vec)-count))
    print('子类错误：',same)
    print('子类错误类型:')
    print(sd1)
    print('大类错误：',different)
    print('大类错误类型:')
    print(sd2)
    accuracy = count/len(predict_vec)
    print('准确率:',accuracy)
    end = time.time()
    print('Running time: %s Seconds'%(end-start))