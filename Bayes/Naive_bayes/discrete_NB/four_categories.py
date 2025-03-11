import pandas as pd
import numpy as np
from collections import defaultdict
# import ma
# import heapq
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
    LOCA = (sum([1 for l in labels if l == 'LOCA']) + 1)/ float(len(labels)+4)
    MSLB = (sum([1 for l in labels if l == 'MSLB']) + 1)/ float(len(labels)+4)
    NORM = (sum([1 for l in labels if l == 'NORM']) + 1)/ float(len(labels)+4)
    SGTR = (sum([1 for l in labels if l == 'SGTR']) + 1)/ float(len(labels)+4)
    NBClassify = {'LOCA':{},'MSLB':{},'NORM':{},'SGTR':{}}
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
    return LOCA,MSLB,NORM,SGTR,NBClassify
def testNB(data,LOCA1,MSLB1,NORM1,SGTR1,NBClassify1):
    print(LOCA1, MSLB1, NORM1, SGTR1)
    predict_vec = list()
    m = 0
    for sample in data:
        m += 1
        LOCA = np.math.log(LOCA1,2)
        MSLB = np.math.log(MSLB1,2)
        NORM = np.math.log(NORM1,2)
        SGTR = np.math.log(SGTR1,2)
        for label in NBClassify1.keys():
            for k,tag in enumerate(sample):
                k += 1
                if label == 'LOCA':
                    LOCA += np.math.log(NBClassify1[label][k][tag],2)
                    # LOCA = LOCA * NBClassify[label][k][tag]
                if label == 'MSLB':
                    MSLB += np.math.log(NBClassify1[label][k][tag],2)
                    # MSLB = MSLB * NBClassify[label][k][tag]
                if label == 'NORM':
                    NORM += np.math.log(NBClassify1[label][k][tag],2)
                    # NORM = NORM * NBClassify[label][k][tag]
                if label == 'SGTR':
                    SGTR += np.math.log(NBClassify1[label][k][tag],2)
                    # SGTR = SGTR * NBClassify[label][k][tag]
        # number2 = heapq.nlargest(2,[LOCA,MSLB,NORM,SGTR])
        # print(number2)
        if LOCA == max(LOCA,MSLB,NORM,SGTR):
            predict_vec.append('LOCA')
        if MSLB == max(LOCA,MSLB,NORM,SGTR):
            predict_vec.append('MSLB')
        if NORM == max(LOCA,MSLB,NORM,SGTR):
            predict_vec.append('NORM')
        if SGTR == max(LOCA,MSLB,NORM,SGTR):
            predict_vec.append('SGTR')
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
    LOCA1,MSLB1,NORM1,SGTR1,NBClassify1 = trainNB(data)
    # print(LOCA,MSLB,NORM,SGTR)
    predict_vec = testNB(test,LOCA1,MSLB1,NORM1,SGTR1,NBClassify1)
    # print(predict_vec.shape[0])
    count = 0
    for i in range(len(predict_vec)):
        if predict_vec[i] == b[i,0]:
            count += 1
        else:
            print(predict_vec[i],b[i,0])
    accuracy = count/len(predict_vec)
    print(accuracy)