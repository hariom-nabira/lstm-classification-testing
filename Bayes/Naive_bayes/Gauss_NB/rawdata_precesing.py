import os,shutil
if os.path.exists("D:/Bayesian inference/traindata1/data1"):
    os.remove("D:/Bayesian inference/traindata1/data1")
data_type = str(input("please enter your data_type(train/test):"))
if data_type == "train":
    IES = ['LOCA', 'MSLB','NORM','SGTR']
    path0 = "D:/Bayesian inference/traindata1"
    data_name = "data1"
elif data_type == "test":
    IES = ['LOCAtest', 'MSLBtest', 'SGTRtest']
    path0 = "D:/Bayesian inference/testdata1"
    data_name = "data2"
else:
    print("enter error")
for i in IES:
    ie = i
    path = path0 + "/" + i
    ie1 = os.listdir(path)
    # for filename in ie1:
    #     a = filename
    #     if "0." in a:
    #         a = a.replace("0.","0")
    #     os.chdir(path)
    #     os.rename(filename, a)
    count = 0
    ie1.sort(key=lambda x: int(x[5:-2]))
    print(ie1)
    for file in ie1:
        count = count + 1

    for i in range(0,count,2):
        print(i)
        txt1 = ie1[i]
        txt2 = ie1[i + 1]
        result = str(int(i/2+1)) + ie + ".txt"
        os.chdir(path)
        print(txt1,txt2)
        with open(txt1,'r') as fa:
            with open(txt2,'r') as fb:
                with open(result,'w') as fc:
                    for line in fa:
                        fc.write(line.strip('\r\n'))
                        fc.write(fb.readline())      #合并数据
for i in IES:
    path = path0 + "/" + i
    ie1 = os.listdir(path)
    wjm = i + "result"
    os.chdir(path0)
    os.mkdir(wjm)
    mubiaopath = path0 + "/" + wjm
    for filename in ie1:
        if ".txt" in filename:
            # print('1')
            src = os.path.join(path,filename)
            dst = os.path.join(mubiaopath,filename)
            shutil.move(src,dst)
datapath = path0
os.chdir(datapath)
os.mkdir(data_name)
data1path = datapath + "/" + data_name
os.chdir(data1path)
file_list = os.listdir(datapath)
for filename in file_list:
    if "result" in filename:
        data2path = datapath + "/" + filename
        shutil.move(data2path,data1path)
print("第一步处理完毕，请进行下一步（更换数据类型/证据信息处理）")
# file_list1 = os.listdir(data1path)
# for filename in file_list1:
#     result_path = data1path + "/" + filename
#     result_list = os.listdir(result_path)
#     for dif_size in result_list:
#         ie = pd.read_csv(dif_size)
#         print(ie)










