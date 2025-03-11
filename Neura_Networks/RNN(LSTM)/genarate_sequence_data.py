# #repository clooned at this point

# #数据预处理
# import os
# import shutil
# import numpy as np
# import pandas as pd
# #显示所有列
# pd.set_option('display.max_columns', 1000)
# #显示所有行
# pd.set_option('display.max_rows', 1000)
# #设置value的显示长度为100，默认为50
# pd.set_option('max_colwidth',5000)

# class pre_data:
#     def __init__(self, path, name, time_length, start):
#         self.name = name
#         self.path = path
#         self.final_path = 'D:/deeplearning/Pre_data'
#         self.file_name_list = os.listdir(self.path)
#         self.file_path = ''
#         self.file_count = 0
#         self.time_length = time_length
#         self.start = start
#         self.final_path_children = 'D:/deeplearning/Pre_data' + '/' + self.name
#         if os.path.exists(self.final_path_children):
#             shutil.rmtree(self.final_path_children)
#     def determine_file_order(self):
#         (self.file_name_list).sort(key=(lambda x: int(x[5:-2])))
#         self.file_count = len(self.file_name_list)
#     def merge_file(self):
#         for i in range(0, self.file_count, 2):
#             file_1 = self.file_name_list[i]
#             file_2 = self.file_name_list[i + 1]
#             file_1_2 = str(int(i/2 + 1)) + self.name + ".txt"
#             os.chdir(self.path)
#             with open(file_1,'r') as fa:
#                 with open(file_2,'r') as fb:
#                     with open(file_1_2,'w') as fc:
#                         for line in fa:
#                             fc.write(line.strip('\r\n'))
#                             fc.write(fb.readline())
#     def move_file(self):
#         os.chdir(self.final_path)
#         os.mkdir(self.name)
#         self.final_path = 'D:/deeplearning/Pre_data' + '/' + self.name
#         self.file_name_list = os.listdir(self.path)
#         for filename in self.file_name_list:
#             if '.txt' in filename:
#                 self.file_path = self.path + '/' + filename
#                 shutil.move(self.file_path, self.final_path)
#     def data_collation(self):
#         self.file_name_list = os.listdir(self.final_path)
#         (self.file_name_list).sort(key=(lambda x: int(x[:-8])))
#         txt_count = 0
#         time_sequence_value = []
#         for txt in self.file_name_list:
#             txt_count += 1
#             # time_sequence_value = []
#             print(txt)
#             Txt = pd.read_table(self.final_path + '/' + txt ,index_col=False, header=1, error_bad_lines=False)
#             Txt = pd.DataFrame(Txt.iloc[::-1]) #倒序
#             Txt = Txt.dropna(axis=1) #删除空白列
#             col_name = ["时间1", "稳压器压力", "稳压器水位", "上充流量", "SG1给水流量", "SG1出口压力", "SG1出口蒸汽流量", "时间2","主蒸汽母管压力", "安全壳压力", "安全壳温度", "安全壳放射性", "地坑水位", "冷却剂平均温度"]
#             print(Txt.head())
#             Txt.columns = col_name #重命名行
#             Txt = Txt.drop(columns="时间2")#删除重复列
#             Txt.index = range(Txt.shape[0])
#             for time in range(self.time_length):
#                 if time < 10:
#                     time = "00:00:" + '0' + str(time)
#                 else:
#                     time = "00:00:" + str(time)
#                 same_second_value = []
#                 for j in range(1, Txt.shape[1]):
#                     same_second_parameters_value = []
#                     for i in range(Txt.shape[0]):
#                         print(time, Txt.iloc[i,0])
#                         if time in Txt.iloc[i,0]:
#                             same_second_parameters_value.append(Txt.iloc[i,j])
#                     print(same_second_parameters_value, len(same_second_parameters_value))
#                     average_value = np.mean(same_second_parameters_value)
#                     print(average_value)
#                     same_second_value.append(average_value)
#                 time_sequence_value.append(same_second_value)
#         return time_sequence_value
# Type = {'LOCA':0, 'MSLB':100, 'SGTR':200, 'NORM':300}#每种事故的起始点
# basic_path = 'D:/deeplearning/RawData/'
# all_time_sequence_value = []
# for (accident,starting_point) in Type.items():
#     Pre_data = pre_data(basic_path + accident, accident, 50, starting_point)
#     Pre_data.determine_file_order()
#     Pre_data.merge_file()
#     Pre_data.move_file()
#     Time_sequence_value = Pre_data.data_collation()
#     all_time_sequence_value += Time_sequence_value
# all_time_sequence_value = pd.DataFrame(all_time_sequence_value)
# all_time_sequence_value.to_csv('D:/deeplearning/sequence_data/dataset50.csv')


#first gpt edit

#!/usr/bin/env python3
import os
import shutil
import numpy as np
import pandas as pd

#Display all columns and rows, and set maximum column width for display
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)
pd.set_option('max_colwidth', 5000)

class PreData:
    def __init__(self, path, name, time_length, start):
        self.name = name
        self.path = path
        # Use a relative folder for processed data output
        self.final_path = './Pre_data'
        self.file_name_list = os.listdir(self.path)
        self.file_path = ''
        self.file_count = 0
        self.time_length = time_length
        self.start = start
        self.final_path_children = os.path.join(self.final_path, self.name)
        if os.path.exists(self.final_path_children):
            shutil.rmtree(self.final_path_children)
    def determine_file_order(self):
        # Sort file names by extracting an integer from the filename (adjust if needed)
        self.file_name_list = [x for x in self.file_name_list if not x.startswith('.')]

        self.file_name_list.sort(key=lambda x: int(x[5:-2]))
        self.file_count = len(self.file_name_list)
    def merge_file(self):
        for i in range(0, self.file_count, 2):
            file_1 = self.file_name_list[i]
            file_2 = self.file_name_list[i + 1]

            # Construct full path for file_1 and file_2
            file_1_path = os.path.join(self.path, file_1)
            file_2_path = os.path.join(self.path, file_2)
            # Output will be written in the same folder (you might change this if desired)
            output_filename = str(int(i/2 + 1)) + self.name + ".txt"
            output_file_path = os.path.join(self.path, output_filename)
            
            print("Processing merge:", file_1_path, "and", file_2_path, "into", output_file_path)
            
            with open(file_1_path, 'r') as fa:
                with open(file_2_path, 'r') as fb:
                    with open(output_file_path, 'w') as fc:
                        for line in fa:
                            fc.write(line.strip('\r\n'))
                            fc.write('\t')
                            fc.write(fb.readline())

            # file_1_2 = str(int(i/2 + 1)) + self.name + ".txt"
            # print('Hello Guys! Testiing hori hai')
            # print(self.path)
            # print(os.path)
            
            # os.chdir(self.path)
            # with open(file_1, 'r') as fa:
            #     with open(file_2, 'r') as fb:
            #         with open(file_1_2, 'w') as fc:
            #             for line in fa:
            #                 fc.write(line.strip('\r\n'))
            #                 fc.write(fb.readline())
    def move_file(self):
        os.makedirs(self.final_path, exist_ok=True)
        os.makedirs(self.final_path_children, exist_ok=True)
        self.file_name_list = os.listdir(self.path)
        for filename in self.file_name_list:
            if '.txt' in filename:
                self.file_path = os.path.join(self.path, filename)
                shutil.move(self.file_path, self.final_path_children)
    def data_collation(self):
        self.file_name_list = os.listdir(self.final_path_children)
        # Sort by filename assuming the filename starts with a number.
        self.file_name_list.sort(key=lambda x: int(x[:-8]))
        time_sequence_value = []
        for txt in self.file_name_list:
            print("Processing file:", txt)
            # Read the text file; header row is assumed at line index 1; adjust if necessary.
            Txt = pd.read_table(os.path.join(self.final_path_children, txt),
            index_col=False, header=1, on_bad_lines='skip')
            Txt = pd.DataFrame(Txt.iloc[::-1]) # reverse the order
            Txt = Txt.dropna(axis=1) # drop empty columns
            # Define new column names (translated to English)
            col_names = ["Time1", "RegulatorPressure", "RegulatorWaterLevel",
            "UpFlow", "SG1FeedwaterFlow", "SG1OutletPressure",
            "SG1SteamFlow", "Time2", "MainSteamPipePressure",
            "ContainmentPressure", "ContainmentTemperature",
            "ContainmentRadioactivity", "SumpWaterLevel", "AverageCoolantTemperature"]
            print(Txt.head())
            Txt.columns = col_names # rename columns
            Txt = Txt.drop(columns="Time2") # remove duplicate time column
            Txt.index = range(Txt.shape[0])
            for t in range(self.time_length):
                # Create time string in HH:MM:SS format for each second in the window.
                if t < 10:
                    second_str = "00:00:0" + str(t)
                else:
                    second_str = "00:00:" + str(t)
                same_second_value = []
                for j in range(1, Txt.shape[1]):
                    same_second_parameters_value = []
                    for i in range(Txt.shape[0]):
                        # Debug print can be commented out if not needed.
                        #print(second_str, Txt.iloc[i, 0])
                        if second_str in Txt.iloc[i, 0]:
                            same_second_parameters_value.append(Txt.iloc[i, j])
                    #print(same_second_parameters_value, len(same_second_parameters_value))
                    average_value = np.mean(same_second_parameters_value)
                    #print(average_value)
                    same_second_value.append(average_value)
                time_sequence_value.append(same_second_value)
        return time_sequence_value
    
#Accident type starting indices    
Type = {'LOCA': 0, 'MSLB': 100, 'SGTR': 200, 'NORM': 300}
#Use relative path to the raw data folder provided in the repository.
basic_path = os.path.join('.', 'raw_data') + os.sep
print(f'Basic Path: {basic_path}')
all_time_sequence_value = []

for (accident, starting_point) in Type.items():
    pre_data = PreData(os.path.join(basic_path, accident), accident, 50, starting_point)
    pre_data.determine_file_order()
    pre_data.merge_file()
    pre_data.move_file()
    time_sequence_value = pre_data.data_collation()
    all_time_sequence_value += time_sequence_value

all_time_sequence_value = pd.DataFrame(all_time_sequence_value)
#Save the merged time-sequence data to a relative folder (create folder "sequence_data" if needed)
os.makedirs('./sequence_data', exist_ok=True)
all_time_sequence_value.to_csv('./sequence_data/dataset50.csv', index=False)