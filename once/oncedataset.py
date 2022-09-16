from abstract.dataset import ADataSet
from abstract.datastruct import ADataStruct
import json
import os
from matplotlib import pyplot as plt
import torch
import numpy as np
import random
from random import sample
import time



class AOnceDataSet(ADataSet):
    def __init__(self,path,data_struct: ADataStruct):
        self.path = path
        self.data_struct = data_struct  
        super().__init__(data_struct)

    def _load(self):
        pass



class OnceSupervisedDataSet(AOnceDataSet): 

    def __init__(self,path, data_struct=ADataStruct):
        self.path = path
        self.data_struct = data_struct
        self.label_period_dict = {'morning': 0, 'noon': 1, 'afternoon': 2,'night':3}
        self.label_weather_dict = {'sunny': 0, 'cloudy': 1, 'rainy': 2}
        # 
        super().__init__(path,data_struct)
    def _load(self):
        dict_json_w,dict_json_p = self.return_jsondict()
        img_path_list = self.find_img()
        num = 1500
        random.seed(0)
        torch.manual_seed(0)
        img_path_list = sample(img_path_list,num)

        train_dict_p,train_dict_w = self.count(img_path_list[:num],dict_json_w,dict_json_p)
        train_class_p = {'morning': train_dict_p[0], 'noon': train_dict_p[1], 'afternoon': train_dict_p[2],'night':train_dict_p[3]}
        train_class_w = {'sunny': train_dict_w[0], 'cloudy': train_dict_w[1], 'rainy': train_dict_w[2]}
        print('period class distribution',train_class_p)
        print('weather class distribution',train_class_w)
        return img_path_list,dict_json_w,dict_json_p

    def load(self):
        img_path_list,dict_json_w,dict_json_p = self._load()

        images = []
        label_weathers = []
        label_periods = []

        for i in range(len(img_path_list)):
            img_path = img_path_list[i]
            img_folder = os.path.split(img_path)[0]
            img_above_folder = os.path.split(img_folder)[0]
            img_above_folder_name = os.path.split(img_above_folder)[1]
            label_weather = dict_json_w[img_above_folder_name]
            label_period = dict_json_p[img_above_folder_name]
            img = plt.imread(img_path)

            images.append(img)
            label_weathers.append(label_weather)
            label_periods.append(label_period)

        images = np.reshape(images, (len(images), images[1].shape[0], images[1].shape[1], images[1].shape[2]))

        images = torch.from_numpy(images)
        label_weathers = torch.tensor(label_weathers)
        label_periods = torch.tensor(label_periods)

        self.data_struct.data['Train']['Input'] = images
        self.data_struct.data['Train']['Output'] = label_weathers,label_periods



        
    def return_jsondict(self):   
        result=[]
        result = self.lm_find_files(self.path,".json",result)
        label_weather_list = []
        label_period_list = []
        json_num_list = []
        for i in range(len(result)):

            with open(result[i],'r',encoding='utf-8') as f:
             jsonload = json.load(f)
            weather = jsonload['meta_info']['weather']
            period = jsonload['meta_info']['period']
            label_period = self.label_period_dict[str(period)]
            label_weather = self.label_weather_dict[str(weather)]
            label_weather_list.append(label_weather)
            label_period_list.append(label_period)
            json_folder = os.path.split(result[i])[0]
            json_above_folder = os.path.split(json_folder)[1]
            json_num_list.append(json_above_folder)

        dict_json_w = dict(zip(json_num_list,label_weather_list))
        dict_json_p = dict(zip(json_num_list,label_period_list)) 

        return dict_json_w,dict_json_p


    def lm_find_files(self,path,target, result): 

        files = os.listdir(path);
        for f in files:
            npath = path + '/' + f
            if(os.path.isfile(npath)):
                if(os.path.splitext(npath)[1] == target):
                    result.append(npath)
            if(os.path.isdir(npath)):
                if (f[0] == '.'):
                    pass
                else:
                    self.lm_find_files(npath, target, result)
        return result

    def is_image_file(self,filename): 
        return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

    def find_img(self):
        g = os.walk(self.path)  
        img_path_list = []
        for path,dir_list,file_list in g:   
            for file_name in file_list:  
                if self.is_image_file(file_name):
                    img_path_list.append(os.path.join(path, file_name))
        return img_path_list

    def count(self,img_path_list,dict_json_w,dict_json_p): 
        label_weather_list = []
        label_period_list = []
        for i in range(len(img_path_list)):
            img_path = img_path_list[i]
            img_folder = os.path.split(img_path)[0]
            img_above_folder = os.path.split(img_folder)[0]
            img_above_folder_name = os.path.split(img_above_folder)[1]

            label_weather = dict_json_w[img_above_folder_name]
            label_period = dict_json_p[img_above_folder_name]


            label_weather_list.append(int(label_weather))
            label_period_list.append(int(label_period))
        dict_p = {}
        for key in label_period_list:
            dict_p[key] = dict_p.get(key, 0) + 1
        dict_w = {}
        for key in label_weather_list:
            dict_w[key] = dict_w.get(key, 0) + 1
        # print(dict)
        return dict_p,dict_w

    def update():
        pass

    def forward(self):
        since = time.time()
        self.load()
        time_elapsed = time.time() - since
        print("Forward compete in {}m {}s".format(time_elapsed // 60, time_elapsed % 60))
        return self.data_struct