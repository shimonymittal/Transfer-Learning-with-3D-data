import os
import glob
from random import shuffle
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from math import floor

class DataSetsLoader:
    def __init__(self, num_points=4096 , num_class=3): 
        self.num_points = num_points
        self.num_class = num_class
        self.load_datasets()
          

    def load_datasets(self):
        # os.chdir("/mnt/Z6Vs-project1/")
        os.chdir("/mnt/ogEk-realtime/")
        new_dir = os.getcwd()   
    
#stratified sampling
    def split_train_test_datasets(self, points, labels, split_size):
        idx_label1 = list([(idx) for idx, label in enumerate(labels) if label.lower() == 'b'])
        idx_label2 = list([(idx) for idx, label in enumerate(labels) if label.lower() == 'g'])
        idx_label3 = list([(idx) for idx, label in enumerate(labels) if label.lower() == 'v'])
        
        split_index1 = floor(len(idx_label1) * split_size)
        split_index2 = floor(len(idx_label2) * split_size)
        split_index3 = floor(len(idx_label3) * split_size)
        
        train_idx = idx_label1[:split_index1] + idx_label2[:split_index2] + idx_label3[:split_index3]
        test_idx  = idx_label1[split_index1:] + idx_label2[split_index2:] + idx_label3[split_index3:] 
        shuffle(train_idx)
        shuffle(test_idx)
        train_points = list([points[i] for i in train_idx])
        train_labels = list([labels[i] for i in train_idx])
        test_points = list([points[i] for i in test_idx])
        test_labels = list([labels[i] for i in test_idx])  
        
        return(train_points,test_points,train_labels,test_labels)
    
    def transform_to_tensorflow_dataset(self, split_size=0.80, split_size_val = 0.75):
        points = []
        labels = []
        train_points = []
        train_labels = []
        test_points = []
        test_labels = []
        
        # all_files = os.listdir("/mnt/Z6Vs-project1/")
        all_files = os.listdir("/mnt/ogEk-realtime/")
        data_files = list(filter(lambda file: file.endswith('.txt'), all_files))
        shuffle(data_files)

        for f in data_files: 
            # os.chdir(r"/mnt/Z6Vs-project1/")                       
            os.chdir(r"/mnt/ogEk-realtime/") 
            try:
                data = pd.read_csv(f, sep=" ")
                data = data.fillna(method='ffill').fillna(method='bfill')
                if (len(data)>=self.num_points):
                     #We sample a point cloud file using `.sample()` which performs a unifrom random sampling. Here we sample at 4096 locations
                    points.append(data.sample(self.num_points))  
                    labels.append(f[0])
                    
#Code for adding RGB values to the data
                    # if(labels.lower() == 'b'):
                    #     z = np.zeros(points.shape)
                    #     z = np.concatenate((points, z), axis=2)
                    #     z[:, :, 3]=1
                    # if(labels.lower() == 'g'):
                    #     z = np.zeros(points.shape)
                    #     z = np.concatenate((points, z), axis=2)
                    #     z[:, :, 4]=1
                    # if(labels.lower() == 'v'):
                    #     z = np.zeros(points.shape)
                    #     z = np.concatenate((points, z), axis=2)
                    #     z[:, :, 5]=1
                    # points = z
                                    
            except Exception as e:
                print(e)
        
        train_points, test_points, train_labels, test_labels = self.split_train_test_datasets(points, labels, split_size)
        train_points, val_points, train_labels, val_labels = self.split_train_test_datasets(train_points, train_labels, split_size_val)
        
        return (
            np.array(train_points),
            np.array(test_points),
            np.array(val_points),
            np.array(train_labels),
            np.array(test_labels),         
            np.array(val_labels)
        )       

if __name__ == "__main__":
    datasets_loader = DataSetsLoader()
    # Get the tensorflow compatiable dataset
    datasets_loader.transform_to_tensorflow_dataset()