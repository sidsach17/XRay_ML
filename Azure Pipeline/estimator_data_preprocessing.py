import pandas as pd
import glob
import numpy as np
import cv2
#from tqdm import tqdm
import keras
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D,BatchNormalization,Activation,LeakyReLU
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.applications.densenet import DenseNet121,DenseNet169
from keras.applications.densenet import preprocess_input
import tensorflow as tf
from keras.utils import multi_gpu_model, Sequence
import os
import sys
import copy
from random import shuffle
from keras import backend as K
import multiprocessing as mp
from azureml.core import Run
import argparse
import matplotlib.pyplot as plt
import keras.backend.tensorflow_backend as tfback
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import json
    

print("Keras version:", keras.__version__)
print("Tensorflow version:", tf.__version__)


weight_path='DenseNet169_'+'weights.best.dense_generator_callback.hdf5'
model_path = 'outputs/'

run=Run.get_context()

################### Azure ML Service Starts #############################

parser = argparse.ArgumentParser()
parser.add_argument('--PreProcessingData', dest='PreProcessingData', required=True)
args = parser.parse_args()

PreProcessing_OutputPath=args.PreProcessingData

################## Azure ML Service Ends ################################

################## Data Loading #########################################


traindata_path=run.input_datasets['traindata_dataset']
data_train = traindata_path.to_pandas_dataframe()

traintarget_path=run.input_datasets['traintarget_dataset']
train_target = traintarget_path.to_pandas_dataframe()

validdata_path=run.input_datasets['validdata_dataset']
data_valid = validdata_path.to_pandas_dataframe()

validtarget_path=run.input_datasets['validtarget_dataset']
valid_target = validtarget_path.to_pandas_dataframe()


########################################################################


#########################Preparing Data for Training - Image Loading and Resizing #######################################################

def load_data_train(image_path):
    blob_name = image_path.replace('xrayml/images/','')
    #print(blob_name)
    dest='/temp/xray_images/'+blob_name
    #print(dest)
    img=cv2.imread(dest)
    img=cv2.resize(img,(256,256))
    return img

def load_data_valid(image_path):
    blob_name = image_path.replace('xrayml/images/','')
    #print(blob_name)
    dest='/temp/xray_images/'+blob_name
    #print(dest)
    img=cv2.imread(dest)
    img=cv2.resize(img,(224,224))
    return img



batch_size=32
percentage=1
#print('Number of epoch {} with weight path {} and percentage {}'.format(epochs,weight_path,percentage))



t_target=train_target.values[:int(len(data_train['Image Index'].values)*percentage)]
v_target=valid_target.values[:int(len(data_valid['Image Index'].values)*percentage)]

t_data=data_train['Image Index'].values[:int(len(data_train['Image Index'].values)*percentage)]
v_data=data_valid['Image Index'].values[:int(len(data_valid['Image Index'].values)*percentage)]


pool = mp.Pool(processes=5)
print("Loading train images")
train_data_image=[pool.apply(load_data_train,args=(path,)) for path in t_data]
print("Loading validation images")
valid_data_image=[pool.apply(load_data_valid,args=(path,)) for path in v_data]


td_shape = t_data.shape[0]
vd_shape = v_data.shape[0]

print("Loading completed")#########################################################
#####################################################################################



class XRAYSequence(Sequence):

    def __init__(self, x_set, y_set, batch_size,training):
        print("Hello1")
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.training=training
        self.shuffle=training
        self.iter=0
        self.on_epoch_end()
        print("init_end")
        

    def __len__(self):
        print("len_starts")
        return int(np.ceil(len(self.x) / float(self.batch_size)))
        print("len_ends")

    def __getitem__(self, index):
        print("getitem starts")
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_x=[]
        if self.training:
            for i in indexes:
                x=np.random.choice(33)
                y=np.random.choice(33)
                img=self.x[i]
                img=img[y:y+224,x:x+224,:]
                n=np.random.uniform(0,1)
                if n < .5:
                    img=cv2.flip(img,0)

                img=preprocess_input(img.astype(float))
                img=np.expand_dims(img,axis=0)
                #print(img.shape)
                batch_x.append(img)
        else:
            for i in indexes:
                img=self.x[i]
                img=preprocess_input(img.astype(float))
                img=np.expand_dims(img,axis=0)
                #print(img.shape)
                batch_x.append(img)
        #batch_x =[self.x[i] for i in indexes] 
        batch_y = [self.y[i] for i in indexes]
        print("getitem ends")
        return np.vstack(batch_x),np.vstack(batch_y)
        
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        print("on_epoch_end_starts")
        self.indexes = np.arange(len(self.x))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        print("on_epoch_end_ends")

print("before training generator")
training_generator = XRAYSequence(train_data_image, t_target, batch_size,True)
print("after training generator")
validation_generator = XRAYSequence(valid_data_image,v_target,batch_size,False)

w_minus=[]
for i in train_target.columns.tolist():
    w_minus.append((train_target[i].sum()/train_target.shape[0]))
w_minus=w_minus #/min(w_minus)
print(w_minus)
n_by_p=[]
for i in train_target.columns.tolist():
    n_by_p.append((train_target.shape[0]-train_target[i].sum())/train_target[i].sum())
n_by_p=n_by_p #/min(n_by_p)
print(n_by_p)


Data={'w_minus':w_minus,'n_by_p':n_by_p,'td_shape':td_shape,'vd_shape':vd_shape}

train_gen=list(training_generator)
valid_gen=list(validation_generator)

os.makedirs(PreProcessing_OutputPath, exist_ok=True)

# Writing to sample.json 
with open(os.path.join(PreProcessing_OutputPath,'PreprocessingData.json'), "w") as outfile: 
    json.dump(Data, outfile) 
    

with open(os.path.join(PreProcessing_OutputPath,'training_generator.pkl'), 'wb') as output_train:
    pickle.dump(train_gen, output_train, pickle.HIGHEST_PROTOCOL)

with open(os.path.join(PreProcessing_OutputPath,'validation_generator.pkl'), 'wb') as output_valid:
    pickle.dump(valid_gen, output_valid, pickle.HIGHEST_PROTOCOL)

########################################### Data Prepared for Training  #######################################################