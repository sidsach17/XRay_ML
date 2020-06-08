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


model_path = 'outputs/'
################### Azure ML Service Starts #############################
parser = argparse.ArgumentParser()
parser.add_argument('--PreProcessingData', dest='PreProcessingData', required=True)
parser.add_argument('--epochs', type=int, dest='epochs', help='No. of epochs', default=2)
parser.add_argument('--batch_size', type=int, dest='batch_size', help='Batch size', default =32)
parser.add_argument('--learning_rate', type=float, dest='learning_rate', help='learning_rate', default =0.001)
args = parser.parse_args()

#args = parser.parse_args()

PreProcessing_OutputPath=args.PreProcessingData
epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.learning_rate


with open(os.path.join(PreProcessing_OutputPath, 'PreprocessingData.json')) as json_file:
    data = json.load(json_file)
    w_minus = data['w_minus']
    n_by_p = data['n_by_p']
    td_shape = data['td_shape']
    vd_shape = data['vd_shape']

with open(os.path.join(PreProcessing_OutputPath, 'training_generator.pkl'), 'rb') as input_train:
    train_generator = pickle.load(input_train)    

training_generator=(obj for obj in train_generator)
    
with open(os.path.join(PreProcessing_OutputPath, 'validation_generator.pkl'), 'rb') as input_valid:
    valid_generator = pickle.load(input_valid) 
    
    
validation_generator=(obj for obj in valid_generator)



################## Azure ML Service Ends ################################

print(w_minus)
print(type(w_minus))
print(n_by_p)
print(type(n_by_p))
print(td_shape)
print(type(td_shape))
print(vd_shape)
print(type(vd_shape))
#print(training_generator)
print(type(training_generator))
#print(validation_generator)
print(type(validation_generator))

##############################Building Model############################################################
from sklearn.metrics import accuracy_score,roc_auc_score
from keras.optimizers import Adam

print('Building Model...')
#with tf.device('/cpu:0'):
model=Sequential()
model.add(DenseNet121(weights='imagenet',include_top=False,input_shape=(224,224,3)))#,dropout_rate=0.5))
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.5))
model.add(Dense(14,activation='sigmoid'))# adds last layer to have 14 neurons [[14 ,kernel_regularizer=regularizers.l2(0.01))]]
model.summary()
parallel_model = model
#parallel_model = multi_gpu_model(model, gpus=2) # model has been built i.e the structure of CNN model has been built



def custom_loss(targets, output):
    _epsilon = tf.convert_to_tensor(K.epsilon(),output.dtype.base_dtype)#, output.dtype.base_dtype
    output = tf.clip_by_value(tf.cast(output,tf.float32), _epsilon, 1 - _epsilon)
    output = tf.math.log(output / (1 - output))
    weight=tf.convert_to_tensor(w_minus,dtype='float32')
    return K.mean(weight*tf.nn.weighted_cross_entropy_with_logits(logits=output, labels=targets, pos_weight=np.array(n_by_p), name=None),axis=-1)
    

auc_roc = tf.keras.metrics.AUC(
    num_thresholds=200, curve='ROC', summation_method='interpolation', name=None,
    dtype=None, thresholds=None)

optimizer=Adam(learning_rate,decay=1e-5)#0.001

# tell the model what loss function and what optimization method to use
parallel_model.compile(loss=custom_loss,optimizer=optimizer,metrics=[auc_roc])

####################Azure ML Service Specific Code###################################
#start an Azure ML run

run = Run.get_context()
####################Azure ML Service Specific Code###################################



######################################## Model Building Ends here ##########################################

####################################### Model Training & Validation Starts ##############################################
from keras.callbacks import ReduceLROnPlateau
            
class Histories(keras.callbacks.Callback):
    def __init__(self,model):
        self.model_for_saving = model
        self.monitor_op = np.less
        self.best = np.Inf
        
    def on_train_begin(self, logs={}):
        self.aucs = []
        self.losses = []
        print("training started")

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        #run.log('Loss',logs['loss'])
        current=logs.get('val_loss')
        ####################Azure ML Service Specific Code###################################
        run.log('train_Loss',logs['loss'])
        run.log('val_Loss',logs['val_loss'])
        run.log('train_auc',logs['auc'])
        run.log('val_auc',logs['val_auc'])
        ####################Azure ML Service Specific Code###################################
        
        
        if self.monitor_op(current, self.best):
            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                  ' saving model'
                  % (epoch + 1, 'val_loss', self.best,
                     current))
                     
            self.best = current
            self.model_for_saving.save(os.path.join(model_path,'weights.best.dense_generator_callback.hdf5'))
            #run.parent.upload_file(os.path.join(model_path,'weights.best.dense_generator_callback.hdf5'))
            print('saved model')
            
        else:
            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, 'val_loss', self.best))
        
        
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

auc=Histories(model=model)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=5, mode='min',min_lr=.00001,cooldown=1)
monitor = EarlyStopping(monitor='val_loss', patience=8, verbose=1, mode='auto')

def train():
    #run = Run.get_context()
    import keras.backend as k
    history=parallel_model.fit_generator(training_generator,
                        steps_per_epoch=td_shape // batch_size, epochs=epochs,
                       validation_data=validation_generator,
                        validation_steps=vd_shape // batch_size,
                         verbose=1,
                        workers=16,
                        use_multiprocessing=True,
                        callbacks=[auc,reduce_lr,monitor]
                       )#
    pd.DataFrame(history.history).to_csv(os.path.join(model_path,'history.csv'),index=False)
    df = pd.read_csv(os.path.join(model_path,'history.csv'))
    run.parent.upload_file(name = 'outputs/weights.best.dense_generator_callback.hdf5', 
                                   path_or_stream =os.path.join(model_path,'weights.best.dense_generator_callback.hdf5'))
    
    
    ####################Azure ML Service Specific Code###################################
    
    fig1 = plt.figure()
    plt.plot(df['loss'])
    plt.title('training loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    run.log_image('Epochs vs Training Loss',plot=fig1) 

    df1 = pd.read_csv(os.path.join(model_path,'history.csv'))
    fig2 = plt.figure()
    plt.plot(df1['val_loss'])
    plt.title('validation loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    run.log_image('Epochs vs Validation Loss',plot=fig2)
    
    df2 = pd.read_csv(os.path.join(model_path,'history.csv'))
    fig3 = plt.figure()
    plt.plot(df2['auc'])
    plt.title('Training AUC')
    plt.ylabel('AUC')
    plt.xlabel('epochs')
    run.log_image('Epochs vs Training AUC',plot=fig3)
    
    df3 = pd.read_csv(os.path.join(model_path,'history.csv'))
    fig4 = plt.figure()
    plt.plot(df3['val_auc'])
    plt.title('Validation AUC')
    plt.ylabel('AUC')
    plt.xlabel('epochs')
    run.log_image('Epochs vs Validation AUC',plot=fig4)
    ####################Azure ML Service Specific Code###################################
    

if __name__ == '__main__':
    
    train()
    # A zero exit code causes the job to be marked a Succeeded.
    sys.exit(0)