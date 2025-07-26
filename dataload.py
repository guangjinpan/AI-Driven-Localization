import numpy as np
import torch
import os
import math
import torch
# from tqdm import tqdm
from torch.utils.data import ConcatDataset
#import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import random
import numpy as np
import matplotlib.pyplot as plt
from train_model import Wrapper

 
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader, ConcatDataset, random_split, Dataset
from pytorch_lightning.callbacks import ModelCheckpoint
import sys
sys.path.append('../pre_data/')
import h5py



def channel_normalization(H):
    P_current = np.sum(np.abs(H) ** 2)
    # alpha = np.sqrt((H.shape[1] * H.shape[2]) / P_current)
    alpha = np.sqrt((H.shape[0] * H.shape[1]) / P_current)

    # Scale the matrix so that the total power becomes N*M
    H_normalized = H * alpha

    return H_normalized  
 

def freq_to_delay_angle_domain(H_freq):
    H_complex = H_freq[0] + 1j * H_freq[1]

    H_delay = np.fft.ifft(H_complex, axis=0)

    H_delay_angle_complex = np.fft.fft(H_delay, axis=1) / np.sqrt(H_delay.shape[0]) #/ 2

    return H_delay_angle_complex


class generate_Dataset(Dataset):
    def __init__(self, EnvPara):
        self.EnvPara = EnvPara


        if self.EnvPara["Dataset"] == "MaMIMO":
            self.labels = np.load(EnvPara["Label_path"]["MaMIMO"])
            data_idx = np.loadtxt("./dataset/MaMIMO_train.txt")
            data_idx = data_idx.astype(int)
            self.len = len(data_idx)
            self.labels = self.labels[data_idx,:2]/100

            self.data = np.zeros((self.len,64, 100)) + 1j*np.zeros((self.len,64, 100))
            for cnt, i in enumerate(data_idx):
                self.data[cnt,:,:] = np.load(EnvPara["Dataset_path"]["MaMIMO"]+f"channel_measurement_{i:06d}.npy")
            print(self.data.shape)


        elif self.EnvPara["Dataset"] == "DeepMIMO_O1":
            data_idx = np.loadtxt("./dataset/DeepMIMO_train.txt")
            data_idx = data_idx.astype(int)
            self.len = len(data_idx)
            self.labels = np.zeros((self.len,2))
            self.data = np.zeros((self.len, 64, 100)) + 1j*np.zeros((self.len,64, 100))


            for cnt, sample_i in enumerate(data_idx):
                with h5py.File(EnvPara["Dataset_path"]["DeepMIMO_O1"]+f"/3_{sample_i}.h5py", 'r') as f:
                    channel_real = f["channel_real"][:]
                    channel_imag = f["channel_imag"][:]
                    UElocation = f["UElocation"][:][:2]

                    self.labels[cnt,:] = UElocation/20
                    self.data[cnt,:,:] = channel_real + 1j*channel_imag
            print(self.data.shape)
        


        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        

        # if self.EnvPara["Dataset"] == "MaMIMO":
        #     channel_1 = self.data[idx,:,:]
        # elif self.EnvPara["Dataset"] == "DeepMIMO_O1":
        #     channel_1 = self.data[idx,:,:]

        channel_1 = self.data[idx,:,:]
        channel_1 = channel_normalization(channel_1)

        # CFR with amplitude and phase 
        channel_input = np.zeros((2, channel_1.shape[0], channel_1.shape[1]))
        channel_input[0,:,:] = np.abs(channel_1).astype(np.float32)
        channel_input[1,:,:] = np.angle(channel_1).astype(np.float32)



        # labels: 0,1:location, 2:distance(toa), 3:angle (aoa)
        UElocation =self.labels[idx,:]
                
        return  channel_input, UElocation
    


class generate_Dataset_val(Dataset):
    def __init__(self, EnvPara):
        self.EnvPara = EnvPara



        if self.EnvPara["Dataset"] == "MaMIMO":
            self.labels = np.load(EnvPara["Label_path"]["MaMIMO"])
            data_idx = np.loadtxt("./dataset/MaMIMO_val.txt")
            data_idx = data_idx.astype(int)
            self.len = len(data_idx)
            self.labels = self.labels[data_idx,:2]/100

            self.data = np.zeros((self.len, 64, 100)) + 1j*np.zeros((self.len , 64, 100))
            for cnt, i in enumerate(data_idx):
                self.data[cnt,:,:] = np.load(EnvPara["Dataset_path"]["MaMIMO"]+f"channel_measurement_{i:06d}.npy")

        elif self.EnvPara["Dataset"] == "DeepMIMO_O1":
            data_idx = np.loadtxt("./dataset/DeepMIMO_val.txt")
            data_idx = data_idx.astype(int)
            self.len = len(data_idx)
            self.labels = np.zeros((self.len,2))
            self.data = np.zeros((self.len, 64, 100)) + 1j*np.zeros((self.len,64, 100))


            for cnt, sample_i in enumerate(data_idx):
                with h5py.File(EnvPara["Dataset_path"]["DeepMIMO_O1"]+f"/3_{sample_i}.h5py", 'r') as f:
                    channel_real = f["channel_real"][:]
                    channel_imag = f["channel_imag"][:]
                    UElocation = f["UElocation"][:][:2]

                    self.labels[cnt,:] = UElocation/20
                    self.data[cnt,:,:] = channel_real + 1j*channel_imag
            print(self.data.shape)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        

        # if self.EnvPara["Dataset"] == "MaMIMO":
        #     channel_1 = self.data[idx,:,:]
        # elif self.EnvPara["Dataset"] == "DeepMIMO_O1":
        #     channel_1 = self.data[idx,:,:]

        channel_1 = self.data[idx,:,:]
        channel_1 = channel_normalization(channel_1)

        # CFR with amplitude and phase 
        channel_input = np.zeros((2, channel_1.shape[0], channel_1.shape[1]))
        channel_input[0,:,:] = np.abs(channel_1).astype(np.float32)
        channel_input[1,:,:] = np.angle(channel_1).astype(np.float32)



        # labels: 0,1:location, 2:distance(toa), 3:angle (aoa)
        UElocation =self.labels[idx,:]
                
        return  channel_input, UElocation
    





class generate_Dataset_test(Dataset):
    def __init__(self, EnvPara):
        self.EnvPara = EnvPara



        if self.EnvPara["Dataset"] == "MaMIMO":
            self.labels = np.load(EnvPara["Label_path"]["MaMIMO"])
            data_idx = np.loadtxt("./dataset/MaMIMO_test.txt")
            data_idx = data_idx.astype(int)
            self.len = len(data_idx)
            self.labels = self.labels[data_idx,:2]/100

            self.data = np.zeros((self.len, 64, 100)) + 1j*np.zeros((self.len , 64, 100))
            for cnt, i in enumerate(data_idx):
                self.data[cnt,:,:] = np.load(EnvPara["Dataset_path"]["MaMIMO"]+f"channel_measurement_{i:06d}.npy")

        elif self.EnvPara["Dataset"] == "DeepMIMO_O1":
            data_idx = np.loadtxt("./dataset/DeepMIMO_test.txt")
            data_idx = data_idx.astype(int)
            self.len = len(data_idx)
            self.labels = np.zeros((self.len,2))
            self.data = np.zeros((self.len, 64, 100)) + 1j*np.zeros((self.len,64, 100))


            for cnt, sample_i in enumerate(data_idx):
                with h5py.File(EnvPara["Dataset_path"]["DeepMIMO_O1"]+f"/3_{sample_i}.h5py", 'r') as f:
                    channel_real = f["channel_real"][:]
                    channel_imag = f["channel_imag"][:]
                    UElocation = f["UElocation"][:][:2]

                    self.labels[cnt,:] = UElocation/20
                    self.data[cnt,:,:] = channel_real + 1j*channel_imag
            print(self.data.shape)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        

        # if self.EnvPara["Dataset"] == "MaMIMO":
        #     channel_1 = self.data[idx,:,:]
        # elif self.EnvPara["Dataset"] == "DeepMIMO_O1":
        #     channel_1 = self.data[idx,:,:]

        channel_1 = self.data[idx,:,:]
        channel_1 = channel_normalization(channel_1)

        # CFR with amplitude and phase 
        channel_input = np.zeros((2, channel_1.shape[0], channel_1.shape[1]))
        channel_input[0,:,:] = np.abs(channel_1).astype(np.float32)
        channel_input[1,:,:] = np.angle(channel_1).astype(np.float32)



        # labels: 0,1:location, 2:distance(toa), 3:angle (aoa)
        UElocation =self.labels[idx,:]
                
        return  channel_input, UElocation

