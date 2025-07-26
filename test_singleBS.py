    
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
from dataload import *


import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader, ConcatDataset, random_split, Dataset
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse


def set_seed(seed=42):
    random.seed(seed)  # Python 
    np.random.seed(seed)  # NumPy 
    torch.manual_seed(seed)  # PyTorch CPU 
    torch.cuda.manual_seed(seed)  # PyTorch GPU 
    torch.cuda.manual_seed_all(seed) 


parser = argparse.ArgumentParser(description="Environment Parameters")

# Training Parameters
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--Dataset", type=str, default="MaMIMO")    # attenna number
parser.add_argument("--input_subcarrier_dim", type=int, default=128)   # CSI  in subcarrier dimension
parser.add_argument("--input_antenna_dim", type=int, default=32)   # CSI in antenna dimension
parser.add_argument("--model", type=str, default="ResNet32")     # model type
parser.add_argument("--model_path", type=str, default='./model/')
parser.add_argument("--lr", type=float, default=1e-4)
args = parser.parse_args()


print(args)
EnvPara = {
    "epochs": args.epochs,
    "input_subcarrier_dim": args.input_subcarrier_dim,
    "input_antenna_dim": args.input_antenna_dim,
    "Dataset": args.Dataset,
    "model": args.model,
    "model_path": args.model_path,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "lr": args.lr,
}


# EnvPara["input_feature_dim"] = 2 *  EnvPara["BS_Num"]
# print("EnvPara[input_feature_dim]",EnvPara["input_feature_dim"])

# Set environment variable
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


EnvPara["Label_path"]={}
EnvPara["Dataset_path"]={}
EnvPara["Label_path"]["MaMIMO"] = "/mimer/NOBACKUP/groups/e2e_comms/guangjin/AILoc/MaMIMO/ultra_dense/ULA_lab_LoS/user_positions.npy"
EnvPara["Dataset_path"]["MaMIMO"] = "/mimer/NOBACKUP/groups/e2e_comms/guangjin/AILoc/MaMIMO/ultra_dense/ULA_lab_LoS/samples/"
EnvPara["Dataset_path"]["DeepMIMO_O1"] = "/mimer/NOBACKUP/groups/e2e_comms/guangjin/AILoc/DeepMIMO_O1/BS3_100sc_64at_20M/samples"

if __name__ == '__main__':   
    EnvPara["model_path"]= EnvPara["model_path"] + EnvPara["Dataset"] + "_"  + EnvPara["model"] + "/" + "model.ckpt"
    EnvPara["save_result_file"] = EnvPara["Dataset"] + "_"  + EnvPara["model"]
    if EnvPara["Dataset"] == "MaMIMO":
        EnvPara["input_subcarrier_dim"] = 100
        EnvPara["input_antenna_dim"] = 64

    elif EnvPara["Dataset"] == "DeepMIMO_O1":
        EnvPara["input_subcarrier_dim"] = 100
        EnvPara["input_antenna_dim"] = 64
 

    print(EnvPara)

    result = np.zeros((500000,2))
    label = np.zeros((500000,2))
    mse = np.zeros((500000,1))
    cnt = 0
    UElocation_est = np.zeros((1,2))

    model = Wrapper(EnvPara)
    model= Wrapper.load_from_checkpoint(EnvPara["model_path"], EnvPara=EnvPara, strict=False)
    model.to(EnvPara["device"])
    test_dataset = generate_Dataset_test(EnvPara)
    test_dataloader = DataLoader(test_dataset, batch_size = 32, num_workers = 8, shuffle = False, drop_last = False, pin_memory=True)


    for  channel_Antenna_subcarrier, UElocation in test_dataloader:
        model.channel_fdmdl.eval()
        channel_Antenna_subcarrier = channel_Antenna_subcarrier.to(EnvPara["device"]).float()
        UElocation = UElocation.to(EnvPara["device"]).float()
        pred, loss = model.channel_fdmdl(channel_Antenna_subcarrier = channel_Antenna_subcarrier, UElocation = UElocation)
        result[cnt : cnt+len(channel_Antenna_subcarrier)] = pred.cpu().detach().numpy() * 20
        label[cnt : cnt+len(channel_Antenna_subcarrier)] = UElocation.cpu().detach().numpy() * 20
        mse[cnt : cnt+len(channel_Antenna_subcarrier)] = loss.cpu().detach().numpy()
        cnt = cnt+len(channel_Antenna_subcarrier)

    # delete the non-coverage data
    result = result[:cnt,:]
    label = label[:cnt,:]
    mse = mse[:cnt,:]

    ave_mse=np.mean(mse) 


    # save results as txt
    merged = np.hstack((result, label))
    np.savetxt("./results/"+EnvPara["save_result_file"]+".txt", merged, fmt="%.6f", delimiter=' ')

    #compute distances
    distances = np.linalg.norm(result - label, axis=1)


    # Compute mean error
    mean_error = np.mean(np.sqrt(distances ** 2))

    print(f"mean error: {mean_error}, loss:{np.mean(mse)}")