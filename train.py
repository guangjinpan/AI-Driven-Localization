    
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



# Set environment variable
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


EnvPara["Label_path"]={}
EnvPara["Dataset_path"]={}
EnvPara["Label_path"]["MaMIMO"] = "/mimer/NOBACKUP/groups/e2e_comms/guangjin/AILoc/MaMIMO/ultra_dense/ULA_lab_LoS/user_positions.npy"
EnvPara["Dataset_path"]["MaMIMO"] = "/mimer/NOBACKUP/groups/e2e_comms/guangjin/AILoc/MaMIMO/ultra_dense/ULA_lab_LoS/samples/"

EnvPara["Dataset_path"]["DeepMIMO_O1"] = "/mimer/NOBACKUP/groups/e2e_comms/guangjin/AILoc/DeepMIMO_O1/BS3_100sc_64at_20M/samples"

if __name__ == '__main__':   
    EnvPara["model_path"]= EnvPara["model_path"] + EnvPara["Dataset"] + "_"  + EnvPara["model"] 
    if EnvPara["Dataset"] == "MaMIMO":
        EnvPara["input_subcarrier_dim"] = 100
        EnvPara["input_antenna_dim"] = 64

    elif EnvPara["Dataset"] == "DeepMIMO_O1":
        EnvPara["input_subcarrier_dim"] = 100
        EnvPara["input_antenna_dim"] = 64

    print(EnvPara)

    train_dataset = generate_Dataset(EnvPara)
    train_dataloader = DataLoader(train_dataset, batch_size=32, num_workers=8, shuffle=True, drop_last=False, pin_memory=True)
    val_dataset = generate_Dataset_val(EnvPara)
    val_dataloader = DataLoader(val_dataset, batch_size=32, num_workers=8, shuffle=True, drop_last=False, pin_memory=True)    

    FMmodel = Wrapper(EnvPara=EnvPara)


    FMmodel.to(EnvPara["device"])

    model_path = "test"
    model_checkpoint = ModelCheckpoint(
        dirpath=EnvPara["model_path"],
        filename=model_path + '{epoch:02d}',
        save_top_k=1,  # Only save the best model
        monitor="val/ave_loss",  # Monitor validation loss to determine the best model
        mode="min",  # 'min' means the lower the better; use 'max' for metrics like accuracy
        save_weights_only=True,
    )

    latest_model_checkpoint = ModelCheckpoint(
        dirpath=EnvPara["model_path"],
        filename=model_path + '_latest',
        save_top_k=1,  # Only keep the latest model
        save_last=True,  # Always save the most recent model
        every_n_epochs=5,  # Save every n epochs
        save_weights_only=True,
    )

    logger = pl.loggers.TensorBoardLogger('./logs', name=EnvPara["Dataset"] + "_"  + EnvPara["model"] )
    trainer = Trainer(
        log_every_n_steps=0,  # Do not log at each step
        enable_progress_bar=False,
        precision=16,
        devices=1,
        accelerator='gpu',
        logger=logger,
        max_epochs=EnvPara["epochs"],
        callbacks=[model_checkpoint, latest_model_checkpoint],
    )
    trainer.fit(FMmodel, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
