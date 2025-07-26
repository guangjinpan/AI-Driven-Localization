import numpy as np

x=np.load("lstm_ground_truth.npy")
x1=np.load("lstm_predictions.npy")
print(np.mean(np.abs(x-x1)),np.max(x),np.min(x))

labels = np.load("/mimer/NOBACKUP/groups/e2e_comms/guangjin/AILoc/MaMIMO/ultra_dense/ULA_lab_LoS/user_positions.npy")
print(labels)