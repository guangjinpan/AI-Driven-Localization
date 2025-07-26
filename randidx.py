import random
import numpy as np

# 随机采样
samples = random.sample(range(0, 252004), 10000)
# 写入 txt 文件
with open("./dataset/MaMIMO_train.txt", "w") as f:
    for number in samples[:7000]:
        f.write(f"{number}\n")

# 写入 txt 文件
with open("./dataset/MaMIMO_val.txt", "w") as f:
    for number in samples[7000:8000]:
        f.write(f"{number}\n")


# 写入 txt 文件
with open("./dataset/MaMIMO_test.txt", "w") as f:
    for number in samples[8000:10000]:
        f.write(f"{number}\n")

samples = np.random.permutation(90000)
with open("./dataset/DeepMIMO_train.txt", "w") as f:
    for number in samples[:7000]:
        f.write(f"{number}\n")

with open("./dataset/DeepMIMO_val.txt", "w") as f:
    for number in samples[7000:8000]:
        f.write(f"{number}\n")


with open("./dataset/DeepMIMO_test.txt", "w") as f:
    for number in samples[8000:10000]:
        f.write(f"{number}\n")