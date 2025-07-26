import random
import numpy as np

# random sample 10000 numbers from 0 to 252003
samples = random.sample(range(0, 252004), 10000)

with open("./dataset/MaMIMO_train.txt", "w") as f:
    for number in samples[:7000]:
        f.write(f"{number}\n")


with open("./dataset/MaMIMO_val.txt", "w") as f:
    for number in samples[7000:8000]:
        f.write(f"{number}\n")



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