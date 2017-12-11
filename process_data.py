import numpy as np

num_users = 943
num_items = 1682
R = np.zeros(shape=(num_users, num_items))
training_data = []

with open("./ml-100k/u1.base", 'r') as f:
    num = 0
    while True:
        line = f.readline()

        if num > 300:
            break
        else:
            num += 1
            line = line.split()
            line = [int(i) for i in line]
            training_data.append(line[:3])

np.save("./ml-100k/data/u1_train_half.npy", training_data)