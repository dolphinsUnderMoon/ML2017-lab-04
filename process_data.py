import numpy as np

num_users = 943
num_items = 1682
R = np.zeros(shape=(num_users, num_items))

with open("./ml-100k/u1.test", 'r') as f:
    while True:
        line = f.readline()

        if not line:
            break
        else:
            line = line.split()
            R[int(line[0]) - 1, int(line[1]) - 1] = int(line[2])

np.save("./ml-100k/data/u1_test.npy", R)