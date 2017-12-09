import numpy as np
import os


user_num = 943
movie_num = 1682
feature_num = 5
R = - np.ones(shape=[user_num, movie_num])
train_data = []
validation_data = []

with open('./ml-100k/u1.base', ) as file:
    for line in file:
        line_ = line.split()
        user_id = int(line_[0])
        movie_id = int(line_[1])
        score = int(line_[2])
        R[user_id - 1][movie_id - 1] = score
        train_data.append([user_id, movie_id, score])

with open('./ml-100k/u1.test', ) as file:
    for line in file:
        line_ = line.split()
        user_id = int(line_[0])
        movie_id = int(line_[1])
        score = int(line_[2])
        validation_data.append([user_id, movie_id, score])


def compute_loss():
    global R, validation_data
    error = []
    for i in validation_data:
        error.append(i[2] - R[i[0] - 1][i[1] - 1])
    error = np.array(error)
    return np.mean(error ** 2)


def compute_singelSample_user_gradient(user_id, movie_id, score):
    global R, user, movie
    row = user_id - 1
    col = movie_id - 1
    gradient = movie[:, col]
    gradient.reshape([feature_num])
    gradient = (R[row][col] - score) * gradient
    return row, gradient


def compute_singelSample_movie_gradient(user_id, movie_id, score):
    global R, user, movie
    row = user_id - 1
    col = movie_id - 1
    gradient = user[row, :]
    gradient.reshape([feature_num])
    gradient = (R[row][col] - score) * gradient
    return col, gradient


# user = np.random.rand(user_num, feature_num)
# movie = np.random.rand(feature_num, movie_num)
user = np.load('user.npy')
movie = np.load('movie.npy')

lr = 0.0001
for i in range(0, 1):
    count = 0
    for j in train_data:
        row, user_grad = compute_singelSample_user_gradient(j[0], j[1], j[2])
        col, movie_grad = compute_singelSample_movie_gradient(j[0], j[1], j[2])
        user[row] -= lr * user_grad
        movie[:, col] -= lr * movie_grad
        R = np.matmul(user, movie)
        loss = compute_loss()
        count += 1
        if count % 100 == 0:
            print(loss)
        if count % 1000 == 0:
            np.save('user.npy', user)
            np.save('movie.npy', movie)
            print('saved')