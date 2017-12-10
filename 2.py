import numpy as np
from numpy import *
import matplotlib.pyplot as plt


num_users = 943
num_items = 1682
k = 20

learning_rate = 1e-4
regularization_coefficient = 1e-2


def loss(dataset, P, Q):
    error_square_sum = 0
    for sample in dataset:
        user_id, item_id, score = sample
        error = score - P[user_id-1, :].dot(Q[item_index, :])
        error_square_sum += error ** 2

    loss = error_square_sum / dataset.shape[0] + regularization_coefficient * (np.sum(P ** 2) + np.sum(Q ** 2))
    return loss


training_data = np.load("./ml-100k/data/u1_train_half.npy")
testing_data = np.load("./ml-100k/data/u1_test_half.npy")

P = np.random.random((num_users, k))
Q = np.random.random((num_items, k))


training_loss_history = []
testing_loss_history = []

iter = 0
max_training_iteration = 200
for i in range(max_training_iteration):
    sample = training_data[np.random.randint(0, training_data.shape[0])]
    user_index = sample[0] - 1
    item_index = sample[1] - 1
    real_score = sample[2]

    predict_score = P[user_index, :].dot(Q[item_index, :])
    error = real_score - predict_score

    grad_p = -2 / training_data.shape[0] * (error * Q[item_index, :] + regularization_coefficient * P[user_index, :])
    grad_q = -2 / training_data.shape[0] * (error * P[user_index, :] + regularization_coefficient * Q[item_index, :])

    P[user_index, :] -= learning_rate * grad_p
    Q[item_index, :] -= learning_rate * grad_q

    training_loss = loss(training_data, P, Q)
    testing_loss = loss(testing_data, P, Q)

    training_loss_history.append(training_loss)
    testing_loss_history.append(testing_loss)
    print(iter, training_loss, testing_loss)

    if iter > max_training_iteration:
        break
    iter += 1

iters = np.arange(iter)
plt.plot(iters, training_loss_history, label='training loss')
plt.plot(iters, testing_loss_history, label='testing loss')

plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.show()