import numpy as np
import matplotlib.pyplot as plt

num_users = 943
num_items = 1682
k = 20

gamma = 1e-2


def loss(X, u, v):
    X_minus_uvT = X - u.dot(v.T)
    loss = np.sum(X_minus_uvT * X_minus_uvT)
    loss += gamma * (np.sum(u * u) + np.sum(v * v))
    return loss


def compute_gradient(X, u, v, flag):
    if flag == 'U':
        math_item_1 = u.dot(v.T).dot(v)
        math_item_2 = X.dot(v)
        math_item_3 = gamma * u
        return 2 * (math_item_1 - math_item_2 + math_item_3)

    if flag == 'V':
        math_item_1 = v.dot(u.T).dot(U)
        math_item_2 = X.T.dot(u)
        math_item_3 = gamma * v
        return 2 * (math_item_1 - math_item_2 + math_item_3)


training_max_iterations = 200
learning_rate = 1e-4

R = np.load("./ml-100k/data/u1_train.npy")
U = np.random.random((num_users, k))
V = np.random.random((num_items, k))
R_test = np.load("./ml-100k/data/u1_test.npy")
#U = np.empty(shape=(num_users, k))
#V = np.empty(shape=(num_items, k))

training_losses = []
testing_losses = []
for i in range(training_max_iterations):
    if i % 2 == 0:
        '''
        grad_U = compute_gradient(R, U, V, 'U')
        U -= learning_rate * grad_U
        '''

        temp = V.T.dot(V) + gamma * np.eye(k)
        U = R.dot(V).dot(temp ** -1)
    if i % 2 == 1:
        '''
        grad_V = compute_gradient(R, U, V, 'V')
        V -= learning_rate * grad_V
        '''
        temp = U.T.dot(U) + gamma * np.eye(k)
        V = R.T.dot(U).dot(temp ** -1)

    new_train_loss = loss(R, U, V)
    new_test_loss = loss(R_test, U, V)
    training_losses.append(new_train_loss)
    testing_losses.append(new_test_loss)
    print(i, new_train_loss, new_test_loss)

iters = np.arange(training_max_iterations)

plt.plot(iters, training_losses, label='training loss')
plt.plot(iters, testing_losses, label='testing loss')

plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.show()