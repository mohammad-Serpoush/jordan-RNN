import matplotlib.pyplot as plt
import numpy as np
from functions import get_data, sigmoid, sigmoid_derivitive, tanh, tanh_derivitive


x_train, y_train, x_test, y_test = get_data()


n0 = 5
n1 = 4
n2 = 6
n3 = 1


w1 = np.random.uniform(low=-0.5, high=0.5, size=(n0, n1))
w2 = np.random.uniform(low=-0.5, high=0.5, size=(n1, n2))
w3 = np.random.uniform(low=-0.5, high=0.5, size=(n2, n3))
w1j = np.random.uniform(low=-0.5, high=0.5, size=(1, 4))
w2j = np.random.uniform(low=-0.5, high=0.5, size=(1, 6))


MSE_train = []
MSE_test = []
epoch = 200
etac = 0.05
eta = 0.001
output_train = []
output_test = []
for i in range(epoch):
    sqr_err_epoch_train = []
    sqr_err_epoch_test = []
    output_train = []
    output_test = []
    o3 = np.zeros((1, 1))

    for input, target in zip(x_train, y_train):
        input = np.array([input])
        net1 = np.matmul(input, w1)
        netj = np.matmul(o3, w1j)
        netx1 = np.add(net1, netj)
        o1 = sigmoid(netx1)

        net2 = np.matmul(o1, w2)
        netj = np.matmul(o3, w2j)
        netx2 = np.add(net2, netj)

        o2 = sigmoid(netx2)

        net3 = np.matmul(o2, w3)
        o_old = o3
        o3 = net3
        output_train.append(o3[0, 0])
        e = target - o3[0, 0]
        delta_w3 = eta * -1 * e * o2.T
        sqr_err_epoch_train.append(e**2)

        f_2_derivitive = sigmoid_derivitive(netx2)
        f_1_derivitive = sigmoid_derivitive(netx1)

        A = np.matmul(f_2_derivitive, w3)
        B = np.matmul(f_1_derivitive, w2)
        delta_w2 = e * -1 * eta * np.matmul(A, o1).T
        C = np.matmul(B, A)
        delta_w1 = e * -1 * eta * np.matmul(C, input).T

        J2 = np.matmul(A, o_old)
        delta_wj2 = e * -1 * eta * J2.T

        # ------------------------------------
        J1 = np.matmul(C, o_old)
        delta_w1j = e * -1 * eta * J1.T

        w1 = np.subtract(w1, delta_w1)
        w2 = np.subtract(w2, delta_w2)
        w3 = np.subtract(w3, delta_w3)
        w2j = np.subtract(w2j, delta_wj2)
        w1j = np.subtract(w1j, delta_w1j)
    mse_epoch_train = 0.5 * \
        ((sum(sqr_err_epoch_train))/np.shape(x_train)[0])
    MSE_train.append(mse_epoch_train)
    for j in range(np.shape(x_test)[0]):
        input = x_test[j, :]
        target = y_test[j]
        net1 = np.matmul(input, w1)
        netj = np.matmul(o3, w1j)
        netx1 = np.add(net1, netj)
        o1 = sigmoid(netx1)

        net2 = np.matmul(o1, w2)
        netj = np.matmul(o3, w2j)
        netx2 = np.add(net2, netj)

        o2 = sigmoid(netx2)

        net3 = np.matmul(o2, w3)
        o3 = net3
        output_test.append(o3[0])
        error = target - o3[0, 0]
        sqr_err_epoch_test.append(error ** 2)

    mse_epoch_test = 0.5 * ((sum(sqr_err_epoch_test))/np.shape(x_test)[0])
    MSE_test.append(mse_epoch_test)
    print("Epoch {}  , Training Error : {}  , Test Error : {}".format(
        i+1, mse_epoch_train, mse_epoch_test))


# Train
m_train, b_train = np.polyfit(y_train, output_train, 1)

# Test
m_test, b_test = np.polyfit(y_test, output_test, 1)

# Plots
fig, axs = plt.subplots(3, 2)
axs[0, 0].plot(MSE_train, 'b')
axs[0, 0].set_title('MSE Train')
axs[0, 1].plot(MSE_test, 'r')
axs[0, 1].set_title('Mse Test')

axs[1, 0].plot(y_train, 'b')
axs[1, 0].plot(output_train, 'r')
axs[1, 0].set_title('Output Train')
axs[1, 1].plot(y_test, 'b')
axs[1, 1].plot(output_test, 'r')
axs[1, 1].set_title('Output Test')
axs[2, 0].plot(y_train, output_train, 'b*')
axs[2, 0].plot(y_train, m_train*y_train+b_train, 'r')
axs[2, 0].set_title('Regression Train')
axs[2, 1].plot(y_test, output_test, 'b*')
axs[2, 1].plot(y_test, m_test*y_test+b_test, 'r')
axs[2, 1].set_title('Regression Test')

plt.show()
