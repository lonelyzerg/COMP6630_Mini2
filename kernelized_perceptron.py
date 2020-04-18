import numpy as np
from numpy import linalg
import matplotlib
import matplotlib.pyplot as plt
from libsvm.svmutil import svm_read_problem
from sklearn.model_selection import train_test_split


def decToBin(array):
    result={
        0: [0, 0, 0, 0],
        1: [0, 0, 0, 1],
        2: [0, 0, 1, 0],
        3: [0, 0, 1, 1],
        4: [0, 1, 0, 0],
        5: [0, 1, 0, 1],
        6: [0, 1, 1, 0],
        7: [0, 1, 1, 1],
        8: [1, 0, 0, 0],
        9: [1, 0, 0, 1]
    }
    return np.array([result[number] for number in array])

def binToDec(array):
    return np.array([num[0]*8 + num[1] * 4 + num[2] * 2 + num[3] for num in array])

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

class KernelPerceptron(object):

    def __init__(self, kernel=polynomial_kernel, T=1):
        self.kernel = kernel
        self.T = T

    def fit(self, X, y):
        n_samples, n_features = X.shape
        #np.hstack((X, np.ones((n_samples, 1))))
        self.alpha = np.zeros(n_samples, dtype=np.float64)

        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j])

        for t in range(self.T):
            for i in range(n_samples):
                if np.sign(np.sum(K[:,i] * self.alpha * y)) != y[i]:
                    self.alpha[i] += 1.0

        # Support vectors
        sv = self.alpha > 1e-5
        ind = np.arange(len(self.alpha))[sv]
        self.alpha = self.alpha[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        print ("%d support vectors out of %d points" % (len(self.alpha),n_samples))

    def project(self, X):
        y_predict = np.zeros(len(X))
        for i in range(len(X)):
            s = 0
            for a, sv_y, sv in zip(self.alpha, self.sv_y, self.sv):
                s += a * sv_y * self.kernel(X[i], sv)
            y_predict[i] = s
        return y_predict

    def predict(self, X):
        X = np.atleast_2d(X)
        n_samples, n_features = X.shape
        #np.hstack((X, np.ones((n_samples, 1))))
        return np.sign(self.project(X))


y_raw, x_raw = svm_read_problem('mnist.scale')

y = np.array(y_raw)
x = np.zeros((len(y_raw), 780))
for i in range(len(y_raw)):
    line = x_raw[i]
    for k, v in line.items():
        x[i][k - 1] = v

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

y_train = decToBin(y_train)

y_train1 = np.copy(y_train[:,0])

y_train2 = np.copy(y_train[:,1])

y_train3 = np.copy(y_train[:,2])

y_train4 = np.copy(y_train[:,3])

perceptron1 = KernelPerceptron(polynomial_kernel, 1)
perceptron1.fit(x_train, y_train1)
perceptron2 = KernelPerceptron(polynomial_kernel, 1)
perceptron2.fit(x_train, y_train2)
perceptron3 = KernelPerceptron(polynomial_kernel, 1)
perceptron3.fit(x_train, y_train3)
perceptron4 = KernelPerceptron(polynomial_kernel, 1)
perceptron4.fit(x_train, y_train4)

predicted = np.zeros(len(y_test))
for i in range(len(y_test)):
    pred = np.zeros(3)
    pred[0] = perceptron1.predict(x_test[i])
    pred[1] = perceptron2.predict(x_test[i])
    pred[2] = perceptron3.predict(x_test[i])
    pred[3] = perceptron4.predict(x_test[i])
    predicted[i] = binToDec([pred])

print(len(predicted), ' test case predicted.', sep='')
correct_num = np.sum(predicted == y_test)
print(correct_num, ' are correct.', sep='')
print('Accuracy = ', np.round(correct_num * 100 / len(predicted)), '%', sep='')
