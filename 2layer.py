import numpy as np
import numpy as np
from numpy import linalg
import matplotlib
import matplotlib.pyplot as plt
from libsvm.svmutil import svm_read_problem
from sklearn.model_selection import train_test_split

def oneHot(array):
    result = np.zeros((len(array),10))
    for number in range(len(array)):
        result[number][int(array[number])] = 1
    return result

def softmax(Z):
    Z_shift = Z - np.max(Z, axis=0)
    A = np.exp(Z_shift) / np.sum(np.exp(Z_shift), axis=0)

    cache = Z_shift

    return A, cache

def relu(Z):
    A = np.maximum(0, Z)
    assert (A.shape == Z.shape)
    cache = Z
    return A, cache


def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(1)

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters

def linear_forward(A, W, b):
    Z = np.dot(W, A) + b

    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    if activation == "softmax":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = softmax(Z)

    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = -(np.sum(Y * np.log(AL))) / float(m)
    # cost = np.squeeze(cost)
    assert (cost.shape == ())

    return cost

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T) / float(m)
    db = np.sum(dZ, axis=1, keepdims=True) / float(m)
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ

def softmax_backward(Y, cache):
    Z = cache

    s = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    dZ = s - Y

    assert (dZ.shape == Z.shape)

    return dZ

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "softmax":
        dZ = softmax_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2  # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(1, L + 1):
        parameters['W' + str(l)] -= learning_rate * grads['dW' + str(l)]
        parameters['b' + str(l)] -= learning_rate * grads['db' + str(l)]

    return parameters


def two_layer_model(X, Y, layers_dims, learning_rate=0.1, num_iterations=3000, print_cost=False):

    np.random.seed(1)
    grads = {}
    costs = []  # to keep track of the cost
    m = X.shape[1]  # number of examples
    (n_x, n_h, n_y) = layers_dims

    # Initialize parameters dictionary, by calling one of the functions you'd previously implemented
    parameters = initialize_parameters(n_x, n_h, n_y)

    # Get W1, b1, W2 and b2 from the dictionary parameters.
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Loop (gradient descent)

    for i in range(0, num_iterations):

        # Forward propagation
        A1, cache1 = linear_activation_forward(X, W1, b1, activation='relu')
        A2, cache2 = linear_activation_forward(A1, W2, b2, activation='softmax')

        # Compute cost
        cost = compute_cost(A2, Y)

        # Backward propagation
        dA1, dW2, db2 = linear_activation_backward(Y, cache2, activation='softmax')
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, activation='relu')

        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        # Print the cost every 100 training example
        if i % 50 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            costs.append(cost)



    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters

def predict(X, y, parameters):
    m = X.shape[1]

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    A1, _ = linear_activation_forward(X, W1, b1, activation='relu')
    probs, _ = linear_activation_forward(A1, W2, b2, activation='softmax')

    predicted = np.argmax(probs, axis=0)

    # print ("predictions: " + str(p))
    # print ("true labels: " + str(y))
    print(m, ' test case predicted.', sep='')
    correct_num = np.sum(predicted == y)
    print(correct_num, ' are correct.', sep='')
    #print('Accuracy = ', np.round(correct_num * 100 / len(predicted)), '%', sep='')
    print("Accuracy = " + str(correct_num / float(m)), sep='')

    return predicted

y_raw, x_raw = svm_read_problem('mnist.scale')

y = np.array(y_raw)
x = np.zeros((len(y_raw), 780))
for i in range(len(y_raw)):
    line = x_raw[i]
    for k, v in line.items():
        x[i][k - 1] = v

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

y_train = oneHot(y_train)

print('====== Binary (Digit 1 and 2) =======')

y_bin = np.concatenate((y[y==1],[0]*sum(y==2)))
temp = np.zeros((len(y_bin), 2))
for number in range(len(y_bin)):
    temp[number][int(y_bin[number])] = 1
y_bin = temp

x_bin = np.concatenate((x[y==1],x[y==2]))

x_train_bin, x_test_bin, y_train_bin, y_test_bin = train_test_split(x_bin, y_bin, test_size=0.3)

parameters_bin = two_layer_model(x_train_bin.T, y_train_bin.T, (780, 100, 2), 0.1, 100, True)

prediction_bin = predict(x_test_bin.T, y_test_bin.T, parameters_bin)

print('====== Multi-class =======')
parameters = two_layer_model(x_train.T, y_train.T, (780, 100, 10), 0.1, 300, True)

prediction = predict(x_test.T, y_test.T, parameters)




