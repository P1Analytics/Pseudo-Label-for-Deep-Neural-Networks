# https://github.com/mnielsen//neural-networks-and-deep-learning/ all the credit to this guy
import random
import numpy as np
from collections import Counter

class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]


    def update_by_mini_batch(self, mini_batch, eta=3):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activations = []
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w, activation = self.backprop(x, y)
            activations.append(activation)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]
        return activations

    def feedforward(self, a):
        layer = 0
        for b, w in zip(self.biases, self.weights):
           a = sigmoid(np.dot(w, a) + b)
            # TODO have to deal with exp overflow error for changing active function to rectifier
            # if layer ==0:
            #     a = rectifier(np.dot(w, a) + b)
            # else:
            #     a = sigmoid(np.dot(w, a) + b)
            # layer += 1
        return a

    def backprop(self, x, y):

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward during training phase
        activation = x
        activations = [activation]  # list to store all the activations, layer by layer,first layer the original input
        zs = []  # list to store all the z vectors, layer by layer
        # layer = -1
        for b, w in zip(self.biases, self.weights):
            # activation = np.multiply(activation, dropout(0.5, np.shape(activation)))
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            # TODO use differnet active functions return all WRONG label ! WHY ??
            # todo also add drop out on the hidd
            # layer += 1
            # if layer == 0 :
            #     activation = rectifier(z)
            # else:
            #     activation = sigmoid(z)
            activations.append(activation)

        # backward phase
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            # TODO different active function in the hidden layer, different prime
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sigmoid_prime(z)
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return nabla_b, nabla_w, activation

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]

        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        # TODO different loss function different derivation before active - y !!! ours: (1-y)/(f*(1-f)*ln10)
        # return (1-y)/(output_activations*(1-output_activations)*np.log(10))
        return (output_activations - y)

    def crossEntropy(self, labels, output):
        return np.sum(np.nan_to_num(-labels * np.log(output) - (1 - labels) * np.log(1 - output)))


    def lossFunction(self, mini_batches, validation_data,currentEpoch):
        lFunctionValue = 0
        lPrimeFunctionValue = 0

        for mini_batch in mini_batches:
            activations = self.update_by_mini_batch(mini_batch)
            y = [ i for x,i in mini_batch]
            shape = np.shape(y)[:-1] # since y shape is (size of mini_batch,10,1), 1 is useless-->cut off
            y = np.reshape(y, shape)
            activations = np.reshape(activations, shape)
            lFunctionValue += self.crossEntropy(y, activations).mean()


        random.shuffle(validation_data)
        np.shape(validation_data)
        mini_valid_batch_size = 3
        n = len(validation_data)
        mini_valid_batches = [validation_data[k:k + mini_valid_batch_size] for k in range(0, n, mini_valid_batch_size)]

        for mini_vilid_batch in mini_valid_batches:
            activations =[]
            Y = []
            for x,y in mini_vilid_batch:
                activation = self.feedforward(x)
                activations.append(activation)
                Y.append(y)
            shape = np.shape(Y)[:-1]  # since y shape is (size of mini_batch,10,1), 1 is useless-->cut off
            Y = np.reshape(Y, shape)
            activations = np.reshape(activations, shape)
            lPrimeFunctionValue += self.crossEntropy(Y, activations)

        print(lFunctionValue / np.size(mini_batches) +
              lPrimeFunctionValue * self.alfaCoefficient(10, 20, currentEpoch)/ np.size(mini_valid_batches))

    def alfaCoefficient(self, epochT1, epochT2, currentEpoch):
        if currentEpoch < epochT1:
            return 0
        elif epochT1 <= currentEpoch & currentEpoch < epochT2:
            return ((currentEpoch - epochT1) * 0.4) / epochT2 - epochT1
        elif epochT2 <= currentEpoch:
            return 0.4

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data: n_test = len(test_data)

        print("number of test images correctly recognized by the neural network after each epoch of training")
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_by_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def SGDPL(self, training_data, epochs, mini_batch_size, eta, validation_data=None, test_data=None):

        # Now we want pseudo labels, by using training data and validation data
        validation_results = []
        mini_batch_activations = []
        for j in range(epochs):
            # training session
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]
            for mini_batch in mini_batches:
                mini_batch_activation = self.update_by_mini_batch(mini_batch, eta)
            # pseudo label session
            mini_batch_activations.append(mini_batch_activation)
            test_results = []
            for (x, y) in validation_data:
                output = self.feedforward(x)
                test_results.append(np.argmax(output))
            print(test_results)
            validation_results.append(test_results)
        print("finally ",validation_results)
        validation_array_results = np.array(validation_results)
        pseudo_labels = []
        for i in range(validation_array_results.shape[1]):
            label = vectorized_result(Counter(validation_array_results[:, i]).most_common(1)[0][0])
            pseudo_labels.append(label)

        X = [ x for x,y in validation_data]
        validation_PL_data = list(zip(X,pseudo_labels))
        # Now we have pseudo labels in validation dataset

        for j in range(epochs):
            self.lossFunction(mini_batches, validation_PL_data, j)




def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def rectifier(z):
    return np.maximum(np.reshape(np.zeros((z.shape)), z.shape), z)


def dropout(probability, shape):
    # create the 1 or 0 mask (same size as input pics X ) with probability
    return np.random.binomial(1, probability, shape)


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


if __name__ == "__main__":
    import mnist_loader
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    # DROP NN
    net = Network([784, 10, 10])
    # net.SGD(training_data, 1, 15, 3.0, test_data=test_data)
    # net = Network([784, 50000, 10])
    # net.SGD(training_data, 4, 100, 3.0, test_data=test_data)
    # net.SGD(training_data, 4, 600, 3.0, test_data=test_data)
    # net.SGD(training_data, 4, 1000, 3.0, test_data=test_data)
    # net.SGD(training_data, 4, 3000, 3.0, test_data=test_data)

    # DROP NN + PL
    # TODO
    net.SGDPL(training_data, 4, 3, 3.0, validation_data=validation_data, test_data=test_data)


    # DROP NN + PL + DAE
    # TODO
