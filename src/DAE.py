# https://github.com/mnielsen//neural-networks-and-deep-learning/ all the credit to this guy
# testing for overfit: plot training set and validation set error as a function of training set size.
# If they show a stable gap at the right end of the plot, you're probably overfitting.
import gzip
import pickle
import random
import numpy as np
from collections import Counter

np.seterr(all='ignore')


class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def backprop_with_mini_batch(self, mini_batch, eta=3, epoch=0, pseu=False, DAE=False):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y, epoch, pseu, DAE)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y, epoch=0, pseu=False, DAE=False):

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = x
        activations = [x]
        zs = []
        layer = 0
        for b, w in zip(self.biases, self.weights):
            # drop out rates: 50 % dropout for all hidden units;  20 % dropout for visible units
            activation = np.multiply(activation, dropout(0.2, np.shape(activation)))
            z = np.dot(w, activation) + b
            zs.append(z)
            if layer == 0:
                # activation = rectifier(z)
                activation = sigmoid(z)
            else:
                activation = sigmoid(z)
            layer += 1
            activations.append(activation)


        # backward phase
        if DAE:
            y = x
        delta = self.cost_derivative(activations[-1], y)

        if pseu:
            delta *= self.alfaCoefficient(epoch, DAE) # TODO when will be both pseudo label and DAE they are in different phase

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sigmoid_prime(z)
            # delta = np.dot(self.weights[-l + 1].transpose(), delta) * rectifier_prime(z)
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return nabla_b, nabla_w

    def feedforward(self, a):
        layer = 0
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b
            if layer == 0:
                # a = rectifier(z) # for hidden layers
                a = sigmoid(z)
            else:
                a = sigmoid(z)  # for output layer
            layer += 1
        return a

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return output_activations - y

    def alfaCoefficient(self, currentEpoch, DAE=False):
        if DAE:
            epochT1, epochT2 = 200, 800
        else:
            epochT1, epochT2 = 100, 600
        if currentEpoch < epochT1:
            return 0
        elif epochT1 <= currentEpoch & currentEpoch < epochT2:
            return ((currentEpoch - epochT1) * 0.4) / epochT2 - epochT1
        elif epochT2 <= currentEpoch:
            return 3

    def SGD_DAE(self, training_data, epochs, mini_batch_size, eta, validation_data=None, test_data=None):
        # DAE unsupervised learning
        # Now we want pseudo labels, by using training data and validation data
        validation_results = []
        for j in range(epochs):
            # training session
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]

            for mini_batch in mini_batches:
                self.backprop_with_mini_batch(mini_batch, eta, DAE=True)

        # Fine tuning supervised learning
        secondNetwork = Network([784, 9, 10])
        secondNetwork.weights[0] = self.weights[0]
        secondNetwork.biases[0] = self.biases[0]

        for mini_batch in mini_batches:
            secondNetwork.backprop_with_mini_batch(mini_batch, eta)

            # pseudo label session
            test_results = []
            for (x, y) in validation_data:
                output = secondNetwork.feedforward(x)
                test_results.append(np.argmax(output))
            validation_results.append(test_results)

        validation_array_results = np.array(validation_results)
        pseudo_labels = []
        validation_results = []
        for i in range(validation_array_results.shape[1]):
            a = Counter(validation_array_results[:, i]).most_common(1)[0][0]
            validation_results.append(a)
            label = vectorized_result(a)
            pseudo_labels.append(label)
        validation_PL_data = list(zip([x for x, y in validation_data], pseudo_labels))
        # print("the real one is ", [y for x, y in validation_data])
        # Now we have pseudo labels in validation dataset

        for j in range(epochs):
            random.shuffle(validation_PL_data)
            np.shape(validation_data)
            mini_valid_batch_size = 3
            n = len(validation_data)
            mini_valid_batches = [validation_PL_data[k:k + mini_valid_batch_size] for k in
                                  range(0, n, mini_valid_batch_size)]
            for mini_vilid_batch in mini_valid_batches:
                secondNetwork.backprop_with_mini_batch(mini_vilid_batch, eta, epoch=j, pseu=True)

            if test_data:
                n_test = len(test_data)
                print("Epoch {0}: {1} / {2}".format(j, secondNetwork.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1.0 - sigmoid(z))


def rectifier(z):  # f(x)=max(0,x)
    return np.maximum(np.reshape(np.zeros((z.shape)), z.shape), z)


def rectifier_prime(z):  # f'(x)=(max(0,x))'  http://kawahara.ca/what-is-the-derivative-of-relu/
    new_z = np.zeros(np.shape(z))
    for i in range(np.shape(z)[0]):
        if z[i] > 0.:
            new_z[i] = 1
    return new_z


def dropout(probability, shape):
    return np.random.binomial(1, probability, shape)


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def load_data():
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    return (training_data, validation_data, test_data)


def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_data, validation_data, test_data)


def split_by_label(dataset, num_per_label):
    # pick out the same size label from data set
    counter = np.zeros(10)  # for 10 classes
    new_dataset = []
    for i in dataset:
        x, y = i
        if type(y) == np.ndarray:
            y = np.argmax(y)
        if y == 0 and counter[0] < num_per_label:
            new_dataset.append(i)
            counter[0] += 1
            continue
        if y == 1 and counter[1] < num_per_label:
            new_dataset.append(i)
            counter[1] += 1
            continue
        if y == 2 and counter[2] < num_per_label:
            new_dataset.append(i)
            counter[2] += 1
            continue
        if y == 3 and counter[3] < num_per_label:
            new_dataset.append(i)
            counter[3] += 1
            continue
        if y == 4 and counter[4] < num_per_label:
            new_dataset.append(i)
            counter[4] += 1
            continue
        if y == 5 and counter[5] < num_per_label:
            new_dataset.append(i)
            counter[5] += 1
            continue

        if y == 6 and counter[6] < num_per_label:
            new_dataset.append(i)
            counter[6] += 1
            continue

        if y == 7 and counter[7] < num_per_label:
            new_dataset.append(i)
            counter[7] += 1
            continue

        if y == 8 and counter[8] < num_per_label:
            new_dataset.append(i)
            counter[8] += 1

            continue

        if y == 9 and counter[9] < num_per_label:
            new_dataset.append(i)
            counter[9] += 1
            continue

    random.shuffle(new_dataset)
    return new_dataset


if __name__ == "__main__":
    training_data, validation_data, test_data = load_data_wrapper();

    # DROP NN + PL + DAE
    DAE = Network([784, 5000, 784])
    DAE.SGD_DAE(training_data, epochs=10, mini_batch_size=32, eta=3.0, validation_data=validation_data,test_data=test_data)
