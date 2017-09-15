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

    def backprop_with_mini_batch(self, mini_batch, eta=3, epoch=0, pl=False):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y, epoch, pl)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y, epoch=0, pl=False):

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward during training phase
        activation = x
        activations = [x]
        zs = []
        layer = 0
        for b, w in zip(self.biases, self.weights):
            # TODO drop out here(9) drop out making things worse not better
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
        # delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1]) # QuadraticCost
        delta = self.cost_derivative(activations[-1], y)

        if pl:
            delta *= self.alfaCoefficient(epoch)

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sigmoid_prime(z)
            # TODO different active function in the hidden layer, different prime(3)(BP2 in Ref)
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
        # cost_derivative https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network2.py
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

    def SGD_PL(self, training_data, epochs, mini_batch_size, eta, validation_data, test_data=None):

        # want pseudo labels, training network with training data and calculate validation data pseudo labels
        validation_results = []
        for j in range(epochs):
            # training session
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]
            for mini_batch in mini_batches:
                self.backprop_with_mini_batch(mini_batch, eta)

            # pseudo label
            labels = []
            for (x, y) in validation_data:
                output = self.feedforward(x)
                labels.append(np.argmax(output))
            validation_results.append(labels)

        # set pseudo labels with the labels in the highest vote for epoch times
        validation_array_results = np.array(validation_results)
        pseudo_labels = []
        validation_results = []
        for i in range(validation_array_results.shape[1]):
            a = Counter(validation_array_results[:, i]).most_common(1)[0][0]
            validation_results.append(a)
            pseudo_labels.append(vectorized_result(a))
        X = [x for x, y in validation_data]
        validation_PL_Label = list(zip(X, pseudo_labels))

        print("pseudo label")
        for j in range(epochs):
            # training the network with validation data with pseudo labels
            random.shuffle(validation_PL_Label)
            mini_valid_batch_size = 256
            mini_valid_batches = [validation_PL_Label[k:k + mini_valid_batch_size] for k in
                                  range(0, len(validation_data), mini_valid_batch_size)]

            for mini_valid_batch in mini_valid_batches:
                self.backprop_with_mini_batch(mini_valid_batch, eta, epoch=j, pl=True)

            # evaluate with test data set
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), len(test_data)))
            else:
                print("Epoch {0} complete".format(j))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1.0 - sigmoid(z))


def rectifier(z):  # f(x)=max(0,x)
    return np.maximum(np.reshape(np.zeros((z.shape)), z.shape), z)


def rectifier_prime(z):  # f'(x)=(max(0,x))'  http://kawahara.ca/what-is-the-derivative-of-relu/
    print(type(z))
    print(z)
    new_z = np.zeros(np.shape(z))
    print(new_z)
    for i in range(np.shape(z)[0]):
        if z[i] > 0.:
            new_z[i] = 1
    print(new_z)
    return new_z


def dropout(probability, shape):
    return np.random.binomial(1, probability, shape)


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    return (training_data, validation_data, test_data)


def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
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
    print(counter)
    return new_dataset


if __name__ == "__main__":
    training_data, validation_data, test_data = load_data_wrapper();

    # DROP NN + PL
    PseudoLabel = Network([784, 5000, 10])
    PseudoLabel.SGD_PL(training_data, epochs=10, mini_batch_size=32, eta=3.0, validation_data=validation_data,test_data=test_data)
