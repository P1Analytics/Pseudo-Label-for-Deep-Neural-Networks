
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

    def backprop_with_mini_batch(self, mini_batch, DAE = False):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y, DAE)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        return nabla_w, nabla_b

    def backprop(self, x, y, DAE = False):

        delta_nabla_b = [np.zeros(b.shape) for b in self.biases]
        delta_nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward phase
        activation = x
        activations = [x]
        zs = []
        layer = 0
        for b, w in zip(self.biases, self.weights):
            if layer == 0:
                mask_1 = dropout(0.5, np.shape(activation))
                activation = np.multiply(activation, mask_1)
            else:
                mask_2 = dropout(0.2, np.shape(activation))
                activation = np.multiply(activation, mask_2)
            layer += 1
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward phase
        if DAE:
            y = x
        delta = self.cost_derivative(activations[-1], y)
        delta_nabla_b[-1] = delta
        delta_nabla_w[-1] = np.dot(delta, np.multiply(activations[-2],mask_2).transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sigmoid_prime(z)
            delta_nabla_b[-l] = delta
            delta_nabla_w[-l] = np.dot(delta,np.multiply(activations[-l - 1],mask_1).transpose())

        return delta_nabla_b, delta_nabla_w

    def predict(self, a):
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b
            a = sigmoid(z)
        return a

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.predict(x)), y) for (x, y) in test_data]
        acc = sum(int(x == y) for (x, y) in test_results)
        error = 1 - acc / len(test_results)
        return error*100


    def cost_derivative(self, output_activations, y):
        return (output_activations - y)/ np.log(10)


    def SGD_DAE(self, training_data, epochs, mini_batch_size, eta, validation_data=None, test_data=None):

        print("# DAE unsupervised learning,init the network ")
        validation_results = []
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]
            delta_w = [np.zeros(w.shape) for w in self.weights]
            delta_b = [np.zeros(b.shape) for b in self.biases]
            for mini_batch in mini_batches:
                nabla_w, nabla_b = self.backprop_with_mini_batch(mini_batch, DAE=True)
                delta_w = [w_pri * p(j) - nw * (1 - p(j)) * learning_rate(eta, j) for w_pri, nw in
                           zip(delta_w, nabla_w)]
                delta_b = [b_pri * p(j) - nb * (1 - p(j)) * learning_rate(eta, j) for b_pri, nb in
                           zip(delta_b, nabla_b)]
                self.weights = [w + d_w for w, d_w in zip(self.weights, delta_w)]
                self.biases = [b + d_b for b, d_b in zip(self.biases, delta_b)]


        input_layer = self.sizes[0]
        hidden_layer = self.sizes[1]
        output_layer = 10
        secondNetwork = Network([input_layer,hidden_layer, output_layer])
        secondNetwork.weights[0] = self.weights[0]
        secondNetwork.biases[0] = self.biases[0]

        print("# Fine tuning supervised learning: pseudo label")
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]
            delta_w = [np.zeros(w.shape) for w in secondNetwork.weights]
            delta_b = [np.zeros(b.shape) for b in secondNetwork.biases]
            for mini_batch in mini_batches:
                nabla_w, nabla_b= secondNetwork.backprop_with_mini_batch(mini_batch)
                delta_w = [w_pri * p(j) - nw * (1 - p(j)) * learning_rate(eta, j) for w_pri, nw in
                           zip(delta_w, nabla_w)]
                delta_b = [b_pri * p(j) - nb * (1 - p(j)) * learning_rate(eta, j) for b_pri, nb in
                           zip(delta_b, nabla_b)]
                secondNetwork.weights = [w + d_w for w, d_w in zip(secondNetwork.weights, delta_w)]
                secondNetwork.biases = [b + d_b for b, d_b in zip(secondNetwork.biases, delta_b)]

            # pseudo label session
            labels = []
            for (x, y) in validation_data:
                output = secondNetwork.predict(x)
                labels.append(np.argmax(output))
            validation_results.append(labels)

        validation_array_results = np.array(validation_results)
        pseudo_labels = []
        validation_results = []
        for i in range(validation_array_results.shape[1]):
            a = Counter(validation_array_results[:, i]).most_common(1)[0][0]
            validation_results.append(a)
            pseudo_labels.append(vectorized_result(a))
        X = [x for x, y in validation_data]
        validation_PL_Label = list(zip(X, pseudo_labels))


        print("# train the network again with (training data+ real label) & (validate data + pseudo label)")
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in
                            range(0, len(training_data), mini_batch_size)]

            random.shuffle(validation_PL_Label)
            mini_valid_batch_size = 256
            mini_valid_batches = [validation_PL_Label[k:k + mini_valid_batch_size] for k in
                                  range(0, len(validation_data), mini_valid_batch_size)]

            delta_w = [np.zeros(w.shape) for w in secondNetwork.weights]
            delta_b = [np.zeros(b.shape) for b in secondNetwork.biases]
            for i in range(0, len(mini_batches)):
                nabla_w, nabla_b = secondNetwork.backprop_with_mini_batch(mini_batches[i])
                nabla_w = [x / mini_batch_size for x in nabla_w]
                nabla_b = [x / mini_batch_size for x in nabla_b]
                if i < len(mini_valid_batches):
                    nabla_w_pl, nabla_b_pl = secondNetwork.backprop_with_mini_batch(mini_valid_batches[i])
                    nabla_w_pl = [x / mini_valid_batch_size for x in nabla_w_pl]
                    nabla_b_pl = [x / mini_valid_batch_size for x in nabla_b_pl]
                    nabla_w += nabla_w_pl
                    nabla_b += nabla_b_pl

                delta_w = [w_pri * p(j) - nw * (1 - p(j)) * learning_rate(eta, j) for w_pri, nw in
                               zip(delta_w, nabla_w)]
                delta_b = [b_pri * p(j) - nb * (1 - p(j)) * learning_rate(eta, j) for b_pri, nb in
                               zip(delta_b, nabla_b)]
                secondNetwork.weights = [w + d_w for w, d_w in zip(secondNetwork.weights, delta_w)]
                secondNetwork.biases = [b + d_b for b, d_b in zip(secondNetwork.biases, delta_b)]


            if test_data:
                print(secondNetwork.evaluate(test_data),"%")


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
    print(counter)
    return new_dataset


def learning_rate(eta, epoch):
    return eta * (0.998 ** epoch)


def p(t):
    pf = 0.99
    pi = 0.5
    T = 500
    if t < T:
        return t / T * pf + (1 - t / T) * pi
    else:
        return pf


def alfaCoefficient(currentEpoch, DAE=False):
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

if __name__ == "__main__":
    training_data, validation_data, test_data = load_data_wrapper()

    loop = 15
    learning = 1.5
    print("eta = ", learning, "epochs = ", loop)

    for i in (10,0):
        training_data_small = split_by_label(training_data, num_per_label=i)
        Neural = Network([784, 5000, 784])
        Neural.SGD_DAE(training_data_small, epochs=loop, mini_batch_size=32, eta=learning, validation_data=validation_data,test_data=test_data)
