import gzip
import pickle
import random
import numpy as np

np.seterr(all='ignore')


class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def backprop_with_mini_batch(self, mini_batch, delta_w, delta_b, eta=1.5, epoch=0):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        delta_w = [w_pri * p(epoch) - nw * (1 - p(epoch)) * learning_rate(eta, epoch) / len(mini_batch) for
                       w_pri, nw in zip(delta_w,nabla_w)]
        delta_b = [b_pri * p(epoch) - nb * (1 - p(epoch)) * learning_rate(eta, epoch) / len(mini_batch) for
                       b_pri, nb in zip(delta_b,nabla_b)]

        self.weights = [w + d_w for w, d_w in zip(self.weights, delta_w)]
        self.biases = [b + d_b for b, d_b in zip(self.biases, delta_b)]

        return delta_w, delta_b

    def backprop(self, x, y):

        delta_nabla_b = [np.zeros(b.shape) for b in self.biases]
        delta_nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward phase
        activation = x
        activations = [activation]
        zs = []
        layer = 0
        for b, w in zip(self.biases, self.weights):
            if layer == 0:
                mask_1 = dropout(0.5, np.shape(activation))
                activation = np.multiply(activation, mask_1)
                z = np.dot(w, activation) + b
                zs.append(z)
                activation = sigmoid(z)

                # TODO use differnet active functions return all WRONG label ! WHY ??
                # activation = rectifier(z)
            else:
                mask_2 = dropout(0.2, np.shape(activation))
                activation = np.multiply(activation, mask_2)
                z = np.dot(w, activation) + b
                zs.append(z)
                activation = sigmoid(z)
            layer += 1
            activations.append(activation)

        # backward phase
        # delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1]) # QuadraticCost
        delta = self.cost_derivative(activations[-1], y)  # cross entropy
        delta_nabla_b[-1] = delta
        drop_1=np.multiply(activations[-2],mask_2)
        delta_nabla_w[-1] = np.dot(delta, drop_1.transpose())


        for l in range(2, self.num_layers):
            z = zs[-l]
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sigmoid_prime(z)
            # TODO different active function in the hidden layer
            # delta = np.dot(self.weights[-l + 1].transpose(), delta) * rectifier_prime(z)
            delta_nabla_b[-l] = delta
            drop_2 = np.multiply(activations[-l - 1],mask_1)
            delta_nabla_w[-l] = np.dot(delta,drop_2.transpose())

        return delta_nabla_b, delta_nabla_w

    def predict(self, a):
        layer = 0
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b
            if layer == 0:
                a = sigmoid(z)
                # TODO different active function in the hidden layer
                # a = rectifier(z) # for hidden layers
            else:
                a = sigmoid(z)  # for output layer
            layer += 1
        return a

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.predict(x)), y) for (x, y) in test_data]
        acc = sum(int(x == y) for (x, y) in test_results)
        error = 1 - acc / len(test_results)
        return error * 100

    def cost_derivative(self, output_activations, y):
        return (output_activations - y)/np.log(10)

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

    def SGD_DropNN(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]
            delta_w = [np.zeros(w.shape) for w in self.weights]
            delta_b = [np.zeros(b.shape) for b in self.biases]
            for mini_batch in mini_batches:
                delta_w, delta_b = self.backprop_with_mini_batch(mini_batch, delta_w, delta_b, eta, j)

            if test_data:
                print(self.evaluate(test_data), "%")


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1.0 - sigmoid(z))


def rectifier(z):  # f(x)=max(0,x)
    return np.maximum(np.reshape(np.zeros(z.shape), z.shape), z)


def rectifier_prime(z):  # f'(x)=(max(0,x))'  http://kawahara.ca/what-is-the-derivative-of-relu/
    new_z = np.zeros(np.shape(z))
    for i in range(np.shape(z)[0]):
        if z[i] > 0.:
            new_z[i] = 1
    return new_z


def dropout(probability, shape):
    return np.random.binomial(1, probability, shape)


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


if __name__ == "__main__":
    training_data, validation_data, test_data = load_data_wrapper()

    loop = 15
    learning = 1.5
    print("eta = ", learning, "epochs = ", loop)

    for i in (100,300):
        training_data_small = split_by_label(training_data, num_per_label=i)
        Neural = Network([784, 5000, 10])
        Neural.SGD_DropNN(training_data_small, epochs=loop, mini_batch_size=32, eta=learning,test_data=test_data)
