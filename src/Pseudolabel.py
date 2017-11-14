# This code runs parallelly a Neural Network with and without Pseudolabeling

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("", one_hot=True)

# Neural Network parameters
iteration_list = np.zeros(1)
neural_network_accuracy_list = np.zeros(1)
pseudo_label_accuarcy_list = np.zeros(1)
neural_network_accuracy = 0
pseudo_label_accuarcy = 0

learningRate = 1.5
trainingEpochs = 500

inputN = 784
hiddenN = 392
hiddenN2 = 89
outputN = 10

batchSize = 100
PLbatchSize = 256

iteration = 0
cPL = 0
T1 = 100
T2 = 600
a = 0.
af = 3.

print("HiddenLayer1:", hiddenN, "HiddenLayer2:", hiddenN2, )

x = tf.placeholder("float", [None, inputN])
y = tf.placeholder("float", [None, outputN])
PLx = tf.placeholder("float", [None, inputN])
PLy = tf.placeholder("float", [None, outputN])
alpha = tf.placeholder("float", )

plt.clf()


def NN(x, w, b):
    # Hidden layer 1
    HL = tf.add(tf.matmul(x, w['h1']), b['b1'])
    HL = tf.nn.sigmoid(HL)  # sigmoid(HL)

    # Hidden layer 2
    HL2 = tf.add(tf.matmul(HL, w['h2']), b['b2'])
    HL2 = tf.nn.sigmoid(HL2)  # sigmoid(HL)

    # Output layer
    out_layer = tf.matmul(HL2, w['out']) + b['out']

    return out_layer


# initialize weights and biases

weightsNN = {
    'h1': tf.Variable(tf.random_normal([inputN, hiddenN])),
    'h2': tf.Variable(tf.random_normal([hiddenN, hiddenN2])),
    'out': tf.Variable(tf.random_normal([hiddenN2, outputN]))
}
biasesNN = {
    'b1': tf.Variable(tf.random_normal([hiddenN])),
    'b2': tf.Variable(tf.random_normal([hiddenN2])),
    'out': tf.Variable(tf.random_normal([outputN]))
}
weightsPL = {
    'h1': tf.Variable(tf.random_normal([inputN, hiddenN])),
    'h2': tf.Variable(tf.random_normal([hiddenN, hiddenN2])),
    'out': tf.Variable(tf.random_normal([hiddenN2, outputN]))
}
biasesPL = {
    'b1': tf.Variable(tf.random_normal([hiddenN])),
    'b2': tf.Variable(tf.random_normal([hiddenN2])),
    'out': tf.Variable(tf.random_normal([outputN]))
}

predNN = NN(x, weightsNN, biasesNN)
predPL = NN(x, weightsPL, biasesPL)
predPL1 = NN(PLx, weightsPL, biasesPL)

costNN = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predNN,
                                                                labels=y))

costPL = tf.add(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predPL,
                                                                       labels=y)),
                (alpha * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predPL1,
                                                                                labels=PLy))))

# Gradient Descent
optimizerNN = tf.train.GradientDescentOptimizer(learningRate).minimize(costNN)
optimizerPL = tf.train.GradientDescentOptimizer(learningRate).minimize(costPL)

# Initializing the variables
init = tf.global_variables_initializer()


# Launch the graph
def accuracytestNN():
    # Test model
    correct_prediction = tf.equal(tf.argmax(predNN, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # To keep sizes compatible with model
    return accuracy.eval({x: mnist.test.images, y: mnist.test.labels})


def accuracytestPL():
    # Test model
    correct_prediction = tf.equal(tf.argmax(predPL, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # To keep sizes compatible with model
    return accuracy.eval({x: mnist.test.images, y: mnist.test.labels})


with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(trainingEpochs):
        avg_costNN = 0.
        avg_costPL = 0.
        total_batch = int(3000 / batchSize)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batchSize)

            _, cNN = sess.run([optimizerNN, costNN], feed_dict={x: batch_x,
                                                                y: batch_y})
            if iteration > T1:
                a = ((iteration - T1) / (T2 - T1)) * af
                if iteration > T2:
                    a = af

            # Pseudolabel
            batch_xpred, _ = mnist.train.next_batch(PLbatchSize)
            batch_ypred = sess.run([predPL], feed_dict={x: batch_xpred})
            batch_ypred = batch_ypred[0]
            batch_ypred = batch_ypred.argmax(1)
            batch_ypre = np.zeros((PLbatchSize, 10))
            for ii in range(PLbatchSize):
                batch_ypre[ii, batch_ypred[ii]] = 1

            _, cPL = sess.run([optimizerPL, costPL], feed_dict={x: batch_x,
                                                                y: batch_y,
                                                                PLx: batch_xpred,
                                                                PLy: batch_ypre,
                                                                alpha: a})
            iteration = iteration + 1
            # Compute average loss
            avg_costNN += cNN / total_batch
            avg_costPL += cPL / total_batch

        if iteration % 100 == 0:
            print('t=', iteration)
            neural_network_accuracy = accuracytestNN()
            pseudo_label_accuarcy = accuracytestPL()

            iteration_list = np.append(iteration_list, iteration)
            neural_network_accuracy_list = np.append(neural_network_accuracy_list, neural_network_accuracy)
            pseudo_label_accuarcy_list = np.append(pseudo_label_accuarcy_list, pseudo_label_accuarcy)

        plt.plot(iteration_list, pseudo_label_accuarcy_list, iteration_list, neural_network_accuracy_list, 'r--')
    print("Optimization Finished!")
    print("Neural Network accuracy:", accuracytestNN())
    print("+PL:", accuracytestPL())
    plt.show()
