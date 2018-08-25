import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(-5, 5 , 200)
y_relu = tf.nn.relu(x)
y_sigmoid = tf.nn.sigmoid(x)
y_tanh = tf.nn.tanh(x)
#y_softplus = tf.nn.softplus(x)
y_softmax = tf.nn.softmax(x)
with tf.Session() as sess:
    y_relu, y_sigmoid, y_tanh, y_softmax = sess.run([y_relu, y_sigmoid, 
                                                y_tanh, y_softmax])
    plt.figure(1, figsize=(8,6))
    plt.subplot(221)
    plt.plot(x, y_relu, 'r', 'relu')
    plt.ylim((-1, 5))
    plt.legend(loc="best")
    plt.title("relu")


    plt.subplot(222)
    plt.plot(x, y_sigmoid, 'r', 'sigmoid')
    plt.ylim((-0.2, 1.2))
    plt.legend(loc="best")
    plt.title("sigmoid")

    plt.subplot(223)
    plt.plot(x, y_tanh, 'r', 'tanh')
    plt.ylim((-1.2, 1.2))
    plt.legend(loc="best")
    plt.title("tanh")

    plt.subplot(224)
    plt.plot(x, y_softmax, 'r', 'softmax')
    plt.ylim((0, 1))
    plt.legend(loc="best")
    plt.title("softmax")
    """
    plt.subplot(155)
    plt.plot(x, y_softmax, 'r', 'softmax')
    plt.ylim((-2, 2))
    plt.legend(loc="best")
    """
    plt.show()


