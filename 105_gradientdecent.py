"""
use gradient descent to resolve regression question

"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#create data
points_num = 100
vectors = []
# use numpy to generate 100 points ,they from the gaussian distribution
#for each point (x,y) the equation is y = 0.1 * x + 0.2
#so the weight is 0.1, the bias is 0.2
for i in range(points_num):
    x1 = np.random.normal(0.0, 0.6)
    y1 = 0.1 * x1 + 0.2 +np.random.normal(0.0, 0.04)
    vectors.append([x1, y1])
x_data = [v[0] for v in vectors]    #the true data x
y_data = [v[1] for v in vectors]    #the true data y
#figure 1: demo 100 points
plt.plot(x_data, y_data, 'r*', label="Original data")
plt.title("Linear Regression using gradient descent")
plt.legend()
plt.show()

#to modeling the linear regression model
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0)) # init weight
b = tf.Variable(tf.zeros([1]))                     # init bias
y_hat = W * x_data + b                             # the output of model


# to define the loss function
# use mean square error loss function
loss = tf.reduce_mean(tf.square(y_data-y_hat))

#use optimizer to optimize model parameter
optimizer = tf.train.GradientDescentOptimizer(0.5)   # learning_rate = 0.5
train = optimizer.minimize(loss)

#create session
sess = tf.Session()

#initial all data about computation graph
init = tf.global_variables_initializer()
sess.run(init)
# start to train, the step is 20
for step in range(20):
    sess.run(train)
    #print loss, weight, bias about every step
    print("step={0}, loss={1}, [Weight={2}, bias={3}]".format(step,sess.run(loss),sess.run(W),sess.run(b))
            )
#plot all data ,this is the best line 
plt.plot(x_data, y_data, 'r*', label="Original Data")
plt.title("Linear Regression using Gradient Descent")
plt.plot(x_data, sess.run(W) * x_data + sess.run(b), label="model")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.show()




#close session
sess.close()
