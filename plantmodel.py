import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
# Create the data set
x_data = np.load('plantinput.npy')
niose = np.random.normal(0,0.001,x_data.shape)
y_data = np.load('plantoutput.npy')
x__data = np.loadtxt("f.txt")
# Define x,y
x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])
#Create the first layer inside
Weights_L1 = tf.Variable(tf.random_normal([1,10]))
biases_L1 = tf.Variable(tf.zeros([1,10]))
Wx_plus_b_L1 = tf.matmul(x,Weights_L1)+biases_L1
L1 = tf.nn.tanh(Wx_plus_b_L1)
#Creat the second layer inside
Weights_L2 = tf.Variable(tf.random_normal([10,10]))
biases_L2 = tf.Variable(tf.zeros([1,10]))
Wx_plus_b_L2 = tf.matmul(L1,Weights_L2)+biases_L2
L2 = tf.nn.tanh(Wx_plus_b_L2)


#Create the output layer
Weights_L3 = tf.Variable(tf.random_normal([10,1]))
biases_L3 = tf.Variable(tf.zeros([1,1]))
Wx_plus_b_L3 = tf.matmul(L2,Weights_L3)+biases_L3
prediction = tf.nn.tanh(Wx_plus_b_L3)
# Define the error and train
loss = tf.reduce_mean(tf.square(y-prediction))

train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})
    print ("Weights_L1: ", Weights_L1.eval())
    print ("Weights_L2: ", Weights_L2.eval())
    print ("Weights_L3: ", Weights_L3.eval())
    print ("biases_L1: ", biases_L1.eval())
    print ("biases_L2: ", biases_L2.eval())
    print ("biases_L3: ", biases_L3.eval())

    print ("x__data: ",x__data)
    prediction_value = sess.run(prediction,feed_dict={x:x_data})
    #np.save('plantmodeloutput.npy', prediction_value)
    #np.save('plantmodelinput.npy', x_data)

    plt.figure()
    plt.scatter(x_data,y_data)
    plt.plot(x_data,prediction_value,'r',lw=5)
    plt.ylabel("output value")
    plt.xlabel("input value")
    plt.show()
    #y = tf.constant([1., 0., 1., 0.])
    #Error = tf.reduce_mean(tf.square(prediction_value-y_data))
    #print biases_L1.eval()







