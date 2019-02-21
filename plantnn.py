import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
#create the data set
x_data = np.linspace(-0.8,0.8,200)[:,np.newaxis]
# Define x,y
x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])
#Create the first layer inside
Weights_L1 = tf.Variable([[3.,2.,-1.,3.,1.]])

biases_L1 = tf.Variable([[-0.5,-0.5,-0.5,-0.5,-0.5]])

Wx_plus_b_L1 = tf.matmul(x,Weights_L1)+biases_L1
L1 = tf.nn.tanh(Wx_plus_b_L1)
#Creat the second layer inside
Weights_L2 = tf.Variable([[1.,2.,-1.,1.,1.],[1.,2.,-1.,1.,1.],[1.,2.,-1.,1.,1.],[1.,2.,-1.,1.,1.],[1.,2.,-1.,1.,1.]])
biases_L2 = tf.Variable([-0.5,-0.5,-0.5,-0.5,-0.5])
Wx_plus_b_L2 = tf.matmul(L1,Weights_L2)+biases_L2
L2= tf.nn.tanh(Wx_plus_b_L2)

#Create the output layer
Weights_L3 = tf.Variable([[1.],[-1.],[1.],[1.],[1.]])
biases_L3 = tf.Variable(tf.zeros([1,1]))
Wx_plus_b_L3 = tf.matmul(L2,Weights_L3)+biases_L3
prediction = tf.nn.tanh(Wx_plus_b_L3)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    prediction_value = sess.run(prediction, feed_dict={x: x_data})
    feed_dict={x: x_data}
    print prediction_value
    print biases_L1.eval()
    np.savetxt("f.txt", x_data)
    plt.figure()

    plt.plot(x_data, prediction_value, 'r', lw=5)
    #np.save('plantoutput.npy',prediction_value)
    #np.save('plantinput.npy',x_data)
    #save_path = saver.save(sess, 'my_net/save_net.ckpt')
    print x_data



