import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
# Create the data set
y_data = np.load('plantmodelinput.npy')
#niose = np.random.normal(0,0.001,x_data.shape)
x_data = np.load('plantmodeloutput.npy')
# Define x,y
x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])
#Create the first layer inside
#Weights_L1 = tf.Variable(tf.random_normal([1,10]))
Weights_L1 = tf.Variable([[1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]])

biases_L1 = tf.Variable(tf.zeros([1,10]))
Wx_plus_b_L1 = tf.matmul(x,Weights_L1)+biases_L1
L1 = tf.nn.tanh(Wx_plus_b_L1)
#Creat the second layer inside
Weights_L2 = tf.Variable(tf.random_normal([10,10]))
biases_L2 = tf.Variable(tf.zeros([1,10]))
Wx_plus_b_L2 = tf.matmul(L1,Weights_L2)+biases_L2
L2 = tf.nn.tanh(Wx_plus_b_L2)
#Creat the third layer inside
Weights_L3 = tf.Variable(tf.random_normal([10,10]))
biases_L3 = tf.Variable(tf.zeros([1,10]))
Wx_plus_b_L3 = tf.matmul(L2,Weights_L3)+biases_L3
L3 = tf.nn.tanh(Wx_plus_b_L3)


#Create the output layer
Weights_L4 = tf.Variable(tf.random_normal([10,1]))
biases_L4 = tf.Variable(tf.zeros([1,1]))
Wx_plus_b_L4 = tf.matmul(L3,Weights_L4)+biases_L4
prediction = tf.nn.tanh(Wx_plus_b_L4)
# Define the error and train
loss = tf.reduce_mean(tf.square(y-prediction))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    for step in range(50000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})
    #save_path = saver.save(sess,"my_net/save_inversemodel.ckpt")
    #print ("Save to path: ",save_path)
    print ("Weights_L1: ",Weights_L1.eval())
    print ("Weights_L2: ",Weights_L2.eval())
    print ("Weights_L3: ", Weights_L3.eval())
    print ("Weights_L4: ", Weights_L4.eval())
    print ("Biase_L1:",biases_L1.eval())
    print ("Biase_L2:", biases_L2.eval())
    print ("Biase_L3:", biases_L3.eval())
    print ("Biase_L4:", biases_L4.eval())

    #W1=tf.Variable(np.arange(10).reshape(()))
    prediction_value = sess.run(prediction,feed_dict={x:x_data})
    pp = sess.run(prediction,feed_dict={x:[[0.7]]})
    print ("pp:",pp)
    print Weights_L1.eval()
    plt.figure()
    #plt.scatter(x_data,y_data)
    plt.plot(x_data,prediction_value,'r',lw=5)
    plt.xlabel("input value")
    plt.ylabel("output value")
    plt.show()
    #y = tf.constant([1., 0., 1., 0.])
    #Error = tf.reduce_mean(tf.square(prediction_value-y_data))
    #print biases_L1.eval()









