import tensorflow as tf;
import numpy as np;
import matplotlib.pyplot as plt;

with tf.variable_scope('inputs'):
    inputs = tf.Variable([[1.2]])

with tf.variable_scope('inversecontroller'):
    #a1 = tf.get_variable(name='a1', shape=[1], initializer=tf.constant_initializer(1))
    a2 = tf.Variable([[0.1,0.2,0.3]])
    Inv_W_1 = tf.Variable([[1.1580741, 1.7268586, 1.654552 , 1.5077028, 1.7014261, 1.7377229,
        1.5080876, 1.9330828, 1.9578081, 2.0615253]], dtype=tf.float32)
    Inv_W_2 = tf.Variable([[ 1.3911813 ,  0.1363242 ,  0.70704216, -1.2127514 ,  0.7186159 ,
        -0.9024186 , -2.319011  , -0.13536568, -0.5028003 , -0.4321035 ],
       [ 1.1694618 , -0.70591855, -1.752004  ,  0.26817766,  1.6974694 ,
         1.5954881 , -0.40087107, -0.2221693 ,  0.71480256, -0.07164146],
       [ 0.21258005,  0.32591024,  0.272139  ,  0.1423478 ,  0.34201849,
         0.50433207, -0.2672579 ,  0.3664661 , -0.8034817 ,  1.1268622 ],
       [-2.1289794 , -0.42296812,  0.89229786,  0.008174  ,  0.90840966,
         0.61191034, -1.0584809 ,  0.7003018 , -0.6102243 ,  0.6262219 ],
       [-0.22779815, -0.30569106, -0.11219057,  0.20509996, -0.02950988,
        -1.622923  , -1.565053  , -2.591839  ,  1.2128931 ,  1.4302274 ],
       [-0.86586094, -0.45890746, -1.2246416 , -0.27648723, -1.7188501 ,
         1.4388239 , -0.05658441, -1.7033242 ,  0.3979581 ,  0.7194326 ],
       [-0.01849977,  0.23863873, -1.645844  , -0.4246534 ,  0.08048869,
        -0.20093593,  0.46116394, -0.51585937,  0.5670062 ,  1.3421643 ],
       [-1.2095798 , -1.2562609 ,  0.5453443 , -0.46112052,  0.5829385 ,
         0.05110769, -0.33601695, -1.4043329 ,  1.0665156 ,  1.1015385 ],
       [ 0.21057084,  0.8443675 , -1.0267714 , -1.7604141 , -0.8248748 ,
        -1.1075494 ,  0.8253879 ,  1.0342883 ,  0.95668733, -0.2765752 ],
       [-1.6554635 , -0.9768497 , -0.98307306, -0.54735464,  2.6146343 ,
         1.4842585 , -0.4740297 , -0.66966873, -0.695678  , -0.08828938]],
      dtype=tf.float32)
    Inv_W_3 = tf.Variable([[ 1.6999341 ,  0.92402375, -0.92667025, -1.1446512 , -0.00378341,
        -0.03216932, -1.6367807 , -0.7877717 ,  1.970298  , -0.14031744],
       [ 0.3414317 ,  1.4773309 ,  1.2961171 ,  0.49582782,  0.93006897,
        -0.02726466, -0.17644075,  0.21326661,  0.67256963,  0.34910762],
       [-0.8873928 , -1.1932889 ,  0.31076786,  0.17346576, -1.9615713 ,
        -0.6671983 ,  1.3845159 , -0.73394966,  0.30507407,  2.5366788 ],
       [ 1.2618247 ,  2.8990011 , -0.571117  ,  0.85202587, -0.38515508,
        -0.25478667,  0.39350832, -0.7742776 ,  0.20879588,  0.51678026],
       [-2.0495465 , -0.11989083, -1.3808402 , -0.40027776, -0.14559726,
         2.5538504 ,  0.11967637, -1.7548931 , -0.67294407, -0.41921553],
       [ 1.178299  , -1.7647405 , -0.08328224, -0.13421215,  1.87404   ,
        -0.61763614,  2.2457454 , -1.1770662 ,  1.6074642 , -0.4802277 ],
       [-0.6967843 , -1.0250287 ,  1.0981714 ,  1.7093493 , -1.1308763 ,
         0.05848604,  1.695454  , -0.93184584,  1.6293519 , -0.883265  ],
       [ 1.7211001 ,  2.4373384 ,  0.8712364 , -0.55630773,  0.45449784,
        -1.6631523 , -0.04578654,  0.5889159 , -0.13199738,  0.05945062],
       [ 1.9091734 , -0.14697188, -0.6102854 , -1.0905341 ,  1.9792467 ,
        -0.40027824,  1.305854  ,  0.31907588,  1.1067197 ,  0.19986139],
       [ 0.985388  ,  1.0315245 ,  0.82813853, -0.02177558,  0.7098702 ,
        -0.03443367,  1.0163809 , -2.0547793 , -0.6113637 ,  0.00942856]],
      dtype=tf.float32)
    Inv_W_4 = tf.Variable([[ 1.7768415 ],
       [-1.412572  ],
       [-0.31559855],
       [-0.8275408 ],
       [-0.02980363],
       [-0.62836623],
       [ 0.5416207 ],
       [-0.19016041],
       [ 0.74000823],
       [ 0.03933945]], dtype=tf.float32)
    Inv_B_1 = tf.Variable([[-0.13765404, -0.48935065, -0.2565536 ,  0.37846512,  1.1434486 ,
         1.0724592 ,  0.00232211,  1.2896616 , -0.5291015 ,  1.3654039 ]],
      dtype=tf.float32)
    Inv_B_2 = tf.Variable([[ 0.3925795 , -0.09769101,  0.09652358, -0.17113045,  0.08343767,
         0.56631786,  0.10505706, -0.11701151, -0.18835145, -0.02782093]],
      dtype=tf.float32)
    Inv_B_3 = tf.Variable([[ 0.7298928 ,  0.04804013,  0.01714226, -0.01983036, -0.0033228 ,
         0.00141146,  0.17847781,  0.01745849, -0.00341618, -0.00119484]],
      dtype=tf.float32)
    Inv_B_4 = tf.Variable([[0.20228659]], dtype=tf.float32)

with tf.variable_scope('PlantModel'):
    #a3 = tf.get_variable(name='a1', shape=[1], initializer=tf.constant_initializer(1))
    a4 = tf.Variable([[0.2],[0.2],[0.2]])
    plantModel_W_1 = tf.Variable([[0.3792335, 0.35829672, 1.8613012, 0.40170887, 1.7167636,
                                        -0.35253206, 0.78094757, -0.80424964, -0.11869979, -1.3642256]],
                                      dtype=tf.float32)
    plantModel_W_2 = tf.Variable([[0.70239997, -1.1486238, 1.6549584, 0.79648674, -0.05574559,
                                        -1.9403614, 1.1603005, -0.4404556, 1.1639857, -0.26900208],
                                       [-2.6695077, -1.3375647, 1.1451317, -0.15974224, -0.08786236,
                                        -0.83024967, -0.20036784, -0.35057506, -1.0779035, 0.5238612],
                                       [0.22545433, -0.99177366, 0.30867136, 1.0167537, 1.0065266,
                                        -0.4638363, -0.02105663, -0.22011888, 0.5869994, -0.41912302],
                                       [-1.0926181, 0.78821963, 0.849451, -0.33960345, 0.91336304,
                                        -0.00373224, 0.0801261, -0.9839265, 2.1058278, 1.3575717],
                                       [-0.38540822, -1.1746596, 0.20238727, 0.38626465, -0.19961777,
                                        0.250466, -0.37104145, 1.2681012, 0.8271107, 0.8884576],
                                       [-1.0669233, 0.0430838, 1.1665618, 1.7501967, -0.228664,
                                        0.82348716, 1.5810498, -0.06235954, -0.00382873, -0.19297586],
                                       [-0.811185, -0.19207928, -1.5038719, 0.7850371, 0.7584014,
                                        0.9704245, -0.07090363, 0.28755358, 0.6140961, 1.5702819],
                                       [-0.04252477, 0.8596387, -0.29119262, -1.8236802, 0.84500647,
                                        0.39092264, 0.6694976, -1.7215143, -1.1308441, -1.1969627],
                                       [0.24005064, 0.46068114, -0.5185555, -0.46952865, -1.2677593,
                                        -0.82801545, 0.31609726, -0.503021, -0.04563146, -0.76434106],
                                       [-1.1900413, -1.3436716, -0.388409, -0.66014576, -0.38040343,
                                        0.5255122, 0.54970694, -1.0988154, 0.68703175, 0.13286124]],
                                      dtype=tf.float32)
    plantModel_W_3 = tf.Variable([[0.5183688],
                                       [0.10267338],
                                       [-0.23946704],
                                       [-0.38087413],
                                       [-0.36236957],
                                       [-1.0811349],
                                       [0.0159929],
                                       [-0.95006543],
                                       [-0.2606715],
                                       [2.4560564]], dtype=tf.float32)
    plantModel_Biase_1 = tf.Variable([[-0.10316379, -0.0959843, 0.1492138, -0.11809526, -0.3223964,
                                            0.07068261, -0.23496705, 0.2905244, 0.0230111, 0.05264661]],
                                          dtype=tf.float32)
    plantModel_Biase_2 = tf.Variable([[-0.02366861, -0.00646496, 0.01593119, -0.0377991, 0.02400768,
                                            0.0919067, -0.00338848, 0.01130115, 0.02866236, -0.17934462]],
                                          dtype=tf.float32)
    plantModel_Biase_3 = tf.Variable([[-0.01799532]], dtype=tf.float32)








with tf.variable_scope('Plant'):
    plant_W_1 = tf.Variable([[3., 2., -1., 3., 1.]])
    plant_W_2 = tf.Variable(
        [[1., 2., -1., 1., 1.], [1., 2., -1., 1., 1.], [1., 2., -1., 1., 1.], [1., 2., -1., 1., 1.],
         [1., 2., -1., 1., 1.]])
    plant_W_3 = tf.Variable([[1.], [-1.], [1.], [1.], [1.]])
    plant_Biase_1 = tf.Variable([[-0.5, -0.5, -0.5, -0.5, -0.5]])
    plant_Biase_2 = tf.Variable([-0.5, -0.5, -0.5, -0.5, -0.5])
    plant_Biase_3 = tf.Variable(tf.zeros([1, 1]))

with tf.variable_scope('Feedback'):
    # Create the first layer inside
    Feedback_Weights_L1 = tf.Variable(tf.random_normal([1, 10]))
    Feedback_biases_L1 = tf.Variable(tf.zeros([1, 10]))

    # Creat the second layer inside
    Feedback_Weights_L2 = tf.Variable(tf.random_normal([10, 10]))
    Feedback_biases_L2 = tf.Variable(tf.zeros([1, 10]))


    # Create the output layer
    Feedback_Weights_L3 = tf.Variable(tf.random_normal([10, 1]))
    Feedback_biases_L3 = tf.Variable(tf.zeros([1, 1]))


# Define x,y
x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])
x_data = np.load('PlantPMDifference.npy')
x_data = [[0.07249055],[0.07010385],[0.060601],[0.04538092],[0.03551331],[0.0219498],[-0.01048297],[-0.02815592]]
y_data = [[0.1],[0.2],[0.22],[0.25],[0.27],[0.3],[0.4],[0.5]]
x_data = [[0.0219498],[0.07249055]]
y_data = [[0.3],[0.1]]
Feedback_L1_Out=tf.nn.tanh(tf.matmul(x,Feedback_Weights_L1)+Feedback_biases_L1)
Feedback_L2_Out=tf.nn.tanh(tf.matmul(Feedback_L1_Out,Feedback_Weights_L2)+Feedback_biases_L2)
Feedback_Out=tf.nn.tanh(tf.matmul(Feedback_L2_Out,Feedback_Weights_L3)+Feedback_biases_L3)

inverseM_L1_Out=tf.nn.tanh(tf.matmul(y-Feedback_Out,Inv_W_1)+Inv_B_1)
inverseM_L2_Out=tf.nn.tanh(tf.matmul(inverseM_L1_Out,Inv_W_2)+Inv_B_2)
inverseM_L3_Out=tf.nn.tanh(tf.matmul(inverseM_L2_Out,Inv_W_3)+Inv_B_3)
inverseM_L4_Out = tf.nn.tanh(tf.matmul(inverseM_L3_Out, Inv_W_4)+Inv_B_4)
inverseM_out = inverseM_L4_Out
#plant network
plant_L1_Out = tf.nn.tanh(tf.matmul(inverseM_out,plant_W_1)+plant_Biase_1)
plant_L2_Out = tf.nn.tanh(tf.matmul(plant_L1_Out,plant_W_2)+plant_Biase_2)
plant_L3_Out = tf.nn.tanh(tf.matmul(plant_L2_Out, plant_W_3)+plant_Biase_3)
plant_Out = plant_L3_Out
#plantModel network
plantModel_L1_Out = tf.nn.tanh(tf.matmul(inverseM_out, plantModel_W_1) + plantModel_Biase_1)
plantModel_L2_Out = tf.nn.tanh(tf.matmul(plantModel_L1_Out, plantModel_W_2) + plantModel_Biase_2)
plantModel_L3_Out = tf.nn.tanh(tf.matmul(plantModel_L2_Out, plantModel_W_3) + plantModel_Biase_3)
plantModel_Out = plantModel_L3_Out
# Compute the difference of the plant output and the plantmodel output
feedback=plant_Out-plantModel_Out
PlantOut = {}
for i in range(50):
    Feedback_L1_Out = tf.nn.tanh(tf.matmul(feedback, Feedback_Weights_L1) + Feedback_biases_L1)
    Feedback_L2_Out = tf.nn.tanh(tf.matmul(Feedback_L1_Out, Feedback_Weights_L2) + Feedback_biases_L2)
    Feedback_Out = tf.nn.tanh(tf.matmul(Feedback_L2_Out, Feedback_Weights_L3) + Feedback_biases_L3)

    inverseM_L1_Out = tf.nn.tanh(tf.matmul(y - Feedback_Out, Inv_W_1) + Inv_B_1)
    inverseM_L2_Out = tf.nn.tanh(tf.matmul(inverseM_L1_Out, Inv_W_2) + Inv_B_2)
    inverseM_L3_Out = tf.nn.tanh(tf.matmul(inverseM_L2_Out, Inv_W_3) + Inv_B_3)
    inverseM_L4_Out = tf.nn.tanh(tf.matmul(inverseM_L3_Out, Inv_W_4) + Inv_B_4)
    inverseM_out = inverseM_L4_Out
    # plant network
    plant_L1_Out = tf.nn.tanh(tf.matmul(inverseM_out, plant_W_1) + plant_Biase_1)
    plant_L2_Out = tf.nn.tanh(tf.matmul(plant_L1_Out, plant_W_2) + plant_Biase_2)
    plant_L3_Out = tf.nn.tanh(tf.matmul(plant_L2_Out, plant_W_3) + plant_Biase_3)
    plant_Out = plant_L3_Out
    # plantModel network
    plantModel_L1_Out = tf.nn.tanh(tf.matmul(inverseM_out, plantModel_W_1) + plantModel_Biase_1)
    plantModel_L2_Out = tf.nn.tanh(tf.matmul(plantModel_L1_Out, plantModel_W_2) + plantModel_Biase_2)
    plantModel_L3_Out = tf.nn.tanh(tf.matmul(plantModel_L2_Out, plantModel_W_3) + plantModel_Biase_3)
    plantModel_Out = plantModel_L3_Out
    # Compute the difference of the plant output and the plantmodel output
    feedback = plant_Out - plantModel_Out
    PlantOut[i]=plant_Out



loss = tf.reduce_mean(tf.square(plant_Out-y))


optimizer = tf.train.AdamOptimizer(1e-3)

output_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Feedback')
train_step = optimizer.minimize(loss,var_list = output_vars)



with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for step in range(2000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})
    #print a1.name
    prediction_value = sess.run(PlantOut, feed_dict={x:x_data,y:y_data})
    prediction_value1 = sess.run(plant_Out, feed_dict={x: x_data, y: y_data})

    print prediction_value
    print prediction_value1
    print a4.eval()
    print a2.name
    #print a3.name
    print a4.name
    print loss.eval()

