#################################################################################
### Feed forward three layer, Artificial Neural Network in Google Tensor Flow ###
### Illustrates the learning of XOR logic with two activation functions       ###
### 									      ###	
### Shows the ability to cope with noisy data and still learn patterns        ###
### Usage: {Activation: Sigmoidal, Cost: [ACE, MSE]}                          ### 
###        {Activation: Tanh, Cost: [MSE]}                                    ### 
### 									      ### 
### Copyright Epoch Ventures 2016                                             ### 
#################################################################################


import tensorflow as tf
import numpy as np

def rand01(digit):
	# Add some random noise to bits, but keep always between 0 and 1
	s = abs(np.random.normal(0.0, 0.1))
	if digit == 0:
		noise = digit + s
	else:
		noise = digit - s
	return noise

### Training Examples
### All combinations of XOR

X = [[0, 0],[0, 1],[1, 0],[1, 1]]
Y = [[0], [1], [1], [0]]

# Add some random noise to our inputs. Useful if we use the tanh activiation function

add_noise = np.vectorize(rand01)  
X = add_noise(X)
Y = add_noise(Y)

# Neural Network Parameters

N_STEPS = 200000
N_EPOCH = 5000
N_TRAINING = len(X)

N_INPUT_NODES = 2
N_HIDDEN_NODES = 5
N_OUTPUT_NODES  = 1 

if __name__ == '__main__':

	##############################################################################
	### Create placeholders for variables and define Neural Network structure  ###
	### Feed forward 3 layer, Neural Network.                                  ###
	


	x_ = tf.placeholder(tf.float32, shape=[N_TRAINING, N_INPUT_NODES], name="x-input")
	y_ = tf.placeholder(tf.float32, shape=[N_TRAINING, N_OUTPUT_NODES], name="y-input")

	theta1 = tf.Variable(tf.random_uniform([N_INPUT_NODES,N_HIDDEN_NODES], -1, 1), name="theta1")
	theta2 = tf.Variable(tf.random_uniform([N_HIDDEN_NODES,N_OUTPUT_NODES], -1, 1), name="theta2")

	bias1 = tf.Variable(tf.zeros([N_HIDDEN_NODES]), name="bias1")
	bias2 = tf.Variable(tf.zeros([N_OUTPUT_NODES]), name="bias2")

	### Use a sigmoidal activation function ###

	layer1 = tf.sigmoid(tf.matmul(x_, theta1) + bias1)
	output = tf.sigmoid(tf.matmul(layer1, theta2) + bias2)

	### Use tanh activation function ###

	#layer1 = tf.tanh(tf.matmul(x_, theta1) + bias1)
	#output = tf.tanh(tf.matmul(layer1, theta2) + bias2)

	# Mean Squared Estimate - the simplist cost function (MSE)

	cost = tf.reduce_mean(tf.square(Y - output)) 
	train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
	
	# Average Cross Entropy - better behaviour and learning rate

	#cost = - tf.reduce_mean( (y_ * tf.log(output)) + (1 - y_) * tf.log(1.0 - output)  )
	#train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)



	init = tf.initialize_all_variables()
	sess = tf.Session()
	sess.run(init)


	for i in range(N_STEPS):
		sess.run(train_step, feed_dict={x_: X, y_: Y})
		if i % N_EPOCH == 0:
			print('Batch ', i)
			print('Inference ', sess.run(output, feed_dict={x_: X, y_: Y}))
			print('Cost ', sess.run(cost, feed_dict={x_: X, y_: Y}))


	


	
