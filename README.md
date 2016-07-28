# Learning XOR with a Neural Network in Google Tensorflow

## Introduction

The code represents a three layer neural network with a configurable number of nodes. In the input layer we have 2 inputs, one for each of the input bits and the hidden layer is configurable to almost any size with the parameter: N_HIDDEN_NODES. 

There are two activation functions that can be chosen from. The softmax which output values in the range 0 to 1 and the tanh function which returns values between -1 and 1. To avoid the use of additional bias nodes, we code the inputs as follows: AX+b, where A is the weight matrix goverining the strength of the neural network connections, X are in the inputs and b is the bias. Bias inputs are handled by Tensorflow in this way as a variable, rather than having to define additional nodes.

XOR may seem like a simple operation to learn, but its history is rich. In fact the pre-cursor to the neural network, the perceptron was not able to learn this pattern and it has been argued that Artificial Intelligence research was pushed back. See more about the perceptron at https://en.wikipedia.org/wiki/Perceptron 

## Example Running the Neural Network

Simply ensure that tensorflow is in your environment and then run:

`python xor.py`

## How does Tensor flow work?

### Create the training set for XOR

In order to learn anything, we need a training set. In the case of XOR, we only have two binary inputs, which means that we only have four cases to consider. The XOR function looks like this:

| X0 | X1 | Y |
---- |--- |---|
0    | 0  | 0 |
0    | 1  | 1 |
1    | 0  | 1 |
1    | 1  | 0 |


We store these combinations as our reference training set:

* `X = [[0, 0],[0, 1],[1, 0],[1, 1]]`
* `Y = [[0], [1], [1], [0]]`

### Specifying the neural network structure

* `N_INPUT_NODES = 2`
* `N_HIDDEN_NODES = 5`
* `N_OUTPUT_NODES  = 1`

### Variables? Placeholders?

In tensor flow, we can specify placeholders to take values that can differ. This is ideal to handle the training data and the ever changing neural network weights that are updated with the learning process. The following illustrates how we set-up a placeholder:

`x_ = tf.placeholder(tf.float32, shape=[N_TRAINING, N_INPUT_NODES], name="x-input")`

This line creates a place holder named `x_` which stores a 2 dimensional structure corresponding to the number of training samples and the the number of input nodes. This is another way of stating we want a `N_TRAINING * N_INPUT_NODES` array that can hold our training example inputs corresponding to our `X` training vector. Similarly we can also define the theta values which represent the weights between each of our layers. 

`theta1 = tf.Variable(tf.random_uniform([N_INPUT_NODES,N_HIDDEN_NODES], -1, 1), name="theta1")`

Theta1 corresponds to a two dimensional array or Matrix whose elements correspond to the strength of the interaction between nodes. For theta1, it corresponds to the weights between input layer and the hidden layer and for theta2 corresponds to the weights between the hidden layer and the output layer. To initialize we specify random element weights between -1 and 1.

We initialize the bias to zero as follows and we will let Tensorflow update these biases (bias1 and bias2):

`bias1 = tf.Variable(tf.zeros([N_HIDDEN_NODES]), name="bias1")`

### Activation functions

Nodes are similar to those in the brain - well not quite, but they take the inputs from all nodes in the layer before them and sum them, and perform a function upon that sum and produce an output. The strength of the inputs into any given node depends on the weights that have been learnt and the strength of the inputs. Remember we had the matrix structure called theta1 and our inputs X, we that means we can simply peform this sum for each node as follows using the matrix multiplicaion `theta1 * x_`.

There are choices of activation function, that perform differently on the same inputs. One common function that returns values between 0 and 1 is called sigmoid function and the beauty of such a function is that no matter the size of the input, output values are always constrained between 0 and 1. Another activiation function is: the tanh function which outputs a range between -1 to 1. It consists of computing the hyperbolic sin and cos function as follows sinh/cosh. 

### Forward propogation and Back propogation

Feed forward neural networks get their name from their behaviour. The idea is that any pattern presented at the input nodes produces a pattern at the output, therefore signals are propogated from the input nodes, through hidden layers (however many) to the output layer. This forward passing is called inference - and once we have trained the NN we can see useful output at the nodes. 

The NN needs to be trained before it can do something useful. The term back propogation is often employed to describe how we perform the training. Training works by figuring out the error at the output and update the weights between the nodes, according to this error. One of the breakthroughs of the backpropogation technique is how that error is pushed towards the input layer in a way that it means that earlier nodes can also update their weights based on errors at a later layer.

### Training the Neural Network

In order to determine whether the neural network has learnt it's input patterns, a cost function is employed to evaluate how different the predictions from the neural network are compared with the actual training data (Y). The cost function can take many forms. Perhaps the simplist and most robust function is the MSE or the mean squared estimation. It literally takes the difference between the predicted output and the training value for that set of inputs. 

`cost = tf.reduce_mean(tf.square(Y - output))` Mean Squared Estimate

Another popular cost function is the Average Cross Entropy. We wont discuss it here, but you can use see this cost here:

`cost = - tf.reduce_mean( (y_ * tf.log(output)) + (1 - y_) * tf.log(1.0 - output) )` 

## Experiments with a neural network

### Change the number of hidden nodes

The input nodes do not have to match the number of nodes in the hidden. Experiment by changing the number of hidden nodes.

* `N_HIDDEN_NODES = 5` Experiment with the number of hidden nodes, by just changing this one parameter. You don't need to make any changes elsewhere in the code.






