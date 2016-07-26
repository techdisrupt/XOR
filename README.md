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

`X = [[0, 0],[0, 1],[1, 0],[1, 1]]
Y = [[0], [1], [1], [0]]`

### Specifying the neural network structure

`N_INPUT_NODES = 2
N_HIDDEN_NODES = 5
N_OUTPUT_NODES  = 1`

### Variables? Placeholders?






## Experiments with a neural network

Comment out the functionality which you wish to ignore and uncomment that which you wish to see.


