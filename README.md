# Learning XOR with a Neural Network in Google Tensorflow

## Introduction

The code represents a three layer neural network with a configurable number of nodes. In the input layer we have 2 inputs, one for each of the input bits and the hidden layer is configurable to almost any size with the parameter: N_HIDDEN_NODES. 

There are two activation functions that can be chosen from. The softmax which output values in the range 0 to 1 and the tanh function which returns values between -1 and 1. To avoid the use of additional bias nodes, we code the inputs as follows: AX+b, where A is the weight matrix goverining the strength of the neural network connections, X are in the inputs and b is the bias. Bias inputs are handled by Tensorflow in this way as a variable, rather than having to define additional nodes.

## Example Running the Neural Network

Simply ensure that tensorflow is in your environment and then run:

`python xor.py`


