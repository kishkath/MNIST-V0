# MNIST-V0
A CNN architecture desired to achieve 99.4 validation accuracy on top of pytorch framework.

Part1: Backpropagation

Backpropagation is a way of finding the gradients of loss & helps in updating the learnable parameters as in a way of making the model converge to global minima. 

The Entire process of backpropagation is made in the Excel sheet: 

Two Inputs are made which combinely make a input layer of 2 neurons, which include a single hidden layer, and a output layer which makes a three-layered Network(in case of input as a layer, else 2)

A/c to first layer: 2 Input neurons are present, which has 4 weights to be learnt (W1,W2,W3,W4) 

A/c to second layer: A hidden layer with 2 neurons are present which in-take input neurons as a weight sum w.r.t to weights (W1,W2,W3,W4).

A/c to third layer: The outputs of second layer using newly formed weights (W5,W6,W7,W8) computes with an activation function and gets the loss. 

Process of Backpropagation:

The main procedure of backpropagation is that the obtained loss which inturn computes from back to first layer.

The loss computes gradients w.r.t parameters, like the weights (W5,W6,W7,W8) finds their rate of change of loss w.r.t to change in weights using chain rule including activation functions in between.   
                                     
                                     
                                     ###A Sample: dl/dW5



<img src="https://github.com/kishkath/S6-MNIST-V1/assets/60026221/121074e9-7acd-4d88-9041-63adf0dfab35" width = 256 height = 256>

<img src="https://github.com/kishkath/S6-MNIST-V1/assets/60026221/40cfca94-016c-4050-97db-89e127420329" width = 256 height = 256>

<img src="https://github.com/kishkath/S6-MNIST-V1/assets/60026221/8035554a-842b-4553-bfd5-d13c7a9db902" width = 256 height = 256>


