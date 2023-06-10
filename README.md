# MNIST-V0

## Learnable Parameters Navigation 

<img src="https://github.com/kishkath/S6-MNIST-V1/assets/60026221/dbbf36be-2034-465f-941a-77d0043de4f9" width = 360 height = 360>

<img src="https://github.com/kishkath/S6-MNIST-V1/assets/60026221/9aae722e-7a34-40f2-aa31-7235dc7f66ad" width = 360 height = 360>



## Part1: Backpropagation

Reference: https://towardsdatascience.com/understanding-backpropagation-abcc509ca9d0

Backpropagation is a way of finding the gradients of loss & helps in updating the learnable parameters as in a way of making the model converge to global minima. 

The Entire process of backpropagation is made in the Excel sheet: 

Two Inputs are made which combinely make a input layer of 2 neurons, which include a single hidden layer, and a output layer which makes a three-layered Network(in case of input as a layer, else 2)

A/c to first layer: 2 Input neurons are present, which has 4 weights to be learnt (W1,W2,W3,W4) 

A/c to second layer: A hidden layer with 2 neurons are present which in-take input neurons as a weight sum w.r.t to weights (W1,W2,W3,W4).

A/c to third layer: The outputs of second layer using newly formed weights (W5,W6,W7,W8) computes with an activation function and gets the loss. 

Process of Backpropagation:

The main procedure of backpropagation is that the obtained loss which inturn computes from back to first layer.

The loss computes gradients w.r.t parameters, like the weights (W5,W6,W7,W8) finds their rate of change of loss w.r.t to change in weights using chain rule including activation functions in between.                                        
                                     
                * We Compute using chain rule dL/dw5, dL/dw6, dL/dw7, dL/dw8, dL/dw1, dL/dw2, dL/dw3, dL/dw4
    
<img src="https://github.com/kishkath/S6-MNIST-V1/assets/60026221/2b6851d0-839c-48d8-9b3f-e7ae2a49ee07" width = 360 height = 360>

<img src="https://github.com/kishkath/S6-MNIST-V1/assets/60026221/c8e2b46b-bc7b-4f33-be96-108d954b26f8" width = 360 height = 360>
          
<img src= "https://github.com/kishkath/S6-MNIST-V1/assets/60026221/33510b7d-f24a-49ce-ab0a-34a18d208e61" width = 360 height = 360>

                 
