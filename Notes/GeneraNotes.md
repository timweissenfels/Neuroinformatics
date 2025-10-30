# General Notes

## How many layers?
* Universal Approximation Theorem: one wide layer can approximate continuous functions
* Add more layers if there is underfitting which can not be fixed by increasing the amount of nodes in the first layer (width)

## Width of the hidden layer
* Begin small (e.g 8 or 17)
* If the model underfits increase width
* If the model overfits, decrease width or add regulizer

## Activation
* Hidden: thanh (zero-centered). Relu also works for sin, but tanh matches the output range nicely
* Output: Sigmoid + BCE for XOR. Linear + MSE for sin/cos (or tanh + MSE if targets are normalized [-1, 1))

## Shapes
* With hidden size h: W1: (h x in), b1 (h x 1), W2: (out x h), b2: (out x 1)

## Init

### Xavier (from STAT 454 Sebastian Raschka)
1. Initialize weights from gaussian or uniform distribution W ~ N(0, 0.01)
2. Scale the weights proportinal to the number of inputs to the layer W * sqrt(1/m) where m is the number of features of the previous layer 

Sidenote: If you didnt initliaze the bias units to all zeros, also include those in the scaling

Also some people use fan_avg instead of m which is (fan_in + fan_out)/2 

Use with linear, tanh, softmax, logistic activation functions

(For the first hidden layer, that is the number of featuers in the dataset for the second hidden layer that is the number of units in the 1st hidden layer etc.)

### Kaiming He (from STAT 454 Sebastian Raschka) 
* Assuming activations with mean 0, which is reasonable, Xavier initialization assumes a derivative of for the activation function (s.a. tanh)
* For ReLU, this is different, as the acitavtions are not centered at zero anzmore
* He initialization takes this into account. The result is, that we add a scaling factor of 2^0.5 W*sqrt(2/m)

Best used with ReLu or variants of ReLu

### Lecun
* Same as Kaiming He but with 1/ instead of 2/

Best used for SeLU

### Vanishing Gradients
Deep Neural Networks face the difficulty that variance of the layer outputs gets lower the more upstream the data you go

That causes a slow model convergence

Early layers have mostly gradients close to 0, later layers have high gradients -> Network does only update later layers -> not optimal