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
* Xavier/Glorot for tanh: Weights ~ U(−√(6/(fan_in+fan_out)), √(…)), biases = 0