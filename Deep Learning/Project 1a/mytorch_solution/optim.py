import numpy as np

from . import nn

class SGD:
    """Stochastic Gradient Descent (SGD). Similar to `torch.optim.SGD`.
    
    Args:
        model (nn.Sequential): Your initialized network, stored in a `Sequential` object.
                               ex) nn.Sequential(Linear(2,3), ReLU(), Linear(3,2))
        lr (float): Learning rate. ex) 0.01
    """
    def __init__(self, model, lr):
        self.layers = model.layers
        self.lr = lr
        
    def zero_grad(self):
        """[Given] Resets the gradients of weights to be filled with zeroes."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                layer.grad_weight.fill(0)
                layer.grad_bias.fill(0)
    
    def step(self):
        """Called after backprop. This updates the weights with the gradients generated during backprop."""
        # TODO: Update the weights/biases of any Linear layers.
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                layer.weight = layer.weight - self.lr * layer.grad_weight
                layer.bias = layer.bias - self.lr * layer.grad_bias