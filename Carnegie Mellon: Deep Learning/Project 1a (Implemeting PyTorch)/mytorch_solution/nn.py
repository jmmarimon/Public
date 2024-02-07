import numpy as np

class Linear:
    """Linear layer, similar to torch.nn.Linear.
    Args:
        in_features (int): Integer representing # input features expected by layer
        out_features (int): Integer representing # features to be output by layer (i.e. # of neurons)
    """
    def __init__(self, in_features, out_features):
        # Randomly initializes weights and biases ("Kaiming Uniform" init)
        bound = np.sqrt(6 / in_features)
        self.weight = np.random.uniform(low=-bound, high=bound, size=(in_features, out_features))
        self.bias = np.zeros(shape=(1, out_features))

        # Initializes placeholder arrays to store gradients generated during backprop.
        # Notice: The gradients are the same shapes as their originals
        self.grad_weight = np.zeros((in_features, out_features))
        self.grad_bias = np.zeros((1, out_features))

        # You can use this variable to store the input during `self.forward()` to use during `self.backward()`
        self.x = None

    def forward(self, x):
        """Forward pass for linear layer.
        Args:
            x (np.array): Input to the layer, shaped (batch_size, in_features)
        Returns:
            np.array: Output of the layer, shaped (batch_size, out_features)
        """
        # [Given] Store the input (we need it for backward)
        self.x = x

        # TODO: Return the output of the given equation
        return x @ self.weight + self.bias

    def backward(self, grad):
        """Backward pass for linear layer.
        Args:
            grad (np.array): The gradient of the loss w.r.t. the output of this layer
                             shaped (batch_size, out_features)
        Returns:
            np.array: The gradient of the loss w.r.t. the input to this layer
                      shaped (batch_size, in_features)
        """
        # TODO: Calculate and store the gradient of the loss w.r.t. this layer's weight array.
        self.grad_weight = self.x.T @ grad

        # TODO: Calculate and store the gradient of the loss w.r.t. this layer's bias array.
        self.grad_bias = np.sum(grad, axis=0, keepdims=True)

        # TODO: Calculate and return the gradient of the loss w.r.t. this layer's input
        return grad @ self.weight.T

class ReLU:
    """The ReLU activation function, similar to `torch.nn.ReLU`."""
    def __init__(self):
        self.x = None

    def forward(self, x):
        """Forward pass for ReLU.
        Args:
            x (np.array): Input shaped (batch_size, *), where * means any number of additional dims.
        Returns:
            np.array: Output, same shape as input (batch_size, *)
        """
        # [Given] We store the input for you to use during backprop
        self.x = x

        # TODO: Return the output of the given equation
        return np.maximum(0, x)

    def backward(self, grad):
        """Backward pass for ReLU.
        Args:
            grad (np.array): The gradient of the loss w.r.t. the output of this function
                             shaped (batch_size, *)
        Returns:
            np.array: The gradient of the loss w.r.t. the input to this function
                      shaped (batch_size, *)
        """
        # [Given] Retrieve the input that we stored during forward propagation.
        state = self.x

        # TODO: Using the stored input, make the matrix described in Hint 2.
        mask = np.where(state <= 0, 0, 1)

        # TODO: Return the output of the given equation
        return grad * mask

class Sequential:
    """Takes given layers and makes a simple feed-forward network from them. Similar to `torch.nn.Sequential`
    Accepts any number of layers.
    
    Example:
    >>> model = Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 5))
    The input to model must be shaped (batch_size, 3)
    The output of model will be shaped (batch_size, 5) 
    """
    def __init__(self, *layers):
        self.layers = list(layers) # Stores layers in a list, ex) [nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 5)]
        
    def forward(self, x):
        """Passes input `x` through each of the layers in order, returns final output.
        Args:
            x (np.array): Input shaped (batch_size, num_features)
                          Must be shaped appropriately to go in first layer.
        Returns:
            np.array: Output after passing through all layers shaped (batch_size, num_classes)
        """
        # TODO: Pass input through the network, return final output 
        for layer in self.layers:
            x = layer.forward(x)
        return x
        
    def backward(self, loss_function):
        """Runs backpropagation. Does not return anything.
        Args:
            loss_function (nn.CrossEntropyLoss): Loss function after running the forward pass (input/target already stored)
        """
        # TODO: Get the gradient from the loss function's backward method
        grad = loss_function.backward()
        
        # TODO: Give the gradient to the last layer, then continue passing through the network backwards
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

def softmax(x):
    """[Given] Calculates the softmax of the input array using the LogSumExp trick for numerical stability.
    Args:
        x (np.array): Input array, shaped (batch_size, d), where d is any integer.
    Returns:
        np.array: Same shape as input, but the values of each row are now scaled to add up to 1.
    """
    # [Given] Use this in CrossEntropyLoss.
    a = np.max(x, axis=1, keepdims=True)
    denom = a + np.log(np.sum(np.exp(x - a), axis=1, keepdims=True))
    return np.exp(x - denom)

def make_one_hot(idx_labels, num_classes=10):
    """[Given] Converts index labels to one-hot encoding.
    
    >>> make_one_hot(np.array([0, 2, 1]), num_classes=3)
    array([[1., 0., 0.],
           [0., 0., 1.],
           [0., 1., 0.]])

    Args:
        idx_labels (np.array): Array of labels
        num_classes (int, optional): # of possible classes there are.

    Returns:
        np.array: One-hot encoded labels.
    """
    # [Given] Used in cross entropy loss
    one_hot_labels = np.zeros((len(idx_labels), num_classes))
    one_hot_labels[np.arange(len(idx_labels)), idx_labels] = 1
    return one_hot_labels

class CrossEntropyLoss:
    """Cross-entropy loss function. Similar to `torch.nn.CrossEntropyLoss`."""
    def __init__(self):
        self.input = None
        self.target = None

    def forward(self, input, target):
        """Calculates the average loss of the batched inputs and targets.
        Args:
            input (np.array): Logits (output of model) of shape (batch_size, num_classes)
            target (np.array): Label array of shape (batch_size,), after conversion to one-hot it's shaped (batch_size, num_classes)
        Returns:
            float or np.float64: The loss value (averaged across the batch)
        """
        # [Given] Convert the targets to a one-hot encoding shaped (batch_size, num_classes)
        target = make_one_hot(target, num_classes=input.shape[1])

        # [Given] Store the inputs and the one-hot encoded targets for backward
        self.input = input
        self.target = target

        # TODO: Calculate the cross entropy loss and return the average across the batch
        loss = - np.sum(target * np.log(softmax(input)), axis=1)
        return np.mean(loss)

    def backward(self):
        """Begins backprop by calculating the gradient of the loss w.r.t. CrossEntropyLoss's forward.
        Similar to calling `loss.backward()` in the real Torch.

        Returns:
            np.array: the gradient of the loss w.r.t. the input of CrossEntropyLoss (batch size, num_classes)
        """
        # TODO: Calculate and return the gradient of the loss w.r.t. the input of CrossEntropyLoss.
        batch_size = self.input.shape[0]
        return (softmax(self.input) - self.target) / batch_size