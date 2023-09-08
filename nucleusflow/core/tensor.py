import numpy as np

class Tensor:

    def __init__(self, data, requires_grad=False):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = None  # We'll use this later to store the function that generated this tensor

    def __add__(self, other):
        # Basic tensor addition
        result = Tensor(self.data + other.data)
        return result

    def __mul__(self, other):
        # Basic tensor multiplication
        result = Tensor(self.data * other.data)
        return result

    def backward(self, grad=None):
        # Placeholder for backpropagation implementation
        pass

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"