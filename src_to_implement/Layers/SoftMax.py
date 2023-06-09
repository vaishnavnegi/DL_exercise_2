from .Base import BaseLayer
import numpy as np

class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.op = None

    def forward(self, input_tensor):
        #Following step helps to improve numerical stability by avoiding overflow issues when computing the exponential function.
        input_tensor = input_tensor - np.max(input_tensor)
        #Numerator = exp of each element of the input_tensor, denominator = sums up the exp values along the axis 1 (row-wise summation) while keeping the dimensionality.
        #Normalizes the exp values, so that the resulting values lie between 0 and 1 and sum up to 1, representing the probabilities of different classes.
        self.op = np.exp(input_tensor) / np.sum(np.exp(input_tensor), axis=1, keepdims=True)
        return self.op

    def backward(self, error_tensor):
        #This step computes the grad of the softmax activation w.r.t the i/p by utilizing the property that the derivative of the softmax function can be expressed
        # in terms of the softmax output itself.
        #The term np.sum(error_tensor * self.op, axis=1).reshape(-1, 1) calculates the weighted sum of error_tensor based on self.op probabilities along the axis 1,
        #and it is subtracted from error_tensor element-wise.
        #The softmax probs. stored in self.op are multiplied element-wise with the computed gradient, which scales the gradient based on the probab calculated during the forward pass.
        ip_grad = self.op * (error_tensor - np.sum(error_tensor * self.op, axis=1).reshape(-1, 1))
        return ip_grad