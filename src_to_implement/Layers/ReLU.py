from .Base import BaseLayer
import numpy as np

class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()
        self.RLUGrad = None

    def forward(self, input_tensor):
        #As ReLU is max(0,x), clip at 0.
        output_tensor = np.clip(input_tensor, a_min=0., a_max=None)
        self.RLUGrad = output_tensor.copy()
        return output_tensor

    def backward(self , error_tensor):
        #The differentiation of ReLU = 0 for x < 0 and 1 for x > 0.
        self.RLUGrad[self.RLUGrad>0] = 1
        ip_grad = error_tensor * self.RLUGrad
        return ip_grad