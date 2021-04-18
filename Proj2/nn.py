import torch
torch.set_grad_enabled(False)

class Module(object):

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []

class Linear(Module):

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.w = None # initialize weights
        self.b = None # initialize bias

    def forward(self, input):
        return self.w @ inputs.t() + self.b

    def backward(self, gradwrtoutput):
        return gradwrtoutput @ self.w

    def param(self):
        # need to implement gradient and return as pairs
        return [self.w, , self.b]

class ReLu(Module):

    def forward(self, input):
        return input.clamp(min=0)

    def backward(self, output, gradwrtoutput):
        return gradwrtoutput * (output > 0)

class Tanh(Module):

    def forward(self, input):
        return input.tanh()

    def backward(self, output, gradwrtoutput):
        return gradwrtoutput * (1 - output**2)

class LossMSE(object):

    def forward(self, output, target):
        return ((output - target)**2).mean(1)
