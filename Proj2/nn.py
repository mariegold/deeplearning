import torch
import math
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
        # standard PyTorch init
        self.w = torch.empty((output_dim, input_dim)).uniform_(-1 / (input_dim)**2, 1 / (input_dim)**2)
        self.b = torch.empty(output_dim).uniform_(-1 / (input_dim)**2, 1 / (input_dim)**2)

    def forward(self, input):
        return self.w @ input.t() + self.b

    def backward(self, gradwrtoutput):
        return self.w @ gradwrtoutput.t()

    def param_grad(self, input, gradwrtoutput):
        self.w_grad = gradwrtoutput.t() @ input
        self.b_grad = gradwrtoutput

    def param(self):
        # add gradients to output
        return [self.w, self.b]

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

    def __call__(self, output, target):
        return (output - target).pow(2).mean()

    def backward(self, output, target):
        return 2*(output - target) / target.size()[0]
