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
        self.input = None

        self.w = None
        self.b = None
        self.w_grad = torch.empty((self.output_dim, self.input_dim))
        self.b_grad = torch.empty((self.output_dim))

        self.param_init()

    def forward(self, input):
        self.input = input
        return self.input @ self.w.t() + self.b

    def backward(self, gradwrtoutput):
        self.w_grad +=  gradwrtoutput.t() @ self.input
        self.b_grad += gradwrtoutput.sum(0)
        return gradwrtoutput @ self.w

    def param_init(self):
        # can implement different inits
        self.w = torch.empty((self.output_dim, self.input_dim)).uniform_(-1 / (self.input_dim)**2, 1 / (self.input_dim)**2)
        self.b = torch.empty((self.output_dim)).uniform_(-1 / (self.input_dim)**2, 1 / (self.input_dim)**2)

    def param(self):
        return [(self.w, self.grad_w), (self.b, self.grad_b)]


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = layers

    def params(self):
        pars = []
        for layer in self.layers:
            pars += layer.params
        return pars

    def forward(self, input):
        x = input
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, *gradrtoutput):
        x = gradwrtoutput

        for layer in self.layers[::-1]:
            x = layer.backward(x)


class ReLu(Module):
    def __init__(self):
        self.output = None

    def forward(self, input):
        self.output = input.clamp(min=0)
        return self.output

    def backward(self, gradwrtoutput):
        return gradwrtoutput * (self.output > 0)


class Tanh(Module):
    def __init__(self):
        self.output = None

    def forward(self, input):
        self.output = input.tanh()
        return self.output

    def backward(self, output, gradwrtoutput):
        return gradwrtoutput * (1 - self.output**2)


class LossMSE(Module):
    def __init__(self):
        self.output = None
        self.target = None

    def forward(self, output, target):
        self.output = output
        self.target = target
        return (self.output - self.target).pow(2).mean()

    def backward(self):
        return 2*(self.output - self.target) / self.target.size()[0]
