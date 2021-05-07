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

    def zero_grad(self):
        pass


## Modules
class Linear(Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input = None

        self.w = None
        self.b = None
        self.w_grad = torch.empty((self.output_dim, self.input_dim))
        self.b_grad = torch.empty((1, self.output_dim))

        self.w_m = torch.empty((self.output_dim, self.input_dim))
        self.w_v = torch.empty((self.output_dim, self.input_dim))
        self.b_m = torch.empty((1, self.output_dim))
        self.b_v = torch.empty((1, self.output_dim))

        self.param_init()

    def forward(self, input):
        self.input = input
        return self.input @ self.w.t() + self.b

    def backward(self, gradwrtoutput):
        self.w_grad += gradwrtoutput.t() @ self.input
        self.b_grad += gradwrtoutput.sum(0)
        return gradwrtoutput @ self.w

    def param_init(self):
        # can implement different inits
        self.w = torch.empty((self.output_dim, self.input_dim)).normal_(0, (2/(self.output_dim + self.input_dim))**0.5)
        self.b = torch.empty((1, self.output_dim)).normal_(0, (2/(self.output_dim + self.input_dim))**0.5)

        self.w_m.zero_()
        self.w_v.zero_()
        self.b_m.zero_()
        self.b_v.zero_()

    def param(self):
        return [(self.w, self.w_grad, self.w_m, self.w_v), (self.b, self.b_grad, self.b_m, self.b_v)]

    def zero_grad(self):
        self.w_grad.zero_()
        self.b_grad.zero_()


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()

        self.layers = layers

    def param(self):
        pars = []
        for layer in self.layers:
            pars.extend(layer.param())
        return pars

    def forward(self, input):
        x = input
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, *gradwrtoutput):
        x = gradwrtoutput[0]

        for layer in self.layers[::-1]:
            x = layer.backward(x)

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()


## Activation functions
class ReLu(Module):
    def __init__(self):
        super().__init__()

        self.output = None

    def forward(self, input):
        self.output = input.clamp(min=0)
        return self.output

    def backward(self, gradwrtoutput):
        return gradwrtoutput * (self.output > 0)


class Tanh(Module):
    def __init__(self):
        super().__init__()

        self.output = None

    def forward(self, input):
        self.output = input.tanh()
        return self.output

    def backward(self, gradwrtoutput):
        return gradwrtoutput * (1 - self.output**2)

class Sigmoid(Module):
    def __init__(self):
        super().__init__()

        self.output = None

    def forward(self, input):
        self.output = input.sigmoid()
        return self.output

    # Use identity o'(x) = o(x) * (1-o(x))
    def backward(self, gradwrtoutput):
        return gradwrtoutput * self.output * (1 - self.output)        

# Loss functions
class LossMSE(Module):
    def __init__(self):
        super().__init__()

        self.output = None
        self.target = None

    def forward(self, output, target):
        self.output = output
        # Convert target to one hot encoding to be able to use MSE
        target_one_hot = torch.empty((target.size(0), 2)).zero_()
        self.target = target_one_hot.scatter_(1, target.view(-1,1), 1)
        return (self.output - self.target).pow(2).mean()

    def backward(self):
        return 2*(self.output - self.target) / self.target.size(0)

# Optimizers
class SGD():
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def step(self):
        for (param, param_grad, _, _) in self.params:
            param.sub_(self.lr * param_grad)

class Adam():
    def __init__(self, params, lr, beta1 = 0.9, beta2 = 0.999, eps = 1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
    def step(self):
        self.t += 1
        for (param, param_grad, param_m, param_v) in self.params:
            param_m = self.beta1 * param_m + (1 - self.beta1) * param_grad
            param_m_hat = param_m / (1 - self.beta1 ** self.t)
            param_v = self.beta2 * param_v + (1 - self.beta2) * (param_grad ** 2)
            param_v_hat = param_v / (1 - self.beta2 ** self.t)
            param.sub_(self.lr * (param_m_hat / (param_v_hat.sqrt() + self.eps)))    