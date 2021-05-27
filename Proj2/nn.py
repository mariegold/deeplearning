import torch
import math
torch.set_grad_enabled(False)

class Module(object):
    """ Parent module from which classes should inherit. """

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
    """ Class for a fully connected layer. """

    def __init__(self, input_dim, output_dim, act=None):
        """
        input_dim (int > 0): number of input units
        output_dim (int > 0): number of output units
        act (string): activation that follows the layer
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input = None

        # for storing weights, biases and their gradients
        self.w = None
        self.b = None
        self.w_grad = torch.empty((self.output_dim, self.input_dim))
        self.b_grad = torch.empty((1, self.output_dim))

        # for Adam, need to store the first two moments of parameters
        self.w_m = torch.empty((self.output_dim, self.input_dim))
        self.w_v = torch.empty((self.output_dim, self.input_dim))
        self.b_m = torch.empty((1, self.output_dim))
        self.b_v = torch.empty((1, self.output_dim))

        # set values for initialization depending on activation
        self.gain = 1
        self.sd = 1
        if act != None:
            self.sd = (2/(self.output_dim + self.input_dim))**0.5
        if act == 'relu':
            self.gain = 2.0**0.5
        if act == 'tanh':
            self.gain = 5.0/3.0
        if act == 'leaky':
            self.gain = (2.0 / (1 + 0.05**2))

        # initialize weights and biases
        self.param_init()

    def forward(self, input):
        """
        Forward pass
        input (tensor): output from the previous layer
        """
        self.input = input
        return self.input @ self.w.t() + self.b

    def backward(self, gradwrtoutput):
        """
        Backward pass
        gradwrtoutput (tensor): gradient of the following layer
        """
        self.w_grad += gradwrtoutput.t() @ self.input
        self.b_grad += gradwrtoutput.sum(0)
        return gradwrtoutput @ self.w

    def param_init(self):
        """
        Initializes parameters of the layer.
        """
        self.w = torch.empty((self.output_dim, self.input_dim)).normal_(0, self.gain * self.sd)
        self.b = torch.empty((1, self.output_dim)).normal_(0, self.gain * self.sd)

        self.w_m.zero_()
        self.w_v.zero_()
        self.b_m.zero_()
        self.b_v.zero_()

    def param(self):
        """
        Returns a list of parameters of the layer as a tuple pair
        (values, gradients, means, variances) for weights and biases
        """
        return [(self.w, self.w_grad, self.w_m, self.w_v), (self.b, self.b_grad, self.b_m, self.b_v)]

    def zero_grad(self):
        """
        Sets gradients to zero.
        """
        self.w_grad.zero_()
        self.b_grad.zero_()


class Sequential(Module):
    """ Class for connecting multiple layers in a sequential manner. """
    def __init__(self, *layers):
        """
        layers (list): sequence of fully connected layers and activations
        """
        super().__init__()

        self.layers = layers

    def param(self):
        """
        Get parameters from all layers as a list.
        """
        pars = []
        for layer in self.layers:
            pars.extend(layer.param())
        return pars

    def forward(self, input):
        """
        Performs the forward pass of the network, i.e. calculates the predictions for a given input.

        input (tensor): input to the network
        """
        x = input
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, *gradwrtoutput):
        """
        Back-propagates the gradient through the layers.

        gradwrtoutput (tensor): gradient w.r.t. the loss function
        """
        x = gradwrtoutput[0]

        # iterate through the layers in reverse order and perform a backward pass
        for layer in self.layers[::-1]:
            x = layer.backward(x)

    def zero_grad(self):
        """
        Sets gradients in all layers to zero.
        """
        for layer in self.layers:
            layer.zero_grad()


## Activation functions
class ReLu(Module):
    """ Class implementing the ReLU activation. """
    def __init__(self):
        super().__init__()

        self.output = None

    def forward(self, input):
        """
        ReLU forward pass
        input (tensor): output from the previous layer
        """
        self.output = input.clamp(min=0)
        return self.output

    def backward(self, gradwrtoutput):
        """
        ReLU backward pass
        gradwrtoutput (tensor): gradient of the following layer
        """
        return gradwrtoutput * (self.output > 0)

class LeakyReLU(Module):
    """ Class implementing the Leaky ReLU activation. """
    def __init__(self, a = 0.05):
        super().__init__()

        self.output = None
        self.a = a

    def forward(self, input):
        """
        Leaky ReLU forward pass
        input (tensor): output from the previous layer
        """
        self.output = input.clamp(min=0) + self.a * input.clamp(max=0)
        return self.output

    def backward(self, gradwrtoutput):
        """
        Leaky ReLU backward pass
        gradwrtoutput (tensor): gradient of the following layer
        """
        gradwrtrelu = self.output.clone()
        gradwrtrelu[gradwrtrelu < 0] = self.a
        gradwrtrelu[gradwrtrelu > 0] = 1
        return gradwrtoutput * gradwrtrelu

class Tanh(Module):
    """ Class implementing the hyperbolic tangent activation. """
    def __init__(self):
        super().__init__()

        self.output = None

    def forward(self, input):
        """
        Hyperbolic tangent forward pass
        input (tensor): output from the previous layer
        """
        self.output = input.tanh()
        return self.output

    def backward(self, gradwrtoutput):
        """
        Hyperbolic tangent backward pass
        gradwrtoutput (tensor): gradient of the following layer
        """
        return gradwrtoutput * (1 - self.output**2)

# Loss functions
class LossMSE(Module):
    """ Class for the Mean Squared Error loss. """
    def __init__(self):
        super().__init__()

        self.output = None
        self.target = None

    def forward(self, output, target):
        """
        Computes the Mean Squared Error.
        output (tensor): output of the network
        target (tensor): labels
        """
        self.output = output
        # Convert target to one hot encoding to be able to use MSE
        target_one_hot = torch.empty((target.size(0), 2)).zero_()
        self.target = target_one_hot.scatter_(1, target.view(-1,1), 1)
        return (self.output - self.target).pow(2).mean()

    def backward(self):
        """
        Mean Squared Error gradient.
        """
        return 2*(self.output - self.target) / self.target.size(0)

class LossCrossEntropy(Module):
    """ Class for the Cross-Entropy loss. """
    def __init__(self):
        super().__init__()

        self.output = None
        self.target = None

    def forward(self, output, target):
        """
        Computes the Cross-Entropy loss.
        output (tensor): output of the network
        target (tensor): labels
        """
        self.output = output
        # Convert target to one hot encoding
        target_one_hot = torch.empty((target.size(0), 2)).zero_()
        self.target = target_one_hot.scatter_(1, target.view(-1,1), 1)
        entropy = (- self.target * self.output.softmax(dim=1).log()).sum()
        return entropy

    def backward(self):
        """
        Cross-Entropy gradient.
        """
        return self.output.softmax(dim=1) - self.target

# Optimizers
class SGD(object):
    """ Class for the Stochastic Gradient Descent optimizer. """
    def __init__(self, params, lr):
        """
        params (list): output of the params method of a layer; parameters to optimize
        lr (float): learning rate
        """
        self.params = params
        self.lr = lr

    def step(self):
        """
        Performs a gradient step according to the SGD update rule.
        """
        for (param, param_grad, _, _) in self.params:
            param.sub_(self.lr * param_grad)

class Adam(object):
    """ Class for the Adam optimizer. """
    def __init__(self, params, lr, beta1 = 0.9, beta2 = 0.999, eps = 1e-8):
        """
        params (list): output of the params method of a layer; parameters to optimize
        lr (float): learning rate
        beta1 (float): Adam's decay parameters
        beta2 (float): Adam's second moment parameter
        eps (float): a small value to prevent division by zero
        """
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

    def step(self):
        """
        Performs a gradient step according to Adam's update rule.
        """
        self.t += 1 # iteration number
        for (param, param_grad, param_m, param_v) in self.params:
            param_m = self.beta1 * param_m + (1 - self.beta1) * param_grad
            param_m_hat = param_m / (1 - self.beta1 ** self.t)
            param_v = self.beta2 * param_v + (1 - self.beta2) * (param_grad ** 2)
            param_v_hat = param_v / (1 - self.beta2 ** self.t)
            param.sub_(self.lr * (param_m_hat / (param_v_hat.sqrt() + self.eps)))
