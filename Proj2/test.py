from nn import *
from train import *

import math
import torch

torch.set_grad_enabled(False)
torch.manual_seed(1)

# Set to True if you want to see the results for all the models (averaged over 10 runs)
run_training = True
# Set to True if you want to see the training of the best model only (1 run)
run_best = True

def generate_dataset(n):
    """ Generates toy dataset; points in [0,1]x[0,1] such that the labels are 1 inside
     the disk centered at (0.5, 0.5) with radius 1/sqrt(2*pi) and 0 otherwise """

    input = torch.empty(n, 2).uniform_(0, 1)
    target = (input - 0.5).pow(2).sum(1).sub(1/(2*math.pi)).sign().add(1).div(2).long()
    return input, target

if __name__ == '__main__':
    n = 1000

    # Generate datasets
    train_input, train_target = generate_dataset(n)
    test_input, test_target = generate_dataset(n)

    # Standardize dataset
    mu, std = train_input.mean(), train_input.std()
    train_input.sub_(mu).div_(std)
    test_input.sub_(mu).div_(std)

    # Testing different activation functions
    model_relu = Sequential(Linear(2,25,act='relu'), ReLu(),
                            Linear(25,25,act='relu'), ReLu(),
                            Linear(25,25,act='relu'), ReLu(),
                            Linear(25, 2))
    model_tanh = Sequential(Linear(2,25,act='tanh'), Tanh(),
                            Linear(25,25,act='tanh'), Tanh(),
                            Linear(25,25,act='tanh'), Tanh(),
                            Linear(25, 2))
    model_leaky = Sequential(Linear(2,25,act='leaky'), LeakyReLU(),
                             Linear(25,25,act='leaky'), LeakyReLU(),
                             Linear(25,25,act='leaky'), LeakyReLU(),
                             Linear(25, 2))

    models_list = [model_relu, model_tanh, model_leaky]
    activation_names = ["ReLU", "Tanh", "Leaky ReLU"]

    # Testing different optimizers
    criterions_list = [LossMSE(), LossCrossEntropy()]
    criterion_names = ["Mean Squared Error", "Cross Entropy"]

    if run_training:
        train_tune_evaluate(models_list, activation_names, criterions_list, criterion_names)
    if run_best:
        train_evaluate_best(train_input, train_target, test_input, test_target,
                   model = model_relu, criterion = LossCrossEntropy(), lr = 0.001, sgd = True)
