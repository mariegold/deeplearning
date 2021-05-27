from nn import *
from test import *
import math
import torch
import copy
torch.set_grad_enabled(False)
torch.manual_seed(1)

def train_model(model, criterion, train_input, train_target,
                mini_batch_size = 50, nb_epochs = 25, lr = 0.001,
                sgd = True, verbatim = False):
    """ Trains a model using the loss and optimizer of choice. """
    if sgd:
        optimizer = SGD(model.param(), lr)
    else:
        optimizer = Adam(model.param(), lr)
    for e in range(nb_epochs):
        cum_loss = 0.0
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model.forward(train_input.narrow(0, b, mini_batch_size))
            loss = criterion.forward(output, train_target.narrow(0, b, mini_batch_size))
            cum_loss += loss # Accumulate loss per epoch
            model.zero_grad()
            loss_gradient = criterion.backward() # Compute gradient of loss with respect to model's outputs
            model.backward(loss_gradient) # Use the loss gradient to accumulate gradient wrt parameters
            optimizer.step() # Perform the update step on parameters
        if verbatim:
            print("Epoch {}: training loss = {}" .format(e, cum_loss.item()))

def compute_nb_errors(model, test_input, test_target, mini_batch_size = 50):
    """ Computes the number of errors a model makes on the test set. """
    nb_errors = 0
    predictions = []
    for b in range(0, test_input.size(0), mini_batch_size):
        output = model.forward(test_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = output.max(1)
        predictions.extend(predicted_classes.tolist())
        for k in range(mini_batch_size):
            if test_target[b + k] != predicted_classes[k]:
                nb_errors += 1
    return nb_errors

def performance_estimation(datasets, model, criterion, n):
    """ Trains a model using the loss and optimizer of choice. """
    # Parameter grid
    lrs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    batch_size = 25

    # For saving mean and std across datasets for each optimizer and hyperparameter combination
    model_sgd_mean = {}
    model_sgd_std = {}
    model_adam_mean = {}
    model_adam_std = {}
    for lr in lrs:
        model_sgd_mean[lr] = []
        model_sgd_std[lr] = []
        model_adam_mean[lr] = []
        model_adam_std[lr] = []
        # Train with each dataset with the given learning rate using SGD and Adam, save accuracy for each dataset
        for train_input, train_target, test_input, test_target in datasets:
            # Deepcopy is necessary so that we train with the model from scratch with the same initialization
            model_sgd = copy.deepcopy(model)
            train_model(model_sgd, criterion, train_input, train_target,
                        mini_batch_size = batch_size, nb_epochs=30, lr=lr, sgd = True)
            nb_errors_sgd = compute_nb_errors(model_sgd, test_input, test_target)
            model_sgd_mean[lr].append(1-nb_errors_sgd/n)

            model_adam = copy.deepcopy(model)
            train_model(model_adam, criterion, train_input, train_target,
                        mini_batch_size = batch_size, nb_epochs=30, lr=lr,  sgd = False)
            nb_errors_adam = compute_nb_errors(model_adam, test_input, test_target)
            model_adam_mean[lr].append(1-nb_errors_adam/n)

        # Compute mean and standard deviation across the datasets for each optimizer and learning rate
        model_sgd_scores = torch.FloatTensor(model_sgd_mean[lr])
        model_sgd_mean[lr] = model_sgd_scores.mean().item()
        model_sgd_std[lr] = model_sgd_scores.std().item()

        model_adam_scores = torch.FloatTensor(model_adam_mean[lr])
        model_adam_mean[lr] = model_adam_scores.mean().item()
        model_adam_std[lr] = model_adam_scores.std().item()

    # Return means and standard deviations for each learning rate and optimiser
    return model_sgd_mean, model_sgd_std, model_adam_mean, model_adam_std

def train_tune_evaluate(models_list, activation_names, criterions_list, criterion_names, n = 1000, n_runs = 10):
    """
    Generates datasets for tuning and calls performance estimation for each
    model, activation and criterion combination.
    """

    # Generate datasets for evaluation
    datasets = []
    for _ in range(n_runs):
        train_input_tune, train_target_tune = generate_dataset(n)
        test_input_tune, test_target_tune = generate_dataset(n)
        mu, std = train_input_tune.mean(), train_input_tune.std()
        train_input_tune.sub_(mu).div_(std) # standardize
        test_input_tune.sub_(mu).div_(std)

        datasets.append((train_input_tune, train_target_tune, test_input_tune, test_target_tune))

    for (criterion, name) in zip(criterions_list, criterion_names):
        print("----------------------------------------------------------------")
        print(name)
        print("----------------------------------------------------------------")
        # Find best hyperparamters (batch size and learning rate) for each activation function and optimiser combination
        for (model, activation) in zip(models_list, activation_names):
            model_sgd_mean, model_sgd_std, model_adam_mean, model_adam_std = performance_estimation(datasets, model, criterion, n)
            best_sgd_params = max(model_sgd_mean.items(), key = lambda k : k[1])
            print('Best learning rate for model with {} activation and SGD optimiser: {}'.format(activation, best_sgd_params[0]))
            print('With accuracy: {:.3f} +/- {:.3f}.'.format(model_sgd_mean[best_sgd_params[0]], model_sgd_std[best_sgd_params[0]]))
            best_adam_params = max(model_adam_mean.items(), key = lambda k : k[1])
            print('Best learning rate for model with {} activation and Adam optimiser: {}'.format(activation, best_adam_params[0]))
            print('With accuracy: {:.3f} +/- {:.3f}.'.format(model_adam_mean[best_adam_params[0]], model_adam_std[best_adam_params[0]]))

def train_evaluate_best(train_input, train_target, test_input, test_target, model, criterion, lr, sgd):
    """ Trains and evaluates the best model, printing the loss after each epoch. """
    train_model(model, criterion, train_input, train_target,
                mini_batch_size = 25, nb_epochs=30, lr=lr, sgd = sgd, verbatim = True)

    test_nb_errors = compute_nb_errors(model, test_input, test_target)
    train_nb_errors = compute_nb_errors(model, train_input, train_target)
    print('Training accuracy: {:.3f}' .format(1-train_nb_errors/train_target.size(0)))
    print('Test accuracy: {:.3f}' .format(1-test_nb_errors/test_target.size(0)))
