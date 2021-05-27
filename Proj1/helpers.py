import torch
from torch import nn
from torch import optim
import copy
from dlc_practical_prologue import generate_pair_sets
from models import *

def train_model(model, train_input, train_target, mini_batch_size = 50, nb_epochs = 25, lr = 0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = lr)
    for e in range(nb_epochs):
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            model.zero_grad()
            loss.backward()
            optimizer.step()

def train_model_with_aux_loss(model, train_input, train_target, train_classes, mini_batch_size = 50, nb_epochs = 25, lr = 0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = lr)
    for e in range(nb_epochs):
        for b in range(0, train_input.size(0), mini_batch_size):
            output_bool, output_digit1, output_digit2 = model(train_input.narrow(0, b, mini_batch_size))
            loss_bool = criterion(output_bool, train_target.narrow(0, b, mini_batch_size))
            loss_digit1 = criterion(output_digit1, train_classes.narrow(0, b, mini_batch_size)[:,0])
            loss_digit2 = criterion(output_digit2, train_classes.narrow(0, b, mini_batch_size)[:,1])
            loss = loss_bool + loss_digit1 + loss_digit2
            model.zero_grad()
            loss.backward()
            optimizer.step()

def compute_nb_errors(model, test_input, test_target, mini_batch_size = 50):
    nb_errors = 0
    model.eval()
    for b in range(0, test_input.size(0), mini_batch_size):
        output = model(test_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = output.max(1)
        for k in range(mini_batch_size):
            if test_target[b + k] != predicted_classes[k]:
                nb_errors = nb_errors + 1
    return nb_errors

def compute_nb_errors_with_aux_loss(model, test_input, test_target, mini_batch_size = 50):
    nb_errors = 0
    model.eval()
    for b in range(0, test_input.size(0), mini_batch_size):
        output_bool, output_digit1, output_digit2 = model(test_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = output_bool.max(1)
        for k in range(mini_batch_size):
            if test_target[b + k] != predicted_classes[k]:
                nb_errors = nb_errors + 1
    return nb_errors


def performance_estimation(datasets, model, lr, aux_loss, n):
    # For saving scores across runs
    model_mean = []
    # Train model with each dataset, save accuracy for each dataset
    for train_input, train_target, train_classes, test_input, test_target, _ in datasets:
        model_copy = copy.deepcopy(model)
        if aux_loss:
            train_model_with_aux_loss(model_copy, train_input, train_target, train_classes, mini_batch_size = 25, nb_epochs=30, lr=lr)
            nb_errors = compute_nb_errors_with_aux_loss(model_copy, test_input, test_target, mini_batch_size = 25)
            model_mean.append(1 - nb_errors/n)
        else:
            train_model(model_copy, train_input, train_target, mini_batch_size = 25, nb_epochs=30, lr=lr)
            nb_errors = compute_nb_errors(model_copy, test_input, test_target, mini_batch_size = 25)
            model_mean.append(1 - nb_errors/n)

    # Compute mean and standard deviation across the datasets for each model and param combo
    model_scores = torch.FloatTensor(model_mean)
    model_mean = model_scores.mean().item()
    model_std = model_scores.std().item()

    # Return mean and standard deviation
    return model_mean, model_std

def param_tune(init_train_input, init_train_target, init_train_classes, init_test_input, init_test_target, lr, n):
    # Parameter grid 
    dropout_rates = [0.0, 0.1, 0.2, 0.5, 0.8]
    use_bn = [True, False]
    
    param_combinatinos = [(bn, dropout) 
        for bn in use_bn 
        for dropout in dropout_rates]    
    
    model_base_mean = {} 
    model_aux_mean = {}
    model_ws_mean = {}
    model_ws_aux_mean = {}
    for param_combo in param_combinatinos:
        bn, dropout = param_combo

        model_base = BaseNet(batch_normalization=bn, dropout=dropout)
        model_aux = BaseNetAux(batch_normalization=bn, dropout=dropout)
        model_ws = BaseNetWeightShare(batch_normalization=bn, dropout=dropout)
        model_ws_aux = BaseNetWeightShareAux(batch_normalization=bn, dropout=dropout)

        train_model(model_base, init_train_input, init_train_target, mini_batch_size = 25, nb_epochs=30, lr=lr)
        nb_errors_base = compute_nb_errors(model_base, init_test_input, init_test_target, mini_batch_size = 25)
        model_base_mean[param_combo] = 1-nb_errors_base/n

        train_model_with_aux_loss(model_aux, init_train_input, init_train_target, init_train_classes, mini_batch_size = 25, nb_epochs=30, lr=lr)
        nb_errors_aux = compute_nb_errors_with_aux_loss(model_aux, init_test_input, init_test_target, mini_batch_size = 25)
        model_aux_mean[param_combo]= 1-nb_errors_aux/n

        train_model(model_ws, init_train_input, init_train_target, mini_batch_size = 25, nb_epochs=30, lr=lr)
        nb_errors_ws = compute_nb_errors(model_ws, init_test_input, init_test_target, mini_batch_size = 25)
        model_ws_mean[param_combo]= 1-nb_errors_ws/n

        train_model_with_aux_loss(model_ws_aux, init_train_input, init_train_target, init_train_classes, mini_batch_size = 25, nb_epochs=30, lr=lr)
        nb_errors_ws_aux = compute_nb_errors_with_aux_loss(model_ws_aux, init_test_input, init_test_target, mini_batch_size = 25)
        model_ws_aux_mean[param_combo] = 1-nb_errors_ws_aux/n

    return model_base_mean, model_aux_mean, model_ws_mean, model_ws_aux_mean

def train_tune_evaluate(lr, n):
    # Generate an initial dataset for parameter tuning
    init_train_input, init_train_target, init_train_classes, init_test_input, init_test_target, _ = generate_pair_sets(n)
    model_base_mean, model_aux_mean, model_ws_mean, model_ws_aux_mean = param_tune(init_train_input, init_train_target, init_train_classes, init_test_input, init_test_target, lr, n)

    best_base_params, init_base_acc = max(model_base_mean.items(), key = lambda k : k[1])
    print('Best (use_bn, dropout rate) combination with BaseNet:', best_base_params) 
    print('Initial dataset accuracy with BaseNet: {:.3f}'.format(init_base_acc)) 

    best_aux_params, init_aux_acc = max(model_aux_mean.items(), key = lambda k : k[1])
    print('Best (use_bn, dropout rate) combination with BaseNetAux:', best_aux_params) 
    print('Initial dataset accuracy with BaseNetAux: {:.3f}'.format(init_aux_acc))

    best_ws_params, init_wc_acc = max(model_ws_mean.items(), key = lambda k : k[1])
    print('Best (use_bn, dropout rate) combination with BaseNetWeightShare:', best_ws_params) 
    print('Initial dataset accuracy with BaseNetWeightShare: {:.3f}'.format(init_wc_acc)) 

    best_ws_aux_params, init_ws_aux_acc = max(model_ws_aux_mean.items(), key = lambda k : k[1])
    print('Best (use_bn, dropout rate) combination with BaseNetWeightShareAux:', best_ws_aux_params) 
    print('Initial dataset accuracy with BaseNetWeightShareAux: {:.3f}'.format(init_ws_aux_acc)) 

    best_base_bn, best_base_dropout = best_base_params
    best_model_base = BaseNet(batch_normalization=best_base_bn, dropout=best_base_dropout)

    best_aux_bn, best_aux_dropout = best_aux_params
    best_model_aux = BaseNetAux(batch_normalization=best_aux_bn, dropout=best_aux_dropout)

    best_ws_bn, best_ws_dropout = best_ws_params
    best_model_ws = BaseNetWeightShare(batch_normalization=best_ws_bn, dropout=best_ws_dropout)

    best_ws_aux_bn, best_ws_aux_dropout = best_ws_aux_params
    best_model_ws_aux = BaseNetWeightShareAux(batch_normalization=best_ws_aux_bn, dropout=best_ws_aux_dropout)

    # Generate 10 datasets for performance estimation
    datasets = []
    for _ in range(10):
        train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(n)
        # Standardize dataset
        mu, std = train_input.mean(), train_input.std()
        train_input.sub_(mu).div_(std)
        test_input.sub_(mu).div_(std)
        datasets.append((train_input, train_target, train_classes, test_input, test_target, test_classes))

    # Find and report means and standard devs
    base_mean, base_std = performance_estimation(datasets, best_model_base, lr, False, n)
    print('Final BaseNet accuracy: {:.3f} +/- {:.3f}.'.format(base_mean, base_std))
    aux_mean, aux_std = performance_estimation(datasets, best_model_aux, lr, True, n)
    print('Final BaseNetAux accuracy: {:.3f} +/- {:.3f}.'.format(aux_mean, aux_std))
    ws_mean, ws_std = performance_estimation(datasets, best_model_ws, lr, False, n)
    print('Final BaseNetWeightShare accuracy: {:.3f} +/- {:.3f}.'.format(ws_mean, ws_std))
    ws_aux_mean, ws_aux_std = performance_estimation(datasets, best_model_ws_aux, lr, True, n)
    print('Final BaseNetWeightShareAux accuracy: {:.3f} +/- {:.3f}.'.format(ws_aux_mean, ws_aux_std)) 

def train_evaluate_best(model, train_input, train_target, train_classes, test_input, test_target, aux_loss, lr, n):
    if aux_loss:
        train_model_with_aux_loss(model, train_input, train_target, train_classes, mini_batch_size = 25, nb_epochs=30, lr=lr)
        nb_errors = compute_nb_errors_with_aux_loss(model, test_input, test_target, mini_batch_size = 25)    
        print('Accuracy of the best model: {:.3f}'.format(1-nb_errors/n))
    else:
        train_model(model, train_input, train_target, mini_batch_size = 25, nb_epochs=30, lr=lr)
        nb_errors = compute_nb_errors(model, test_input, test_target, mini_batch_size = 25)
        print('Accuracy of the best model: {:.3f}'.format(1-nb_errors/n))