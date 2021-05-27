import torch
from torch import nn
from torch import optim
import copy
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

# Assumes k divides the training input size
def cross_validate(train_input, train_target, train_classes, lr, n, k = 5):
    # Parameter grid
    dropout_rates = [0.0, 0.1, 0.2, 0.5, 0.8]
    use_bn = [True, False]

    param_combinatinos = [(bn, dropout)
        for bn in use_bn
        for dropout in dropout_rates]
    # For saving mean across folds
    model_base_mean = {}
    model_aux_mean = {}
    model_ws_mean = {}
    model_ws_aux_mean = {}

    fold_size = n // k

    for param_combo in param_combinatinos:
        model_base_mean[param_combo] = []
        model_aux_mean[param_combo] = []
        model_ws_mean[param_combo] = []
        model_ws_aux_mean[param_combo] = []
        bn, dropout = param_combo

        model_base = BaseNet(batch_normalization=bn, dropout=dropout)
        model_aux = BaseNetAux(batch_normalization=bn, dropout=dropout)
        model_ws = BaseNetWeightShare(batch_normalization=bn, dropout=dropout)
        model_ws_aux = BaseNetWeightShareAux(batch_normalization=bn, dropout=dropout)


        for i in range(k):
            # Construct training and validation data for this fold
            start = i * fold_size
            end = start + fold_size
            train_input_fold = train_input[start : end]
            val_input_fold = torch.cat((train_input[:start], train_input[end:]),0)
            train_target_fold = train_target[start : end]
            val_target_fold = torch.cat((train_target[:start], train_target[end:]),0)
            train_classes_fold = train_classes[start : end]

            train_model(copy.deepcopy(model_base), train_input_fold, train_target_fold, mini_batch_size = 25, nb_epochs=30, lr=lr)
            nb_errors_base = compute_nb_errors(model_base, val_input_fold, val_target_fold, mini_batch_size = 25)
            model_base_mean[param_combo].append(1-nb_errors_base/n)

            train_model_with_aux_loss(copy.deepcopy(model_aux), train_input_fold, train_target_fold, train_classes_fold, mini_batch_size = 25, nb_epochs=30, lr=lr)
            nb_errors_aux = compute_nb_errors_with_aux_loss(model_aux, val_input_fold, val_target_fold, mini_batch_size = 25)
            model_aux_mean[param_combo].append(1-nb_errors_aux/n)

            train_model(copy.deepcopy(model_ws), train_input_fold, train_target_fold, mini_batch_size = 25, nb_epochs=30, lr=lr)
            nb_errors_ws = compute_nb_errors(model_ws, val_input_fold, val_target_fold, mini_batch_size = 25)
            model_ws_mean[param_combo].append(1-nb_errors_ws/n)

            train_model_with_aux_loss(copy.deepcopy(model_ws_aux), train_input_fold, train_target_fold, train_classes_fold, mini_batch_size = 25, nb_epochs=30, lr=lr)
            nb_errors_ws_aux = compute_nb_errors_with_aux_loss(model_ws_aux, val_input_fold, val_target_fold, mini_batch_size = 25)
            model_ws_aux_mean[param_combo].append(1-nb_errors_ws_aux/n)


        # Compute mean and standard deviation across the datasets for each model and param combo
        model_base_scores = torch.FloatTensor(model_base_mean[param_combo])
        model_base_mean[param_combo] = model_base_scores.mean().item()

        model_aux_scores = torch.FloatTensor(model_aux_mean[param_combo])
        model_aux_mean[param_combo] = model_aux_scores.mean().item()

        model_ws_scores = torch.FloatTensor(model_ws_mean[param_combo])
        model_ws_mean[param_combo] = model_ws_scores.mean().item()

        model_ws_aux_scores = torch.FloatTensor(model_ws_aux_mean[param_combo])
        model_ws_aux_mean[param_combo] = model_ws_aux_scores.mean().item()

    # Return means for each model and param combo
    return model_base_mean, model_aux_mean, model_ws_mean, model_ws_aux_mean


# Try all models with different learning rates, batch sizes, dropout rates and varying use of bn on for multiple datasets
# Record mean and standard deviation of accuracy of each parameter setting
def performance_estimation_param_tune(datasets, lr, n):
    # Parameter grid
    dropout_rates = [0.0, 0.1, 0.2, 0.5, 0.8]
    use_bn = [True, False]

    param_combinatinos = [(bn, dropout)
        for bn in use_bn
        for dropout in dropout_rates]
    # For saving mean and std across datasets for each model and parameter combination
    model_base_mean = {}
    model_base_std = {}
    model_aux_mean = {}
    model_aux_std = {}
    model_ws_mean = {}
    model_ws_std = {}
    model_ws_aux_mean = {}
    model_ws_aux_std = {}

    for param_combo in param_combinatinos:
        model_base_mean[param_combo] = []
        model_aux_mean[param_combo] = []
        model_ws_mean[param_combo] = []
        model_ws_aux_mean[param_combo] = []
        bn, dropout = param_combo
        # Train each model with each dataset with the given param combination, save accuracy for each dataset
        for train_input, train_target, train_classes, test_input, test_target, _ in datasets:
            model_base = BaseNet(batch_normalization=bn, dropout=dropout)
            model_aux = BaseNetAux(batch_normalization=bn, dropout=dropout)
            model_ws = BaseNetWeightShare(batch_normalization=bn, dropout=dropout)
            model_ws_aux = BaseNetWeightShareAux(batch_normalization=bn, dropout=dropout)

            train_model(model_base, train_input, train_target, mini_batch_size = 25, nb_epochs=30, lr=lr)
            nb_errors_base = compute_nb_errors(model_base, test_input, test_target, mini_batch_size = 25)
            model_base_mean[param_combo].append(1-nb_errors_base/n)

            train_model_with_aux_loss(model_aux, train_input, train_target, train_classes, mini_batch_size = 25, nb_epochs=30, lr=lr)
            nb_errors_aux = compute_nb_errors_with_aux_loss(model_aux, test_input, test_target, mini_batch_size = 25)
            model_aux_mean[param_combo].append(1-nb_errors_aux/n)

            train_model(model_ws, train_input, train_target, mini_batch_size = 25, nb_epochs=30, lr=lr)
            nb_errors_ws = compute_nb_errors(model_ws, test_input, test_target, mini_batch_size = 25)
            model_ws_mean[param_combo].append(1-nb_errors_ws/n)

            train_model_with_aux_loss(model_ws_aux, train_input, train_target, train_classes, mini_batch_size = 25, nb_epochs=30, lr=lr)
            nb_errors_ws_aux = compute_nb_errors_with_aux_loss(model_ws_aux, test_input, test_target, mini_batch_size = 25)
            model_ws_aux_mean[param_combo].append(1-nb_errors_ws_aux/n)


        # Compute mean and standard deviation across the datasets for each model and param combo
        model_base_scores = torch.FloatTensor(model_base_mean[param_combo])
        model_base_mean[param_combo] = model_base_scores.mean().item()
        model_base_std[param_combo] = model_base_scores.std().item()

        model_aux_scores = torch.FloatTensor(model_aux_mean[param_combo])
        model_aux_mean[param_combo] = model_aux_scores.mean().item()
        model_aux_std[param_combo] = model_aux_scores.std().item()

        model_ws_scores = torch.FloatTensor(model_ws_mean[param_combo])
        model_ws_mean[param_combo] = model_ws_scores.mean().item()
        model_ws_std[param_combo] = model_ws_scores.std().item()

        model_ws_aux_scores = torch.FloatTensor(model_ws_aux_mean[param_combo])
        model_ws_aux_mean[param_combo] = model_ws_aux_scores.mean().item()
        model_ws_aux_std[param_combo] = model_ws_aux_scores.std().item()

    # Return means and standard deviations for each model and param combo
    return model_base_mean, model_base_std, model_aux_mean, model_aux_std, model_ws_mean, model_ws_std, model_ws_aux_mean, model_ws_aux_std


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

