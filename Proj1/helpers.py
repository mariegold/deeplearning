import torch
from torch import nn
from torch import optim
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

# Try all models with different learning rates, batch sizes, dropout rates and varying use of bn on for multiple datasets
# Record mean and standard deviation of accuracy of each parameter setting
def performance_estimation(datasets):
    n = 1000
    # Parameter grid
    lrs = [1e-4, 1e-3, 1e-2, 1e-1,1]
    batch_sizes = [1, 5, 10, 20, 50, 100]
    dropout_rates = [0.0, 0.1, 0.2, 0.5, 0.8]
    use_bn = [True, False]

    param_combinatinos = [(lr, batch_size, bn, dropout) 
        for lr in lrs
        for batch_size in batch_sizes 
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
        lr, batch_size, bn, dropout = param_combo
        # Train each model with each dataset with the given param combination, save accuracy for each dataset
        for train_input, train_target, train_classes, test_input, test_target, test_classes in datasets:
            model_base = BaseNet(batch_normalization=bn, dropout=dropout)
            model_aux = BaseNetAux(batch_normalization=bn, dropout=dropout)
            model_ws = BaseNetWeightShare(batch_normalization=bn, dropout=dropout)
            model_ws_aux = BaseNetWeightShareAux(batch_normalization=bn, dropout=dropout)
            
            train_model(model_base, train_input, train_target, mini_batch_size = batch_size, nb_epochs=25, lr=lr)
            nb_errors_base = compute_nb_errors(model_base, test_input, test_target, mini_batch_size = 25)
            model_base_mean[param_combo].append(1-nb_errors_base/n)

            train_model_with_aux_loss(model_aux, train_input, train_target, train_classes, mini_batch_size = batch_size, nb_epochs=25, lr=lr)
            nb_errors_aux = compute_nb_errors_with_aux_loss(model_aux, test_input, test_target, mini_batch_size = 25)
            model_aux_mean[param_combo].append(1-nb_errors_aux/n)

            train_model(model_ws, train_input, train_target, mini_batch_size = batch_size, nb_epochs=25, lr=lr)
            nb_errors_ws = compute_nb_errors(model_ws, test_input, test_target, mini_batch_size = 25)
            model_ws_mean[param_combo].append(1-nb_errors_ws/n)

            train_model_with_aux_loss(model_base, train_input, train_target, train_classes, mini_batch_size = batch_size, nb_epochs=25, lr=lr)
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

        # Return means and stadard deviations for each model
        return model_base_mean, model_base_std, model_aux_mean, model_aux_std, model_ws_mean, model_ws_std, model_ws_aux_mean, model_ws_aux_std




