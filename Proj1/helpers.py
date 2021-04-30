import torch
from torch import nn
from torch import optim

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
    for b in range(0, test_input.size(0), mini_batch_size):
        output = model(test_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = output.max(1)
        for k in range(mini_batch_size):
            if test_target[b + k] != predicted_classes[k]:
                nb_errors = nb_errors + 1
    return nb_errors

def compute_nb_errors_with_aux_loss(model, test_input, test_target, mini_batch_size = 50):
    nb_errors = 0
    for b in range(0, test_input.size(0), mini_batch_size):
        output_bool, output_digit1, output_digit2 = model(test_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = output_bool.max(1)
        for k in range(mini_batch_size):
            if test_target[b + k] != predicted_classes[k]:
                nb_errors = nb_errors + 1
    return nb_errors
